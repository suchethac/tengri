#!/usr/bin/env bash
# Watch machine-wide python memory and SIGKILL processes largest-first when
# the machine is in trouble.
#
# Two independent triggers, because either one alone has a blind spot:
#
#   1. SUM-RSS   total RSS across python processes > TOTAL_LIMIT_GB.
#                Catches the classic "30 workers x 4 GB" blow-up.
#   2. PRESSURE  the OS itself is short of memory (available memory below
#                AVAIL_PCT_MIN). Catches the case the sum-RSS trigger
#                structurally cannot: a limit set above what the workload
#                ever reaches, while the box suffocates anyway. Observed
#                2026-08-09 on a 48 GB machine — python sum-RSS sat at
#                12-16 GB against a 30 GB limit and never tripped.
#
# Different from the two per-scope guards, which are both blind to the
# machine-wide failure mode:
#   - scripts/run_with_oom_monitor.sh   wraps ONE command's process tree
#   - scripts/python_oom_guard.sh       checks EACH process individually
#
# Usage:
#   scripts/python_total_oom_guard.sh                  # limit = 75% of RAM, 2 s poll
#   TOTAL_LIMIT_GB=30 INTERVAL_SEC=1 scripts/python_total_oom_guard.sh
#   DRY_RUN=1 scripts/python_total_oom_guard.sh        # log the kill plan, kill nothing
#   scripts/python_total_oom_guard.sh &                # background daemon
#
# For a daemon that survives logout and reboot, install the LaunchAgent:
#   scripts/install_oom_guard_agent.sh
#
# Config (env):
#   TOTAL_LIMIT_GB   python RSS budget [GB]; default 75% of physical RAM.
#                    WEAKEST of the triggers, and it CANNOT be the only one:
#                    RSS counts only resident pages, so every page the kernel
#                    swaps out stops being counted and this number SHRINKS as
#                    the machine thrashes harder. Measured 2026-08-10 either
#                    side of a manual rescue, three minutes apart:
#                      34 python procs, box thrashing  -> total 12.88 GB
#                      10 python procs, box healthy    -> total 23.83 GB
#                    A 32 GB limit was therefore unreachable in exactly the
#                    condition it existed to catch. Keep it as a backstop for
#                    a genuine resident-memory runaway; do not rely on it.
#   AVAIL_PCT_MIN    HARD floor: trip whenever available memory is below this %
#                    of RAM, on its own. Default 10. 0 disables.
#   AVAIL_PCT_SOFT   SOFT threshold, default 20: trips only in CONJUNCTION with
#                    swap above SWAP_MAX_GB. macOS holds available memory up
#                    *by* swapping, so during the 2026-08-10 incident it sat at
#                    17-18% for the whole event and never reached the 10% hard
#                    floor, while swap ran to 43 GB. Neither signal alone is
#                    both sensitive and quiet; their conjunction is.
#   SWAP_MAX_GB      swap-in-use level [GB] that counts as pressure; default 20.
#                    This was 0 (disabled) and that is why the guard missed the
#                    2026-08-10 event. The reasoning for disabling it was FALSE:
#                    it claimed macOS never shrinks the swap file. Measured
#                    across that incident, swap went 8.47 -> 43.23 -> 8.47 GB,
#                    reclaiming 35 GB within minutes of the pressure clearing.
#                    The earlier "chronic 20+ GB baseline" was not a baseline at
#                    all -- it was the opening hours of the same thrash. Swap is
#                    the one signal that tracked the event monotonically. Keep
#                    this well ABOVE the working baseline (~8.5 GB here), which
#                    is what makes it quiet; a threshold set near baseline is
#                    the actual trap.
#   SWAP_GROWTH_GB   trip when swap grows by this much [GB] within
#                    SWAP_GROWTH_WINDOW_SEC; default 3. Catches the ramp before
#                    the absolute level is alarming, and is immune to whatever
#                    the baseline happens to be. On 2026-08-10 swap climbed
#                    6.33 -> 17.06 GB in four minutes before any level trigger
#                    would have fired. 0 disables.
#   SWAP_GROWTH_WINDOW_SEC
#                    sliding window for SWAP_GROWTH_GB [s]; default 120
#   SHED_GB          how much RSS to shed per pressure trip [GB]; default 8
#   PRESSURE_MIN_PYTHON_GB
#                    a PRESSURE trip only sheds if python itself holds at least
#                    this much [GB]; default 8. Swap pressure is often chronic
#                    and is not always python's fault: without this gate the
#                    guard re-trips every cooldown and eventually kills every
#                    python process for no benefit, because killing 2 GB of
#                    python cannot fix a 20 GB shortfall caused by something
#                    else. The gate also self-limits the runaway: once enough
#                    has been shed, python drops below it and shedding stops.
#                    Does not apply to the sum-rss trigger, where python is by
#                    definition the problem.
#   COOLDOWN_SEC     min seconds between kill passes; default 60. Freed memory
#                    is not reflected instantly, so without this the guard
#                    re-trips on stale readings and strips the machine bare.
#   INTERVAL_SEC     poll interval [s]; default 2 (JAX can allocate GBs within 5 s)
#   MIN_KILL_MB      prefer not to kill python processes smaller than this [MB];
#                    default 0 (no floor). A floor is only a PREFERENCE: if the
#                    shed target cannot be met without going below it, the guard
#                    goes below it anyway. See the two-pass selection below.
#   ONLY_RE          only consider processes whose args match this regex; default '.'
#   EXCLUDE_RE       never kill processes whose args match this regex;
#                    default protects Apple system python from being shed
#   LOG              log file; default /tmp/python_total_oom_guard.log
#   DRY_RUN          1 = log what would be killed, kill nothing; default 0
#   MAX_TICKS        exit 0 after this many polls; default 0 = run forever (smoke tests)
#   PS_FIXTURE       test hook: read the process table from this file instead of ps
#   PRESSURE_FIXTURE test hook: read "avail_kb swap_used_kb" from this file
#
# Stop with `kill <pid>` or Ctrl-C. Every tick's total is appended to LOG,
# so after an incident the ramp-up is visible, not just the kill.

set -u

detect_ram_kb() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sysctl -n hw.memsize | awk '{printf "%d", $1 / 1024}'
    else
        awk '/^MemTotal/ {printf "%d", $2}' /proc/meminfo
    fi
}

# RAM_KB_OVERRIDE is a test hook: available-memory percentages are meaningless
# unless the denominator is fixed, so a fixture that pins avail_kb must pin RAM
# too or the same numbers mean different things on every machine.
RAM_KB="${RAM_KB_OVERRIDE:-$(detect_ram_kb)}"
RAM_GB=$(( RAM_KB / 1048576 ))
TOTAL_LIMIT_GB="${TOTAL_LIMIT_GB:-$(( RAM_GB * 3 / 4 ))}"
AVAIL_PCT_MIN="${AVAIL_PCT_MIN:-10}"
AVAIL_PCT_SOFT="${AVAIL_PCT_SOFT:-20}"
SWAP_MAX_GB="${SWAP_MAX_GB:-20}"
SWAP_GROWTH_GB="${SWAP_GROWTH_GB:-3}"
SWAP_GROWTH_WINDOW_SEC="${SWAP_GROWTH_WINDOW_SEC:-120}"
SHED_GB="${SHED_GB:-8}"
PRESSURE_MIN_PYTHON_GB="${PRESSURE_MIN_PYTHON_GB:-8}"
COOLDOWN_SEC="${COOLDOWN_SEC:-60}"
INTERVAL_SEC="${INTERVAL_SEC:-2}"
MIN_KILL_MB="${MIN_KILL_MB:-0}"
ONLY_RE="${ONLY_RE:-.}"
EXCLUDE_RE="${EXCLUDE_RE:-^/(System|usr/libexec|Applications/Xcode)}"
LOG="${LOG:-/tmp/python_total_oom_guard.log}"
DRY_RUN="${DRY_RUN:-0}"
MAX_TICKS="${MAX_TICKS:-0}"
PS_FIXTURE="${PS_FIXTURE:-}"
PRESSURE_FIXTURE="${PRESSURE_FIXTURE:-}"

case "$TOTAL_LIMIT_GB$AVAIL_PCT_MIN$AVAIL_PCT_SOFT$SWAP_MAX_GB$SWAP_GROWTH_GB$SWAP_GROWTH_WINDOW_SEC$SHED_GB$PRESSURE_MIN_PYTHON_GB$COOLDOWN_SEC$INTERVAL_SEC$MIN_KILL_MB" in
    *[!0-9]*)
        echo "error: TOTAL_LIMIT_GB, AVAIL_PCT_MIN, AVAIL_PCT_SOFT, SWAP_MAX_GB," \
             "SWAP_GROWTH_GB, SWAP_GROWTH_WINDOW_SEC, SHED_GB," \
             "PRESSURE_MIN_PYTHON_GB, COOLDOWN_SEC, INTERVAL_SEC, MIN_KILL_MB" \
             "must be non-negative integers" >&2
        exit 64 ;;
esac

LIMIT_KB=$(( TOTAL_LIMIT_GB * 1024 * 1024 ))
MIN_KILL_KB=$(( MIN_KILL_MB * 1024 ))
SHED_KB=$(( SHED_GB * 1024 * 1024 ))
SWAP_MAX_KB=$(( SWAP_MAX_GB * 1024 * 1024 ))
SWAP_GROWTH_KB=$(( SWAP_GROWTH_GB * 1024 * 1024 ))
PRESSURE_MIN_PYTHON_KB=$(( PRESSURE_MIN_PYTHON_GB * 1024 * 1024 ))
SELF_PID=$$

kb_to_gb() { awk -v k="$1" 'BEGIN{printf "%.2f", k / 1048576}'; }

# Available memory and swap-in-use, both in KB.
#
# On macOS "available" is free + inactive + speculative + purgeable: inactive
# pages are reclaimable, so counting only `Pages free` reads as a permanent
# emergency on any warm machine. Deliberately uses vm_stat/sysctl rather than
# `memory_pressure`, which samples over a window and is far too slow to run
# every INTERVAL_SEC under load.
read_pressure() {
    if [[ -n "$PRESSURE_FIXTURE" ]]; then
        # One "avail_kb swap_used_kb" line per tick, so a test can drive a ramp
        # rather than a constant -- without that, the swap-growth trigger has no
        # way to be exercised at all. Past the last line the fixture holds its
        # final value, so a one-line file behaves as a constant.
        _line=$(sed -n "$(( ${tick:-0} + 1 ))p" "$PRESSURE_FIXTURE")
        [[ -z "$_line" ]] && _line=$(tail -n 1 "$PRESSURE_FIXTURE")
        read -r AVAIL_KB SWAP_USED_KB <<< "$_line"
        return
    fi
    if [[ "$(uname)" == "Darwin" ]]; then
        AVAIL_KB=$(vm_stat | awk '
            NR==1 { for (i=1;i<=NF;i++) if ($i=="of") ps=$(i+1); next }
            /^Pages free/        { gsub(/\./,"",$3); f=$3 }
            /^Pages inactive/    { gsub(/\./,"",$3); ia=$3 }
            /^Pages speculative/ { gsub(/\./,"",$3); sp=$3 }
            /^Pages purgeable/   { gsub(/\./,"",$3); pu=$3 }
            END { printf "%d", (f+ia+sp+pu) * ps / 1024 }')
        SWAP_USED_KB=$(sysctl -n vm.swapusage | awk '
            { for (i=1;i<=NF;i++) if ($i=="used") { v=$(i+2); sub(/M$/,"",v); printf "%d", v*1024; exit } }')
    else
        AVAIL_KB=$(awk '/^MemAvailable/ {printf "%d", $2; exit}' /proc/meminfo)
        SWAP_USED_KB=$(awk '/^SwapTotal/ {t=$2} /^SwapFree/ {f=$2} END {printf "%d", t-f}' /proc/meminfo)
    fi
    # A probe that fails must not read as "plenty of memory, nothing to do".
    [[ -z "$AVAIL_KB" ]] && AVAIL_KB=$RAM_KB
    [[ -z "$SWAP_USED_KB" ]] && SWAP_USED_KB=0
}

echo "[total-guard $(date +%H:%M:%S)] start pid=$SELF_PID limit=${TOTAL_LIMIT_GB}GB (ram=${RAM_GB}GB)" \
     "avail_min=${AVAIL_PCT_MIN}% avail_soft=${AVAIL_PCT_SOFT}%" \
     "swap_max=${SWAP_MAX_GB}GB swap_growth=+${SWAP_GROWTH_GB}GB/${SWAP_GROWTH_WINDOW_SEC}s" \
     "shed=${SHED_GB}GB" \
     "pressure_min_python=${PRESSURE_MIN_PYTHON_GB}GB cooldown=${COOLDOWN_SEC}s" \
     "interval=${INTERVAL_SEC}s min_kill=${MIN_KILL_MB}MB dry_run=${DRY_RUN} log=$LOG" | tee -a "$LOG"

trap 'echo "[total-guard $(date +%H:%M:%S)] stop pid=$SELF_PID" | tee -a "$LOG"; exit 0' INT TERM

tick=0
last_kill_epoch=0
swap_hist_t=()
swap_hist_v=()
while true; do
    # One process-table snapshot per tick: collect every python process as a
    # tab-separated "rss_kb pid args" line and sum the total.
    total_kb=0
    n_procs=0
    candidates=""
    while IFS= read -r line; do
        # Strip leading whitespace ps adds for right-justified pid/rss.
        line="${line#"${line%%[![:space:]]*}"}"
        pid="${line%% *}"
        rest="${line#* }"
        rest="${rest#"${rest%%[![:space:]]*}"}"
        rss_kb="${rest%% *}"
        rest="${rest#* }"
        args="${rest#"${rest%%[![:space:]]*}"}"

        # Filter to python interpreters (python, python3.12, .venv/bin/python,
        # pypy, ...) by basename of argv[0]. Do NOT filter on the ps `comm`
        # column: macOS truncates comm to 16 chars unless it is the last
        # column, so `.venv/bin/python` under a long user path becomes
        # `/Users/<user>` and silently never matches.
        exe="${args%% *}"
        base="${exe##*/}"
        case "$base" in
            python|python[0-9]*|pypy|pypy[0-9]*) ;;
            *) continue ;;
        esac

        # Never count or kill ourselves or our parent shell.
        if [[ "$pid" == "$SELF_PID" || "$pid" == "$PPID" ]]; then continue; fi

        [[ "$args" =~ $ONLY_RE ]] || continue
        if [[ -n "$EXCLUDE_RE" && "$args" =~ $EXCLUDE_RE ]]; then continue; fi

        total_kb=$(( total_kb + rss_kb ))
        n_procs=$(( n_procs + 1 ))
        candidates+="${rss_kb}"$'\t'"${pid}"$'\t'"${args:0:180}"$'\n'
    done < <(if [[ -n "$PS_FIXTURE" ]]; then cat "$PS_FIXTURE"; else ps -axo pid=,rss=,args=; fi)

    read_pressure
    avail_pct=$(( AVAIL_KB * 100 / RAM_KB ))
    now_epoch=$(date +%s)

    # Sliding window of swap readings, so growth is measured against the oldest
    # sample still inside SWAP_GROWTH_WINDOW_SEC. A single "reference sample,
    # reset every window" would straddle boundaries and miss a climb that spans
    # two of them; keeping the samples costs ~40 entries at the default cadence.
    swap_hist_t+=("$now_epoch")
    swap_hist_v+=("$SWAP_USED_KB")
    while (( ${#swap_hist_t[@]} > 1 )) \
          && (( now_epoch - swap_hist_t[0] > SWAP_GROWTH_WINDOW_SEC )); do
        swap_hist_t=("${swap_hist_t[@]:1}")
        swap_hist_v=("${swap_hist_v[@]:1}")
    done
    swap_growth_kb=$(( SWAP_USED_KB - swap_hist_v[0] ))
    swap_growth_span=$(( now_epoch - swap_hist_t[0] ))
    (( swap_growth_kb < 0 )) && swap_growth_kb=0

    echo "$(date +%H:%M:%S) total=$(kb_to_gb "$total_kb")GB n=$n_procs limit=${TOTAL_LIMIT_GB}GB" \
         "avail=${avail_pct}% swap=$(kb_to_gb "$SWAP_USED_KB")GB" \
         "swap_d=+$(kb_to_gb "$swap_growth_kb")GB/${swap_growth_span}s" >> "$LOG"

    # Decide whether to shed, and by how much. Both triggers reduce to a
    # single "shed this many KB" target so victim selection has one code path.
    reason=""
    target_kb=0
    if (( total_kb > LIMIT_KB )); then
        reason="sum-rss $(kb_to_gb "$total_kb")GB > ${TOTAL_LIMIT_GB}GB"
        target_kb=$(( total_kb - LIMIT_KB ))
    fi
    pressure=""

    # Hard floor: available memory alone, no corroboration needed.
    if (( AVAIL_PCT_MIN > 0 && avail_pct < AVAIL_PCT_MIN )); then
        pressure="available ${avail_pct}% < ${AVAIL_PCT_MIN}%"
    fi

    # Conjunction: neither of these is both sensitive and quiet on its own.
    # macOS keeps available memory up *by* swapping, so on 2026-08-10 avail sat
    # at 17-18% for the entire event and never reached the 10% hard floor --
    # while swap ran to 43 GB. Requiring both keeps a merely-busy box silent.
    if (( AVAIL_PCT_SOFT > 0 && SWAP_MAX_KB > 0 )) \
       && (( avail_pct < AVAIL_PCT_SOFT && SWAP_USED_KB > SWAP_MAX_KB )); then
        pressure="${pressure:+$pressure; }available ${avail_pct}% < ${AVAIL_PCT_SOFT}%"
        pressure="$pressure with swap $(kb_to_gb "$SWAP_USED_KB")GB > ${SWAP_MAX_GB}GB"
    fi

    # Rate: swap climbing fast is an emergency at ANY baseline, which is what
    # makes this the one trigger that cannot be defeated by a box whose normal
    # swap happens to sit high.
    if (( SWAP_GROWTH_KB > 0 && swap_growth_kb > SWAP_GROWTH_KB )); then
        pressure="${pressure:+$pressure; }swap +$(kb_to_gb "$swap_growth_kb")GB"
        pressure="$pressure in ${swap_growth_span}s"
    fi
    if [[ -n "$pressure" ]]; then
        if (( total_kb >= PRESSURE_MIN_PYTHON_KB )); then
            reason="${reason:+$reason; }$pressure"
            (( target_kb < SHED_KB )) && target_kb=$SHED_KB
        else
            # Shedding here would be superstition: python is not holding enough
            # for its death to fix the shortfall.
            echo "$(date +%H:%M:%S) pressure ($pressure) but python holds only" \
                 "$(kb_to_gb "$total_kb")GB < ${PRESSURE_MIN_PYTHON_GB}GB — not shedding," \
                 "the memory is elsewhere" >> "$LOG"
        fi
    fi

    # now_epoch was taken with the swap sample above, so the cooldown is
    # measured against the same instant the trigger was evaluated at.
    if [[ -n "$reason" ]] && (( now_epoch - last_kill_epoch < COOLDOWN_SEC )); then
        echo "[total-guard $(date +%H:%M:%S)] would shed ($reason) but cooling down" \
             "$(( COOLDOWN_SEC - (now_epoch - last_kill_epoch) ))s — freed memory lags the kill" >> "$LOG"
        reason=""
    fi

    if [[ -n "$reason" ]]; then
        echo "[total-guard $(date +%H:%M:%S)] TRIP: $reason — shedding" \
             "$(kb_to_gb "$target_kb")GB largest-first across $n_procs python processes" \
             | tee -a "$LOG" >&2

        # Select victims largest-first until the shed target is met.
        #
        # Two passes. Pass 1 respects MIN_KILL_KB, which exists to spare
        # language servers and idle kernels. Pass 2 runs only if pass 1 could
        # not reach the target and ignores the floor entirely.
        #
        # Pass 2 is the whole point: a floor is a preference, never a veto. A
        # single-pass guard with MIN_KILL_MB=512 facing 200 workers of 300 MB
        # selects NOTHING and logs a warning while the machine dies. Measured
        # 2026-08-09: 66% of python RSS sat below the 512 MB floor.
        #
        # RSS is not reclaimed instantly after SIGKILL, so re-measuring
        # between kills would keep shooting; one consistent snapshot decides
        # the whole rescue. Selection is pure shell arithmetic and the kill is
        # one batched signal — under memory pressure every fork/exec (awk,
        # tee, date) can take ~1 s, so logging waits until the kills land.
        sorted=$(printf '%s' "$candidates" | sort -t$'\t' -k1,1 -rn)
        freed_kb=0
        kill_pids=""
        victims=""
        chosen=""
        for pass in 1 2; do
            (( freed_kb >= target_kb )) && break
            while IFS=$'\t' read -r rss_kb pid args; do
                [[ -z "$pid" ]] && continue
                (( freed_kb >= target_kb )) && break
                case " $chosen " in *" $pid "*) continue ;; esac
                if (( pass == 1 )) && (( rss_kb < MIN_KILL_KB )); then continue; fi
                chosen+=" $pid"
                kill_pids+=" $pid"
                victims+="${rss_kb}"$'\t'"${pid}"$'\t'"${args}"$'\n'
                freed_kb=$(( freed_kb + rss_kb ))
            done <<< "$sorted"
        done

        if [[ -n "$kill_pids" && "$DRY_RUN" != "1" ]]; then
            # Intentionally unquoted: word-splits into one pid per argument.
            kill -KILL $kill_pids 2>/dev/null
        fi
        last_kill_epoch=$(date +%s)

        verb="SIGKILL"
        [[ "$DRY_RUN" == "1" ]] && verb="DRY_RUN would SIGKILL"
        while IFS=$'\t' read -r rss_kb pid args; do
            [[ -z "$pid" ]] && continue
            echo "[total-guard $(date +%H:%M:%S)] $verb pid=$pid rss=$(kb_to_gb "$rss_kb")GB cmd=${args:0:140}" \
                 | tee -a "$LOG" >&2
        done <<< "$victims"

        echo "[total-guard $(date +%H:%M:%S)] shed $(kb_to_gb "$freed_kb")GB of" \
             "$(kb_to_gb "$target_kb")GB target" | tee -a "$LOG" >&2
        if (( freed_kb < target_kb )); then
            echo "[total-guard $(date +%H:%M:%S)] WARNING: target not met — every python process" \
                 "was already killed or excluded by ONLY_RE/EXCLUDE_RE. Memory is elsewhere:" \
                 "this guard only sheds python." | tee -a "$LOG" >&2
        fi
    fi

    tick=$(( tick + 1 ))
    if (( MAX_TICKS > 0 && tick >= MAX_TICKS )); then
        echo "[total-guard $(date +%H:%M:%S)] exit after $tick tick(s) (MAX_TICKS)" | tee -a "$LOG"
        exit 0
    fi
    sleep "$INTERVAL_SEC"
done
