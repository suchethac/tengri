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
#   TOTAL_LIMIT_GB   python RSS budget [GB]; default 75% of physical RAM
#   AVAIL_PCT_MIN    trip when available memory falls below this % of RAM;
#                    default 10. 0 disables the availability trigger. This is
#                    the pressure signal that actually predicts death: it is
#                    what the kernel is short of and cannot reclaim.
#   SWAP_MAX_GB      trip when swap in use exceeds this [GB]; default 0 =
#                    DISABLED, and you almost certainly want to leave it off
#                    on macOS. Swap there is not an emergency reading: the
#                    kernel grows the swap file on demand and does not shrink
#                    it, so a box merely up for a few days sits permanently
#                    above any fixed threshold. Enabling it made the guard
#                    want to fire on every tick forever (measured 2026-08-09:
#                    swap pinned at 20+ GB for hours while available memory
#                    stayed a healthy 22-29%), which would have killed 8 GB
#                    out of every pytest run 60 s after it started. Swap is
#                    logged every tick regardless, for diagnosis.
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

RAM_KB=$(detect_ram_kb)
RAM_GB=$(( RAM_KB / 1048576 ))
TOTAL_LIMIT_GB="${TOTAL_LIMIT_GB:-$(( RAM_GB * 3 / 4 ))}"
AVAIL_PCT_MIN="${AVAIL_PCT_MIN:-10}"
SWAP_MAX_GB="${SWAP_MAX_GB:-0}"
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

case "$TOTAL_LIMIT_GB$AVAIL_PCT_MIN$SWAP_MAX_GB$SHED_GB$PRESSURE_MIN_PYTHON_GB$COOLDOWN_SEC$INTERVAL_SEC$MIN_KILL_MB" in
    *[!0-9]*)
        echo "error: TOTAL_LIMIT_GB, AVAIL_PCT_MIN, SWAP_MAX_GB, SHED_GB," \
             "PRESSURE_MIN_PYTHON_GB, COOLDOWN_SEC, INTERVAL_SEC, MIN_KILL_MB" \
             "must be non-negative integers" >&2
        exit 64 ;;
esac

LIMIT_KB=$(( TOTAL_LIMIT_GB * 1024 * 1024 ))
MIN_KILL_KB=$(( MIN_KILL_MB * 1024 ))
SHED_KB=$(( SHED_GB * 1024 * 1024 ))
SWAP_MAX_KB=$(( SWAP_MAX_GB * 1024 * 1024 ))
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
        read -r AVAIL_KB SWAP_USED_KB < "$PRESSURE_FIXTURE"
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
     "avail_min=${AVAIL_PCT_MIN}% swap_max=${SWAP_MAX_GB}GB shed=${SHED_GB}GB" \
     "pressure_min_python=${PRESSURE_MIN_PYTHON_GB}GB cooldown=${COOLDOWN_SEC}s" \
     "interval=${INTERVAL_SEC}s min_kill=${MIN_KILL_MB}MB dry_run=${DRY_RUN} log=$LOG" | tee -a "$LOG"

trap 'echo "[total-guard $(date +%H:%M:%S)] stop pid=$SELF_PID" | tee -a "$LOG"; exit 0' INT TERM

tick=0
last_kill_epoch=0
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

    echo "$(date +%H:%M:%S) total=$(kb_to_gb "$total_kb")GB n=$n_procs limit=${TOTAL_LIMIT_GB}GB" \
         "avail=${avail_pct}% swap=$(kb_to_gb "$SWAP_USED_KB")GB" >> "$LOG"

    # Decide whether to shed, and by how much. Both triggers reduce to a
    # single "shed this many KB" target so victim selection has one code path.
    reason=""
    target_kb=0
    if (( total_kb > LIMIT_KB )); then
        reason="sum-rss $(kb_to_gb "$total_kb")GB > ${TOTAL_LIMIT_GB}GB"
        target_kb=$(( total_kb - LIMIT_KB ))
    fi
    pressure=""
    if (( AVAIL_PCT_MIN > 0 && avail_pct < AVAIL_PCT_MIN )); then
        pressure="available ${avail_pct}% < ${AVAIL_PCT_MIN}%"
    fi
    if (( SWAP_MAX_KB > 0 && SWAP_USED_KB > SWAP_MAX_KB )); then
        pressure="${pressure:+$pressure; }swap $(kb_to_gb "$SWAP_USED_KB")GB > ${SWAP_MAX_GB}GB"
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

    now_epoch=$(date +%s)
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
