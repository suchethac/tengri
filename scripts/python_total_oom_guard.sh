#!/usr/bin/env bash
# Watch the machine-wide TOTAL RSS of all `python` processes and, when the
# sum exceeds TOTAL_LIMIT_GB, SIGKILL processes largest-first until the
# projected total is back under the limit.
#
# Different from the two existing guards, which are both blind to this
# failure mode:
#   - scripts/run_with_oom_monitor.sh   wraps ONE command's process tree
#   - scripts/python_oom_guard.sh       checks EACH process individually
# Many ~3 GB workers (pytest-xdist, parallel sessions, orphaned notebook
# kernels) can each stay under a per-process limit yet sum past physical
# RAM and trip the macOS jetsam killer. This daemon watches the global sum
# and sheds load starting from the most memory-hungry python process.
#
# Usage:
#   scripts/python_total_oom_guard.sh                  # limit = 75% of RAM, 2 s poll
#   TOTAL_LIMIT_GB=30 INTERVAL_SEC=1 scripts/python_total_oom_guard.sh
#   DRY_RUN=1 scripts/python_total_oom_guard.sh        # log the kill plan, kill nothing
#   scripts/python_total_oom_guard.sh &                # background daemon
#
# Config (env):
#   TOTAL_LIMIT_GB   global python RSS budget [GB]; default 75% of physical RAM
#   INTERVAL_SEC     poll interval [s]; default 2 (JAX can allocate GBs within 5 s)
#   MIN_KILL_MB      never kill a python process smaller than this [MB]; default 512
#                    (protects language servers, idle kernels, IDE helpers)
#   ONLY_RE          only consider processes whose args match this regex; default '.'
#   EXCLUDE_RE       never kill processes whose args match this regex; default '^$'
#   LOG              log file; default /tmp/python_total_oom_guard.log
#   DRY_RUN          1 = log what would be killed, kill nothing; default 0
#   MAX_TICKS        exit 0 after this many polls; default 0 = run forever (smoke tests)
#
# Stop with `kill <pid>` or Ctrl-C. Every tick's total is appended to LOG,
# so after an incident the ramp-up is visible, not just the kill.

set -u

detect_ram_gb() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}'
    else
        awk '/^MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo
    fi
}

RAM_GB=$(detect_ram_gb)
TOTAL_LIMIT_GB="${TOTAL_LIMIT_GB:-$(( RAM_GB * 3 / 4 ))}"
INTERVAL_SEC="${INTERVAL_SEC:-2}"
MIN_KILL_MB="${MIN_KILL_MB:-512}"
ONLY_RE="${ONLY_RE:-.}"
EXCLUDE_RE="${EXCLUDE_RE:-^$}"
LOG="${LOG:-/tmp/python_total_oom_guard.log}"
DRY_RUN="${DRY_RUN:-0}"
MAX_TICKS="${MAX_TICKS:-0}"

case "$TOTAL_LIMIT_GB$INTERVAL_SEC$MIN_KILL_MB" in
    *[!0-9]*)
        echo "error: TOTAL_LIMIT_GB, INTERVAL_SEC, MIN_KILL_MB must be non-negative integers" >&2
        exit 64 ;;
esac

LIMIT_KB=$(( TOTAL_LIMIT_GB * 1024 * 1024 ))
MIN_KILL_KB=$(( MIN_KILL_MB * 1024 ))
SELF_PID=$$

kb_to_gb() { awk -v k="$1" 'BEGIN{printf "%.2f", k / 1048576}'; }

echo "[total-guard $(date +%H:%M:%S)] start pid=$SELF_PID limit=${TOTAL_LIMIT_GB}GB (ram=${RAM_GB}GB)" \
     "interval=${INTERVAL_SEC}s min_kill=${MIN_KILL_MB}MB dry_run=${DRY_RUN} log=$LOG" | tee -a "$LOG"

trap 'echo "[total-guard $(date +%H:%M:%S)] stop pid=$SELF_PID" | tee -a "$LOG"; exit 0' INT TERM

tick=0
while true; do
    # One ps snapshot per tick: collect every python process as a
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
    done < <(ps -axo pid=,rss=,args=)

    echo "$(date +%H:%M:%S) total=$(kb_to_gb "$total_kb")GB n=$n_procs limit=${TOTAL_LIMIT_GB}GB" >> "$LOG"

    if (( total_kb > LIMIT_KB )); then
        echo "[total-guard $(date +%H:%M:%S)] total python RSS $(kb_to_gb "$total_kb")GB" \
             "exceeds ${TOTAL_LIMIT_GB}GB across $n_procs processes — shedding largest-first" \
             | tee -a "$LOG" >&2

        # Select victims: largest RSS first, subtracting each from a
        # projected total. RSS is not reclaimed instantly after SIGKILL, so
        # re-measuring between kills would keep shooting; the projection
        # decides the whole rescue from one consistent snapshot. Selection is
        # pure shell arithmetic and the kill is one batched signal — under
        # memory pressure every fork/exec (awk, tee, date) can take ~1 s, so
        # logging waits until after the kills have landed.
        projected_kb=$total_kb
        kill_pids=""
        victims=""
        while IFS=$'\t' read -r rss_kb pid args; do
            (( projected_kb <= LIMIT_KB )) && break
            (( rss_kb < MIN_KILL_KB )) && continue
            kill_pids+=" $pid"
            victims+="${rss_kb}"$'\t'"${pid}"$'\t'"${args}"$'\n'
            projected_kb=$(( projected_kb - rss_kb ))
        done < <(printf '%s' "$candidates" | sort -t$'\t' -k1,1 -rn)

        if [[ -n "$kill_pids" && "$DRY_RUN" != "1" ]]; then
            # Intentionally unquoted: word-splits into one pid per argument.
            kill -KILL $kill_pids 2>/dev/null
        fi

        verb="SIGKILL"
        [[ "$DRY_RUN" == "1" ]] && verb="DRY_RUN would SIGKILL"
        while IFS=$'\t' read -r rss_kb pid args; do
            [[ -z "$pid" ]] && continue
            echo "[total-guard $(date +%H:%M:%S)] $verb pid=$pid rss=$(kb_to_gb "$rss_kb")GB cmd=${args:0:140}" \
                 | tee -a "$LOG" >&2
        done <<< "$victims"

        if (( projected_kb > LIMIT_KB )); then
            echo "[total-guard $(date +%H:%M:%S)] WARNING: still $(kb_to_gb "$projected_kb")GB projected" \
                 "after kill pass — remaining processes are under MIN_KILL_MB=${MIN_KILL_MB}MB or excluded" \
                 | tee -a "$LOG" >&2
        fi
    fi

    tick=$(( tick + 1 ))
    if (( MAX_TICKS > 0 && tick >= MAX_TICKS )); then
        echo "[total-guard $(date +%H:%M:%S)] exit after $tick tick(s) (MAX_TICKS)" | tee -a "$LOG"
        exit 0
    fi
    sleep "$INTERVAL_SEC"
done
