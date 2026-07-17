#!/usr/bin/env bash
# Watch all `python` processes and SIGKILL any whose RSS exceeds LIMIT_GB.
#
# Different from scripts/run_with_oom_monitor.sh, which wraps a single
# command. This one runs as a long-lived daemon to protect the laptop
# from runaway JIT compiles in *any* shell session.
#
# Usage:
#   scripts/python_oom_guard.sh                  # 10 GB default, 3 s poll
#   LIMIT_GB=8 INTERVAL_SEC=2 scripts/python_oom_guard.sh
#   scripts/python_oom_guard.sh &                # background
#
# Stop with `kill <pid>` or Ctrl-C. Killed processes are logged to
# /tmp/python_oom_guard.log.

set -u

LIMIT_GB="${LIMIT_GB:-10}"
INTERVAL_SEC="${INTERVAL_SEC:-3}"
LOG="${LOG:-/tmp/python_oom_guard.log}"
# Optional: regex of comm fields to *exclude* (e.g. avoid killing your IDE).
EXCLUDE_RE="${EXCLUDE_RE:-^$}"

LIMIT_KB=$(( LIMIT_GB * 1024 * 1024 ))
SELF_PID=$$

echo "[guard $(date +%H:%M:%S)] start pid=$SELF_PID limit=${LIMIT_GB}GB interval=${INTERVAL_SEC}s log=$LOG" | tee -a "$LOG"

trap 'echo "[guard $(date +%H:%M:%S)] stop pid=$SELF_PID" | tee -a "$LOG"; exit 0' INT TERM

while true; do
    # ps columns: pid, rss(KB), args
    while IFS= read -r line; do
        # Strip leading whitespace ps adds for right-justified pid/rss.
        line="${line#"${line%%[![:space:]]*}"}"
        pid="${line%% *}"
        rest="${line#* }"
        rest="${rest#"${rest%%[![:space:]]*}"}"
        rss_kb="${rest%% *}"
        rest="${rest#* }"
        args="${rest#"${rest%%[![:space:]]*}"}"

        # Filter to python interpreters (matches python, python3, python3.12,
        # .venv/bin/python, etc.) by basename of argv[0]. Do NOT filter on the
        # ps `comm` column: macOS truncates comm to 16 chars unless it is the
        # last column, so `.venv/bin/python` under a long user path becomes
        # `/Users/<user>` and silently never matches.
        exe="${args%% *}"
        base="${exe##*/}"
        case "$base" in
            python|python[0-9]*|pypy|pypy[0-9]*) ;;
            *) continue ;;
        esac

        # Never kill ourselves or our shell/parent.
        if [[ "$pid" == "$SELF_PID" || "$pid" == "$PPID" ]]; then continue; fi

        # User-supplied exclude regex (matched against args).
        if [[ -n "$EXCLUDE_RE" && "$args" =~ $EXCLUDE_RE ]]; then continue; fi

        if (( rss_kb > LIMIT_KB )); then
            rss_gb=$(awk -v k="$rss_kb" 'BEGIN{printf "%.2f", k/1048576}')
            echo "[guard $(date +%H:%M:%S)] SIGKILL pid=$pid rss=${rss_gb}GB cmd=${args:0:140}" | tee -a "$LOG" >&2
            kill -KILL "$pid" 2>/dev/null
        fi
    done < <(ps -axo pid=,rss=,args=)

    sleep "$INTERVAL_SEC"
done
