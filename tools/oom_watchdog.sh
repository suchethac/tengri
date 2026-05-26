#!/bin/bash
# OOM watchdog — kills python processes exceeding a memory limit.
# Polls every POLL_SEC, logs to stderr each tick.
#
# Usage:
#   tools/oom_watchdog.sh [LIMIT_GB] [POLL_SEC]
# Defaults: 30 GB, 1 second.

LIMIT_GB="${1:-30}"
POLL_SEC="${2:-1}"
LIMIT_KB=$((LIMIT_GB * 1024 * 1024))

echo "[$(date +%H:%M:%S)] watchdog up: limit=${LIMIT_GB} GB, poll=${POLL_SEC}s, pid=$$" >&2

while true; do
    # Get every python proc with its RSS (kB) and pid.
    peak=$(ps -axo pid=,rss=,comm= |
           awk -v lim="$LIMIT_KB" '
             $3 ~ /[Pp]ython/ {
               procs[NR] = $1 " " $2 " " $3
               if ($2 > max_rss) { max_rss = $2; max_proc = $1 " " $2 " " $3 }
               if ($2 > lim) { print "KILL", $1, $2/1048576 "GB", $3 > "/dev/stderr"; system("kill -9 " $1) }
             }
             END {
               if (max_proc) print max_proc
             }')

    if [ -n "$peak" ]; then
        pid=$(echo "$peak" | awk '{print $1}')
        rss_kb=$(echo "$peak" | awk '{print $2}')
        cmd=$(echo "$peak" | awk '{print $3}')
        rss_gb=$(awk "BEGIN{printf \"%.2f\", $rss_kb/1048576}")
        echo "[$(date +%H:%M:%S)] peak python: pid=$pid  rss=${rss_gb} GB  ($cmd)" >&2
    fi
    sleep "$POLL_SEC"
done
