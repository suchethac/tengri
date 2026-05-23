#!/usr/bin/env bash
# Kill any python process whose RSS exceeds the threshold (default 20 GB).
# Usage: ./watchdog.sh [threshold_kb]
THRESHOLD_KB=${1:-20971520}   # 20 GiB in KiB
LOG=$(dirname "$0")/watchdog.log
echo "[$(date +%H:%M:%S)] watchdog started, threshold=${THRESHOLD_KB} KiB ($((THRESHOLD_KB/1024/1024)) GiB)" >> "$LOG"
while true; do
  # macOS ps: rss in KiB
  ps -axo pid,rss,comm | awk -v T="$THRESHOLD_KB" '
    $2 > T && $3 ~ /python/ { print $1, $2, $3 }
  ' | while read pid rss cmd; do
    echo "[$(date +%H:%M:%S)] KILL pid=$pid rss_kib=$rss cmd=$cmd (>${THRESHOLD_KB} KiB)" >> "$LOG"
    kill -9 "$pid" 2>/dev/null
  done
  sleep 3
done
