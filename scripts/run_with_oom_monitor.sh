#!/usr/bin/env bash
# Run a command and SIGKILL its entire process tree if total RSS exceeds LIMIT_GB.
# Used to catch jetsam-killer scenarios where the kernel dies under memory
# pressure before any single process hits its own RLIMIT.
#
# Usage:
#   LIMIT_GB=20 scripts/run_with_oom_monitor.sh -- <cmd> [args...]
#
# Logs samples to /tmp/nb_mem.log and peak RSS to /tmp/nb_peak.txt.

set -u

LIMIT_GB="${LIMIT_GB:-20}"
INTERVAL_SEC="${INTERVAL_SEC:-5}"
LOG="${LOG:-/tmp/nb_mem.log}"
PEAK="${PEAK:-/tmp/nb_peak.txt}"

# Drop the optional leading "--"
if [[ "${1:-}" == "--" ]]; then shift; fi

if [[ $# -eq 0 ]]; then
    echo "usage: LIMIT_GB=20 $0 -- <cmd> [args...]" >&2
    exit 64
fi

LIMIT_BYTES=$(( LIMIT_GB * 1024 * 1024 * 1024 ))

# Start the target in its own process group so we can kill the whole tree.
# macOS lacks `setsid`; use a Python launcher that calls os.setsid before exec.
python3 -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@" &
TARGET_PID=$!
# The new session leader's PGID equals its PID.
TARGET_PGID=$TARGET_PID

: > "$LOG"
echo "0" > "$PEAK"
echo "[monitor] pid=$TARGET_PID pgid=$TARGET_PGID limit=${LIMIT_GB}GB interval=${INTERVAL_SEC}s" | tee -a "$LOG"

peak=0
while kill -0 "$TARGET_PID" 2>/dev/null; do
    # Walk the process tree from TARGET_PID and sum RSS (KB) of every
    # descendant. We rely on pid+ppid rather than pgid because some tools
    # (e.g. jupyter) reset their kernel's process group, hiding it from a
    # plain pgid query.
    total_kb=$(python3 - "$TARGET_PID" <<'PY'
import subprocess, sys, collections
root = int(sys.argv[1])
out = subprocess.run(["ps","-A","-o","pid=,ppid=,rss="], capture_output=True, text=True).stdout
children = collections.defaultdict(list); rss = {}
for line in out.splitlines():
    parts = line.split()
    if len(parts) < 3: continue
    pid, ppid, r = int(parts[0]), int(parts[1]), int(parts[2])
    children[ppid].append(pid); rss[pid] = r
total = 0; stack = [root]
while stack:
    p = stack.pop()
    total += rss.get(p, 0)
    stack.extend(children.get(p, []))
print(total)
PY
)
    total_bytes=$(( total_kb * 1024 ))
    total_gb=$(awk -v b="$total_bytes" 'BEGIN{printf "%.2f", b/1073741824}')
    ts=$(date +%H:%M:%S)
    echo "$ts rss=${total_gb}GB" >> "$LOG"
    if (( total_bytes > peak )); then
        peak=$total_bytes
        printf "%.2f\n" "$(awk -v b="$peak" 'BEGIN{print b/1073741824}')" > "$PEAK"
    fi
    if (( total_bytes > LIMIT_BYTES )); then
        echo "[monitor] RSS ${total_gb}GB exceeded limit ${LIMIT_GB}GB — SIGKILL pgid=$TARGET_PGID" | tee -a "$LOG" >&2
        kill -KILL -- -"$TARGET_PGID" 2>/dev/null
        wait "$TARGET_PID" 2>/dev/null
        exit 137
    fi
    sleep "$INTERVAL_SEC"
done

wait "$TARGET_PID"
RC=$?
peak_gb=$(cat "$PEAK")
echo "[monitor] target exited rc=$RC peak_rss=${peak_gb}GB" | tee -a "$LOG"
exit "$RC"
