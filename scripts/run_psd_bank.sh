#!/usr/bin/env bash
# Run the PSD interim bank to completion on a SHARED, guarded machine.
#
# This machine runs a total-RSS guard that SIGKILLs python largest-first when
# the sum across ALL sessions crosses its ceiling (observed: 15 GB, with ~12 GB
# already held by other agents' pytest and benchmark jobs). A single-galaxy
# interim fit peaks near 2.6 GB, so the bank is a routine target -- it was
# killed twice within a minute during development, silently and with no
# traceback, which is what SIGKILL looks like from inside Python.
#
# The bank checkpoints each galaxy to its own .npz via write-then-rename and
# skips existing ones, so a kill costs only the galaxy in flight. This wrapper
# turns that property into completion: restart until every galaxy exists, with
# a short backoff so a restart does not immediately re-trip the guard.
#
# Usage:
#   scripts/run_psd_bank.sh [N] [OUTDIR] [extra args to the python script...]
#
# Progress and each restart are appended to $OUTDIR/bank_run.log.

set -u

N="${1:-256}"
OUT="${2:-psd_bank}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"
BACKOFF_SEC="${BACKOFF_SEC:-20}"

mkdir -p "$OUT"
LOG="$OUT/bank_run.log"

count_done() { ls "$OUT"/gal_*.npz 2>/dev/null | wc -l | tr -d ' '; }

echo "[bank $(date +%H:%M:%S)] start N=$N out=$OUT worktree=$WORKTREE" | tee -a "$LOG"

for ((attempt = 1; attempt <= MAX_RESTARTS; attempt++)); do
    have="$(count_done)"
    if [[ "$have" -ge "$N" ]]; then
        echo "[bank $(date +%H:%M:%S)] complete: $have/$N galaxies" | tee -a "$LOG"
        exit 0
    fi

    echo "[bank $(date +%H:%M:%S)] attempt $attempt — $have/$N done" | tee -a "$LOG"
    OMP_NUM_THREADS=1 PYTHONPATH="$WORKTREE/src:$WORKTREE" JAX_PLATFORMS=cpu \
        "$PY" -u "$WORKTREE/scripts/hierarchical_psd_fit_bank.py" \
        --n "$N" --out "$OUT" "$@" >>"$LOG" 2>&1
    rc=$?

    after="$(count_done)"
    if [[ "$rc" -ne 0 ]]; then
        # 137 = 128 + SIGKILL(9), the guard's signature. Anything else that
        # made no progress is a real error and should not be retried forever.
        echo "[bank $(date +%H:%M:%S)] exit=$rc ($have -> $after done)" | tee -a "$LOG"
        if [[ "$after" -le "$have" && "$rc" -ne 137 ]]; then
            echo "[bank $(date +%H:%M:%S)] no progress and exit!=137 — stopping, see $LOG" | tee -a "$LOG"
            exit "$rc"
        fi
    fi
    sleep "$BACKOFF_SEC"
done

echo "[bank $(date +%H:%M:%S)] gave up after $MAX_RESTARTS attempts ($(count_done)/$N)" | tee -a "$LOG"
exit 1
