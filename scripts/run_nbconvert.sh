#!/usr/bin/env bash
# Execute one notebook via nbconvert, capturing wall time + peak RSS.
#
# Usage: scripts/run_nbconvert.sh <notebook_stem>
#   e.g. scripts/run_nbconvert.sh 05_fitting_photometry
#
# Writes full log to /tmp/nbconvert_<stem>.log and prints a one-line
# summary on stdout. Used by docs/dev/benchmarks/*_notebook_renewal.md.
set -u
nb=${1:?nb stem required (e.g. 05_fitting_photometry)}
log=/tmp/nbconvert_${nb}.log
echo "=== START $nb $(date +%H:%M:%S) ===" > "$log"
/usr/bin/time -l env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 \
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  "notebooks/${nb}.ipynb" >> "$log" 2>&1
ec=$?
echo "=== END $nb (exit $ec) $(date +%H:%M:%S) ===" >> "$log"
real_t=$(grep -E "^[ ]+[0-9.]+ real" "$log" | tail -1 | awk '{print $1}')
peak_b=$(grep "maximum resident set size" "$log" | tail -1 | awk '{print $1}')
peak_gb=$(awk -v b="$peak_b" 'BEGIN{printf "%.2f", b/1073741824}')
printf "%-30s exit=%d  wall=%ss  peak_rss=%sGB\n" "$nb" "$ec" "${real_t:-?}" "$peak_gb"
