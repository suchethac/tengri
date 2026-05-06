#!/usr/bin/env bash
# Spec-obs benchmark: spectroscopic forward model (3000–7500 Å rest, R≈500),
# K=1, both methods, N up to 1024. Produces the σ_PSD vs N curve where
# Hα + UV continuum + Balmer decrement actually constrain the PSD.
#
# Waits for the current finish_overnight.sh pipeline to release before
# starting (so we don't compete for CPU). After the spec sweep, renders
# the spec-mode figure.

set -u
cd "$(dirname "$0")/.."
mkdir -p /tmp/vi_overnight
stamp () { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
log_for () { echo "/tmp/vi_overnight/$1.log"; }

phase () {
    local name="$1"; shift
    local log; log=$(log_for "$name")
    echo "[$(stamp)] >>> $name"
    echo "[$(stamp)] cmd: $*" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
    echo "[$(stamp)] <<< $name done"
}

# Wait for finish_overnight.sh to release (or for there to be no benchmark
# process at all).
echo "[$(stamp)] waiting for current pipeline to finish..."
while pgrep -f "finish_overnight.sh\|benchmark_population_native --worker" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(stamp)] pipeline released — starting spec sweep."

# Spec sweep — N capped at 1024 since per-cell wall time is ~5–10× longer
# than photometry. Both methods at K=1 (memory ~13 GB at small N already;
# K>1 would push past budget).
phase G_spec_obs_constraint_run_K1 \
    env JAX_PLATFORMS=cpu VI_BENCHMARK_TIMEOUT=7200 \
        .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --spec-obs --noise-frac 0.05 --ks 1 \
        --ns 4,8,16,32,64,128,256,512,1024

phase H_render_spec \
    .venv/bin/python analysis/render_vi_scaling.py --spec-obs

echo "[$(stamp)] === spec sweep complete ==="
