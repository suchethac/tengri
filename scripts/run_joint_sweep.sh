#!/usr/bin/env bash
# Joint photometry + line-flux sweep (rich 10-band + Hα/Hβ/[OIII]/[OII]
# extracted from wNE-SSP spectrum at line centers). K=1, both methods,
# N up to 1024 (memory at small N is already ~21 GB, watchdog kills any
# cell exceeding 30 GB).

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

phase J_joint_obs_constraint_run_K1 \
    env JAX_PLATFORMS=cpu VI_BENCHMARK_TIMEOUT=7200 WORKER_MEM_LIMIT_GB=30 \
        .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --joint-obs --noise-frac 0.05 --ks 1 \
        --ns 4,8,16,32,64,128,256,512,1024

phase K_render_joint \
    .venv/bin/python analysis/render_vi_scaling.py --joint-obs

echo "[$(stamp)] === joint sweep complete ==="
