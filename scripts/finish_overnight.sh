#!/usr/bin/env bash
# Tighter finishing sequence after the resume reruns.
#
# Drops geoVI K=8 (long tail), prioritizes:
#   1. Backfill MGVI K=128 N=8192 (timed out at 40 min before; retry at 2 h).
#   2. Fill geoVI K=1,2,4 missing cells.
#   3. Basic large-N at K=1 (N=16384,32768).
#   4. Rich-obs K=1 sweep (the σ_PSD constraint figure).
#   5. Final renders.

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

# Skipping MGVI K=128 N=8192 backfill and any N>=16384 cells — both fail
# with XLA compile hangs at large shapes. User is preparing a JIT-compile
# change; revisit those once that lands.
#
# 1. Fill geoVI K=1,2,4 cells (idempotent — cached cells skip).
phase B_geovi_K1_K2_K4 \
    env JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --geovi-only --ks 1,2,4 \
        --ns 4,8,16,32,64,128,256,512,1024,2048,4096,8192

# 2. Rich-obs K=1 sweep for σ_PSD constraint figure (N capped at 8192).
phase D_rich_obs_constraint_run_K1 \
    env JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --rich-obs --noise-frac 0.05 --ks 1 \
        --ns 4,8,16,32,64,128,256,512,1024,2048,4096,8192

# 3. Final renders.
phase E_render_basic .venv/bin/python analysis/render_vi_scaling.py
phase F_render_rich  .venv/bin/python analysis/render_vi_scaling.py --rich-obs

echo "[$(stamp)] === finish_overnight pipeline complete ==="
