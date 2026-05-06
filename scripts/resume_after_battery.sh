#!/usr/bin/env bash
# Resume sequence after a battery-pause: force-rerun cells suspected of
# thermal/power-save throttling, then continue the overnight orchestrator
# (which is idempotent, so cached cells are skipped).

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

# ── Resume A — force rerun MGVI K=16 entire column ─────────────────────────
# (suspected thermal throttling — column sat above the K=8/K=32 envelope)
phase 0a_force_mgvi_K16 \
    env JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --force --linear-only --ks 16 \
        --ns 16,32,64,128,256,512,1024,2048,4096,8192

# ── Resume B — force rerun geoVI K=1 at the large N where the overnight
#    stretch likely throttled (N >= 1024).
phase 0b_force_geovi_K1_largeN \
    env JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --force --geovi-only --ks 1 \
        --ns 1024,2048,4096,8192

# ── Now hand off to the standard overnight orchestrator. It will skip every
#    cached (method, N, K) cell instantly and only run what's missing.
phase resume_continuation \
    bash scripts/orchestrate_overnight.sh

echo "[$(stamp)] === resume sequence complete ==="
