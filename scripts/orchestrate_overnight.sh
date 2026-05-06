#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight orchestrator for the population-VI scaling benchmark.
#
# Sequential pipeline. Each phase runs `scripts/benchmark_vi_xlarge.py` (which
# spawns one fresh Python subprocess per (method, N, K) cell so peak-RSS is
# clean), idempotently skipping cells already cached in JSON.
#
# Phases (in order)
# ─────────────────
#   1_mgvi_K_sweep_basic_obs
#       MGVI (native_vi_linear) full K sweep, K∈{1,2,4,8,16,32,64,128},
#       N∈{4..8192}, SDSS-only 5-band photometry. The linear engine is cheap
#       per iter, so we can afford the full K range.
#
#   2_geovi_K_sweep_basic_obs
#       geoVI (native_vi_nonlinear) restricted K sweep, K∈{1,2,4,8}, same N
#       grid and observation. geoVI is ~3× slower per iter than MGVI, so we
#       cap K to keep the run finite overnight.
#
#   3_basic_obs_large_N_K1
#       Both methods at K=1, N∈{16384, 32768} only. Probes peak memory at
#       extreme N where chunk parallelism is irrelevant. Timeout bumped to
#       2 h per cell because XLA compile blows up at large shapes.
#
#   4_rich_obs_constraint_run_K1
#       10-band photometry (GALEX FUV/NUV + SDSS ugriz + 2MASS JHKs) at 5%
#       noise, K=1, both methods, all N. This is the run where σ_PSD and
#       τ_PSD posteriors actually tighten with N — the figure's bottom row.
#       Writes to data/vi_scaling_benchmark_rich.json (separate from basic).
#
#   5_render_basic_figure
#       Render analysis/figures/vi_scaling.{png,pdf} from basic JSON.
#
#   6_render_rich_figure
#       Render analysis/figures/vi_scaling_rich.{png,pdf} from rich JSON.
#
# Notes
# ─────
#   * `set -u` only — a single phase OOM/timeout must not kill the pipeline.
#   * Per-phase logs land in /tmp/vi_overnight/<phase_name>.log.
#   * The benchmark driver auto-skips cached cells (`--force` to override).
#   * `VI_BENCHMARK_TIMEOUT` (seconds) overrides the per-cell subprocess
#     timeout — defaults to 2400 (40 min) inside the driver.
#   * N capped at 8192 for Phases 1–2 (XLA compile time at N≥16384 is
#     pathological); Phase 3 covers those large-N cells separately.
# ─────────────────────────────────────────────────────────────────────────────

set -u
cd "$(dirname "$0")/.."

mkdir -p /tmp/vi_overnight
PHASE_LOG () { echo "/tmp/vi_overnight/$1.log"; }
stamp () { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

run_phase () {
    local name="$1"; shift
    local log
    log=$(PHASE_LOG "$name")
    echo "[$(stamp)] >>> phase start: $name"
    echo "[$(stamp)] cmd: $*" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
    echo "[$(stamp)] <<< phase done:  $name"
}

# ── Phase 1: MGVI K sweep on SDSS-only photometry ──────────────────────────
run_phase 1_mgvi_K_sweep_basic_obs \
    env JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --ns 4,8,16,32,64,128,256,512,1024,2048,4096,8192 \
        --ks 1,2,4,8,16,32,64,128 \
        --linear-only

# ── Phase 2: geoVI restricted K sweep on SDSS-only photometry ──────────────
run_phase 2_geovi_K_sweep_basic_obs \
    env JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --ns 4,8,16,32,64,128,256,512,1024,2048,4096,8192 \
        --ks 1,2,4,8 \
        --geovi-only

# ── Phase 3: basic observation at large N (K=1, both methods) ──────────────
run_phase 3_basic_obs_large_N_K1 \
    env JAX_PLATFORMS=cpu VI_BENCHMARK_TIMEOUT=7200 \
        .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --ns 16384,32768 --ks 1

# ── Phase 4: rich-obs constraint run (FUV/NUV+SDSS+JHK, 5% noise) ──────────
run_phase 4_rich_obs_constraint_run_K1 \
    env JAX_PLATFORMS=cpu VI_BENCHMARK_TIMEOUT=7200 \
        .venv/bin/python scripts/benchmark_vi_xlarge.py \
        --rich-obs --noise-frac 0.05 --ks 1

# ── Phase 5: final renders ─────────────────────────────────────────────────
run_phase 5_render_basic_figure \
    .venv/bin/python analysis/render_vi_scaling.py

run_phase 6_render_rich_figure \
    .venv/bin/python analysis/render_vi_scaling.py --rich-obs

echo "[$(stamp)] === overnight pipeline complete ==="
