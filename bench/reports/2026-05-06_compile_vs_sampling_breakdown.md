# Notebook speedup: NUTS → HMC, plus compile vs sampling breakdown

## Final wall-time delta

| Notebook | baseline (NUTS) | post (HMC) | Δ | other change |
|---|---:|---:|---:|---|
| 06_fitting_spectroscopy | 386s | **61.3s** | 6.3× | n_pix 1000→200 |
| 07_joint_photo_spec | 458s | **134s** | 3.4× | — |

For nb06, HMC scan compile+execute is **45s** (was 561s under NUTS in
the split-timing measurement). The dominant factor is HMC's fixed
leapfrog count (L=10) vs NUTS's variable treedepth (up to 2^10 = 1024
leapfrogs per sample). For well-behaved posteriors with low intrinsic
correlations, NUTS's variance and worst-case spikes dominate wall.
Switching to HMC trades NUTS's adaptive trajectory length for
predictable cost — at the price of needing a hand-tuned step count
(L=10 here, matches nb05).

Peak RSS dropped 1.7× on both notebooks (nb06: 7.0→4.0 GB, nb07: 18→11 GB).

---

# Original investigation — what's actually slow in nb06

**Date:** 2026-05-06 (continued 2026-05-07)
**Branch:** main (smart-lean cache + Phase 0 compile-tracer + Phase 1 cleanup)
**Hardware:** macOS (CPU only via `JAX_PLATFORMS=cpu`); 64 GB RAM laptop

## TL;DR

The premise behind the original plan (lower NUTS `max_num_doublings` to
shrink HLO graph; tighten `compile_signature` to dedupe compiles across
methods) was based on a hypothesis that **compile time dominates nb06's
wall**. After instrumenting and measuring:

| Phase | Duration | Fraction |
|---|---:|---:|
| MLIR lowering | 0.43s | 0.1% |
| XLA compile | 2.63s | 0.7% |
| **NUTS sampling (warmup + burnin + samples)** | **561.90s** | **97%** |
| Other (data load, MAP init, plotting) | ~15s | 2% |

**Compile is ~3 seconds, not ~360.** The benchmark doc's earlier "25s
NUTS warmup compile" was actually warmup-compile + warmup-sampling
combined. The disk cache (already enabled, `~/.cache/tengri_jax_cache`)
turns the cold compile into a 2-3s file load.

This means lowering `max_num_doublings` cannot help wall time — even a
perfect 100% reduction in compile saves at most 3s of 386s. Observed
wall delta was indeed ~0% (386 vs 385s baseline), matching this.

## What landed

| Phase | Status | Reasoning |
|---|---|---|
| 0 — Compile-event tracer (`TENGRI_LOG_COMPILES=1`) | **kept** | Just paid for itself by surfacing this finding |
| 1 — Drop `_memory_mode` from `compile_signature` | **kept** | Low-risk principled cleanup; no measurable wall effect either way |
| 2 — Lower `max_num_doublings` 10 → 8 | **reverted** | No benefit; slightly weakens stiff-posterior NUTS |
| 3 — Decoupled forward kernel (opt-in) | **deferred indefinitely** | Was premised on compile being the bottleneck; it isn't |

## Methodology

The `nuts_full_scan` call was originally instrumented with a single
`compile_timer` context that measured "compile + execute" combined,
which is what produced the misleading 371s figure. We added split
timing using the JAX AOT path:

```python
lowered = _nuts_full_scan.lower(*args)        # MLIR lowering
lowered.compile()                              # XLA compile
_nuts_full_scan(*args)                         # uses cached compile, executes
```

Runs three back-to-back timers and emits three log events. The
diagnostic code was reverted after the measurement was captured; the
production path uses the simple `compile_timer` again.

## The real bottleneck

NUTS at 1000 spectroscopy pixels does:
- 300 warmup + 100 burnin + 400 samples = 800 chain iterations
- Each iteration: variable leapfrog steps depending on tree depth
- Each leapfrog: forward + grad of `predict_spectrum` on a 1000-d output

Total forward-grad evaluations: ~10-25M depending on observed tree
depth. The cost is per-step kernel work, not graph compile.

## Real leverage points (none from the original plan)

| Lever | Mechanism | Realistic gain | Cost |
|---|---|---|---|
| **vmap over chains** (`n_chains=4`) | Compile once, sample N chains in parallel via SIMD | Per-sample wall ≈ equal, **N× ESS for free** | None (already wired, not enabled in tutorials) |
| **Faster forward kernel** | Optimize `predict_spectrum` inner loop (reduce JAX ops) | Direct 1:1 wall reduction | Engineering depth |
| **Better mass matrix → fewer leapfrog steps** | Tune warmup adaptation | Sub-linear; depends on posterior | Validation work |
| **GPU dispatch** | If hardware available | Large but device-dependent | Hardware |
| **Reduce `n_pix` for tutorial nb06** | 1000 → 500 | ~2× | Pedagogical compromise |

## What this changes for the original plan file

`~/.claude/plans/is-there-a-way-sleepy-elephant.md` proposed three
phases (1-3 above) all targeting compile time. Phases 1 and 0 (tracer)
are kept; phase 2 reverted; phase 3 was never built. The user's
question — "is there a way to reduce the cold compile time for the
inference methods and to reduce recompilations across session" — has
two honest answers now:

1. **Cold compile time is already minimized.** First-process compile
   without disk cache is ~25-35s for nb06's NUTS. Disk cache (auto-on
   via `~/.cache/tengri_jax_cache`) reduces subsequent processes to
   2-3s. Subsequent calls within a process are free via smart-lean.
   There is no within-method recompilation problem to solve.

2. **Recompilations across method switches are also not the bottleneck.**
   A MAP→HMC→NUTS sequence in nb07 does pay 3 separate compiles, but
   each is a few seconds. The 458s wall is sampling cost, not
   recompilation cost. Phase 1's cleanup (`_memory_mode` drop) is
   correct in principle but was never going to move wall time.

## Files touched (kept, post-revert)

| File | Change |
|---|---|
| `src/tengri/utils/compile_log.py` | New: opt-in event tracer |
| `scripts/analyze_compile_log.py` | New: log analyzer |
| `src/tengri/inference/jit_engine.py` | Wrapped 6 JIT sites with `instrument_first_call` (no-op when env var unset) |
| `src/tengri/inference/backends/mcmc/hmc.py` | `compile_timer` around `_hmc_full_scan`/`_hmc_chain_scan` calls |
| `src/tengri/inference/backends/mcmc/nuts.py` | `compile_timer` around `_nuts_full_scan`/`_nuts_chain_scan` calls (max_num_doublings reverted to 10) |
| `src/tengri/inference/fitter.py` | `_memory_mode` removed from `compile_signature` |
| `tests/unit/test_compile_log.py` | 18 unit tests for tracer |
| `tests/unit/test_compile_signature_invariants.py` | 4 white-box invariants pinning `_memory_mode` exclusion |
| `docs/internal/inference/compilation_diagnostics.md` | How-to for the tracer |

## Reproducing the measurement

```bash
# Clean isolated cache
mkdir -p /tmp/tengri_bench_cache /tmp/tengri_bench_out
rm -f /tmp/tengri_bench_compile.log

# Run with the diagnostic split timing temporarily added back
# (see this commit's parent for the patch). Otherwise the compile_timer
# wraps compile+execute as a single event.
TENGRI_LOG_COMPILES=1 \
TENGRI_COMPILE_LOG_PATH=/tmp/tengri_bench_compile.log \
TENGRI_JAX_CACHE_DIR=/tmp/tengri_bench_cache \
JAX_PLATFORMS=cpu \
.venv/bin/jupyter nbconvert --to notebook --execute \
  --output-dir=/tmp/tengri_bench_out \
  --ExecutePreprocessor.timeout=900 \
  notebooks/06_fitting_spectroscopy.ipynb

cat /tmp/tengri_bench_compile.log | jq .
```
