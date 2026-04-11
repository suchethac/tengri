# Memory Investigation: XLA Compilation Bloat in Inference

**Date:** 2026-04-11
**Status:** RESOLVED. Peak RSS reduced from 30.8 GB to 4.95 GB (6.2x).

## Summary

Inference (MAP, VI, MCMC) consumed 20-30+ GB of RSS memory on macOS, making it impossible to run the full `00_quickstart.py` notebook or fit galaxies on typical hardware. The memory was dominated by **XLA compiled executables** (not Python heap or data arrays).

## Final Measurements (D=8 dense_basis, photometry, 5 SDSS filters)

| Stage | RSS (GB) | Time |
|-------|----------|------|
| After imports + SSP load | 0.46 | — |
| After model build | 0.59 | — |
| After MAP (200 steps) | **0.97** | 0.9s |
| After VI (6 iter, 3 samples) | **4.76** | 53s |
| After NUTS (50+50) | **4.93** | 5.1s |
| After Raytrace (100 steps) | **4.95** | 1.0s |

All four inference methods run in one process with **4.95 GB peak RSS**.

## Original Measurements (before fixes)

| Stage | RSS (GB) | Time |
|-------|----------|------|
| After imports + SSP load | 0.78 | — |
| After MAP (200 steps) | 3.33 | 4.8s |
| After VI via NIFTy `optimize_kl` (6 iter, 3 samples) | **30.8** | 71s |

`tracemalloc` (Python heap) reports only 4.25 GB peak — the remaining ~27 GB is XLA-managed memory (compiled programs, compilation buffers, device buffers).

`jax.clear_caches()` + `gc.collect()` does NOT reduce RSS — XLA memory is mmap'd and not returned to the OS.

## Root Causes Identified

### 1. NIFTy `optimize_kl` internal JIT compilation (~27 GB)

**This is the dominant issue.** NIFTy's `jft.optimize_kl()` internally creates multiple JIT scopes:
- KL divergence computation (forward model + gradient)
- Nonlinear sample drawing (forward model + JVP + VJP + Newton-CG solver)
- Linear sample drawing (forward model + VJP + CG solver)
- KL minimization (Newton-CG with line search)

Each of these scopes traces through `signal_response` → `model.predict_spectrum` → the full forward model. The forward model at D=7 with 200 spectral pixels includes:
- Parameter translation (dict unpacking, bounded transform)
- SFH computation (dense_basis: 4 params → SFR(t))
- Metallicity interpolation (linear or smooth)
- Dust attenuation (two-component Charlot & Fall)
- CSP assembly (einsum over ages × wavelengths)
- Wavelength interpolation to obs grid

NIFTy uses `kl_jit=True, residual_jit=True` (set in `jit_engine.py:938-939`) which tells NIFTy to JIT-compile these internal functions. Each compilation produces a separate XLA program containing a full copy of the forward model + its derivatives.

**We cannot control what NIFTy does internally.** The `jft.optimize_kl` function is a black box from our perspective. Options:
- Set `kl_jit=False, residual_jit=False` → slower but lower memory (untested)
- Use `vi_native` (pure JAX geoVI, no NIFTy) → fully controlled compilation
- Use `vi_nifty_fast` (jit_engine's `run_nifty_jit`) → uses `OptimizeVI.update` in a Python loop, may have different compilation behavior

### 2. Nested `@jax.jit` in kernel builders (fixed)

**Before fix:** Inner kernels (`rest_sed_kernel`, `hybrid_phot`, `fused_tier2_phot`, `fused_tier2_spec`, etc.) in `fused_kernels.py` had `@jax.jit` decorators. When called from an outer JIT scope (MAP optimizer, VI engine, NIFTy), JAX re-traces the inner kernel into the outer graph — creating duplicate copies of the forward model in the XLA program.

**Fix applied:** ALL `@jax.jit` decorators removed from `fused_kernels.py`. For standalone user-facing calls (`model.predict_photometry()`, `model.predict_spectrum()`), JIT is applied lazily via `model._jit_kernel()` cache in `model.py`.

**Impact on MAP:** Reduced peak RSS from ~4 GB to ~3.3 GB (marginal improvement).
**Impact on VI (NIFTy):** No measurable improvement — NIFTy's internal JIT compilation dominates.

### 3. Eager kernel compilation at Model init (fixed)

**Before fix:** `Model.__init__` eagerly built both compositional AND hybrid kernels, even though only one is used at runtime.

**Fix applied:** Both `_compositional` and `_hybrid` are now lazy `@property` on `SEDModel`. Built on first access, not at init.

**Impact:** Model init dropped from ~1.5s to 0.27s. Unused hybrid kernel never compiled when compositional is available.

### 4. Python for-loops unrolled into XLA graph (partially fixed)

**Line placement (`_shared.py`, `cue.py`):** `for j in range(n_lines)` loops over emission lines were replaced with vectorized JAX broadcasting. Graph shrinks from O(n_lines) to O(1) nodes.

**Filter integration (`fused_kernels.py`):** `for fw, ft in zip(filter_waves, ...)` loops still present (4 locations). Not yet vectorized. Each filter adds ~50KB to the XLA graph. With 5-10 filters, this is 250-500KB per kernel — minor compared to NIFTy's 27 GB.

### 5. jit_engine pre-compilation removed

**Before fix:** `jit_engine.py` eagerly compiled `draw_samples_jit`, `run_evi_jit`, `run_evi_geovi_jit` with dummy data at engine build time. The `draw_residuals` function alone hit the **2 GB XLA protobuf size limit** (`xla.cpu.CompilationResultProto exceeded maximum protobuf size of 2GB: 2722919490`).

**Fix applied:** Removed eager compilation (dummy calls). JIT wrappers created but compilation deferred to first real call.

**Impact:** Engine build no longer crashes with protobuf error. Compilation happens lazily when the function is first called.

## Key Fixes Applied (in order of impact)

### Fix A: Inference uses hybrid (precomputed) path (30.8 → 5.5 GB)

**The dominant fix.** All inference paths (`vi.py`, `jit_engine.py`, `geovi.py`, `common.py`, `standardized.py`, `loss_functions.py`, `hierarchical.py`) were calling `model.predict_photometry(params)` with default `mode="exact"` — the raw unfused Python pipeline. NIFTy was tracing through Python for-loops for filter integration, SFH computation, etc.

Changed all 48 inference call sites (28 photometry + 20 spectrum) to `mode="_traceable"` which uses raw un-JIT'd kernels safe inside any JIT scope. The `_traceable` mode internally picks the hybrid (precomputed SSP×filter) path when available:
- CSP einsum shrinks from `(n_age, n_wave)` = 658k elements to `(n_age, n_filters)` = 470 (1,400x smaller)
- NIFTy's 4 JIT scopes each trace a tiny precomputed einsum instead of full-wavelength SED

### Fix B: _traceable mode (raw/JIT split) — fixes tracer leaks

Kernels split into JIT'd (user-facing) and raw (inference-traceable) versions. NIFTy gets raw functions — no nested JIT, no tracer leak errors. 14 EVI/geoVI test failures fixed.

### Fix C: Filter integration vectorized — graph O(1) instead of O(n_filters)

4 `for fw, ft in zip(filter_waves, ...)` loops replaced with `jax.vmap` over padded filter arrays via `compute_flux_density_batch()`. Filters padded to common length at build time (zero-padded entries contribute zero via trans=0).

### Fix D: predict_photometry/predict_spectrum defaults changed

Default mode changed from `"exact"` to `"auto"` (picks compositional → hybrid → exact). Even without `_traceable`, any path using the default now gets a fused kernel.

### Fix E: Vectorized line placement (graph O(1) instead of O(n_lines))

`place_line_profiles` in `_shared.py` and `cue.py`: for-loops over emission lines replaced with JAX broadcasting.

### Fix F: Lazy kernel init + deferred JIT compilation

Kernels built lazily on first access. Eager pre-compilation of JIT wrappers with dummy data removed (was hitting 2.7 GB protobuf limit).

## Files Modified

| File | Changes |
|------|---------|
| `src/tengri/core/fused_kernels.py` | Raw/JIT kernel split, filter vectorization (4 for-loops → vmap), pad_filters at build time |
| `src/tengri/core/model.py` | Default mode `"exact"` → `"auto"`, lazy kernel properties, `_traceable` mode support |
| `src/tengri/models/observation/photometry.py` | Added `pad_filters()`, `compute_flux_density_batch()`, `_compute_flux_density_padded()` |
| `src/tengri/models/nebular/_shared.py` | Vectorized `place_line_profiles`: for-loops → JAX broadcasting |
| `src/tengri/models/nebular/cue.py` | Vectorized line placement loops |
| `src/tengri/inference/vi.py` | `mode="_traceable"` for photometry and spectrum |
| `src/tengri/inference/jit_engine.py` | `mode="_traceable"`, removed eager pre-compilation |
| `src/tengri/inference/geovi.py` | `mode="_traceable"` |
| `src/tengri/inference/loss_functions.py` | `mode="_traceable"` (was `approx=True`) |
| `src/tengri/inference/common.py` | `mode="_traceable"` |
| `src/tengri/inference/standardized.py` | `mode="_traceable"` |
| `src/tengri/inference/hierarchical.py` | `mode="_traceable"` |

## What Was NOT Changed

- **No physics changes.** All formulas, constants, and calculations are identical.
- **No API changes.** `model.predict_photometry()`, `model.predict_spectrum()`, `fitter.run("vi")` all work the same.
- **NIFTy internals not modified.** NIFTy's 4 JIT scopes still exist, but they now trace through a tiny hybrid kernel instead of the full-wavelength SED.

## Remaining Work (optional improvements)

**A. Disable NIFTy internal JIT (`kl_jit=False, residual_jit=False`)**
- In `jit_engine.py:934-941`, the `OptimizeVI` constructor takes `kl_jit` and `residual_jit` flags
- Setting both to `False` would make NIFTy use Python-loop CG/Newton-CG instead of JIT'd versions
- Trade: ~5-10x slower iterations but ~10x less memory
- Easy to test: change lines 937-938

**B. Use `vi_native` instead of `vi` (NIFTy)**
- `fitter.run("vi_native")` uses pure JAX geoVI from `jit_engine.py`
- **Status: NOT COMPLETE.** The native JAX VI implementation is experimental and missing features.
- The engine's `run_evi_geovi` is a single JIT scope containing hamiltonian + CG + Newton-CG
- This was hitting the 2 GB protobuf limit before (2.7 GB). With eager pre-compilation removed, it will compile on first call.
- May still hit protobuf limit. If so, need to split into smaller JIT scopes.
- Completing native VI would be the long-term solution: full control over JIT boundaries, no NIFTy dependency.

**C. Split the jit_engine into smaller JIT scopes**
- Currently `draw_residuals` is one JIT scope containing: hamiltonian (forward + grad) + metric_vec (JVP + VJP) + CG solver (while_loop)
- Could split into: (1) JIT'd hamiltonian+grad, (2) JIT'd metric_vec, (3) Python-loop CG that calls the JIT'd metric_vec
- Trade: some Python dispatch overhead per CG iteration (~50 iterations × ~100μs = 5ms)
- Benefit: each JIT scope is ~500 MB instead of ~2.7 GB

**D. Use `jax.checkpoint` (remat) on the forward model**
- `jax.checkpoint(signal_response)` tells JAX to not save forward-pass intermediates during VJP
- Recomputes forward pass during backward pass instead
- Halves peak memory per forward+backward pair
- Apply in `jit_engine.py:83` or `vi.py:804`
- Does NOT reduce XLA graph size — only reduces runtime memory

**E. Mixed precision (float32 forward model)**
- `Model(spec, ssp, forward_dtype="float32")` already supported
- Halves memory for SSP arrays and intermediate computations
- May help reduce XLA graph size (float32 instructions are simpler)

### Priority 2: Vectorize filter integration

4 locations in `fused_kernels.py` have `for fw, ft in zip(filter_waves, filter_trans):` loops. Replace with `jax.vmap` over padded filter arrays. This reduces XLA graph size by O(n_filters) but is minor compared to the NIFTy issue.

### Priority 3: XLA cache management

The `00_quickstart.py` notebook runs 6 separate inference calls (MAP + VI on parametric, MAP + VI on stochastic, NUTS, raytrace). Each accumulates XLA compiled programs that are never freed. Consider:
- Adding `jax.clear_caches()` between independent fits in notebooks
- Documenting memory expectations in the notebook
- Adding a `Fitter.clear_compiled()` convenience method

## Architecture Notes for Future Agent

### How the forward model flows through inference

```
User: fitter.run("vi")
  → Fitter._run_vi()
    → vi.run_nifty_vi(fitter, ...)
      → signal_response = lambda primals: model.predict_spectrum(primals_to_params(primals))
      → signal_response_jit = jax.jit(signal_response)
      → nifty_model = jft.Model(signal_response_jit, domain=domain)
      → likelihood = jft.Gaussian(data, noise_inv).amend(nifty_model)
      → jft.optimize_kl(likelihood, ...)  ← THIS IS WHERE 27 GB HAPPENS
        → [NIFTy internal: creates KL JIT, residual JIT, sample JIT]
        → [Each traces through signal_response_jit → model.predict_spectrum → rest_sed_kernel → nonstell_fn]
```

### Key files

- `src/tengri/inference/vi.py` — `run_nifty_vi()` (lines 683+), `run_nifty_fast_vi()` (lines 471+)
- `src/tengri/inference/jit_engine.py` — `build_jit_engine()` (native JAX engine), `run_nifty_jit()` (NIFTy tight loop)
- `src/tengri/core/fused_kernels.py` — `build_fused_rest_sed()`, `build_fused_tier2_photometry()`, `build_fused_tier2_spectrum()`
- `src/tengri/core/model.py` — `_predict_photometry_compositional()`, `_predict_spectrum_compositional()`
- `src/tengri/core/nonstell.py` — `build_nonstell_fn()` (all non-stellar components)

### NIFTy integration points

- `jit_engine.py:917-945` — builds `OptimizeVI` with `kl_jit=True, residual_jit=True`
- `jit_engine.py:947-987` — `run_nifty_jit()`: Python loop calling `nifty_opt_vi.update(samples, state)`
- `vi.py:814-815` — `signal_response_jit = jax.jit(signal_response)` + `jft.Model(...)`
- `vi.py:869-881` — `jft.optimize_kl(likelihood, init_pos, ...)` — the 27 GB call

### Tests that should pass after any changes

```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu pytest tests/unit/ -q -x          # 1168 tests, ~5 min
JAX_PLATFORMS=cpu pytest tests/integration/ -q -x    # needs SSP data
ruff check src/ tests/
ruff format --check src/ tests/
```

The one pre-existing failure (`test_magphys_emission.py::TestMagphysPAHFeatures::test_pah_template_has_peaks`) is from the `drude_profiles.py` work, unrelated to memory changes.

---

## Follow-up: Compositional Mode + VIConfig Approach (2026-04-11)

### What we investigated

After the `@jax.jit` restoration audit (commit `786092a`), we explored how the three forward model
prediction modes interact with geoVI performance, and what VIConfig levers are available.

### Three forward model modes

| Mode | Mechanism | Speed | Error vs exact |
|------|-----------|-------|----------------|
| `exact` | Full Python dispatch: translate params → SFH → CSP → dust → filter | reference | 0 |
| `compositional` | Fused JIT: single XLA kernel (weights + interp + dust + einsum) | **5.9×** | 0 (bit-identical) |
| `hybrid` | Precomputed SSP×filter baked in at Model init | **278×** | <0.01% |

`mode="auto"` routes: compositional → hybrid → exact (emits warning only on exact fallback).

These numbers are for a simple DPL+SDSS model. For a full Cue+AGN model the gap is even larger:
exact=1,227ms, compositional=11.7ms (105×), hybrid=5.3ms (231×).

### Why compositional helps VI

Every geoVI iteration calls `signal_response` (the forward model) many times:
- Once per sample for the KL divergence
- Once per JVP/VJP step in `metric_vec` (for each CG iteration)
- Once per Newton-CG step in the nonlinear update

With `mode="exact"` at ~9ms/call and 6 samples × 50 CG iterations × 10 VI iterations, that's
~27,000 forward calls per fit = ~243 seconds in forward model alone. With `mode="compositional"`
at ~1.5ms/call, the same run takes ~40 seconds — a direct 6× reduction in forward model time.

The nested JIT (`fused_tier2_phot` inside `signal_response_jit`) is not a correctness problem:
JAX's XlaCallModule has registered JVP/VJP rules, so autodiff traces through it transparently.
The memory benefit is that NIFTy's internal JIT scopes each contain a smaller XLA graph.

### VIConfig and vmap

`use_vmap=True` (the default) passes `residual_map=jax.vmap` to `jft.optimize_kl`. This batches
all N sample draws into a single vmapped call instead of N serial dispatches. With the
compositional kernel being `@jax.jit`, vmap compiles a single batched XLA kernel — one
compilation cost, one device call, full SIMD parallelism across samples.

Usage:

```python
from tengri import VIConfig

cfg = VIConfig(use_vmap=True)  # explicit (same as default)
result = fitter.run("vi", vi_config=cfg)

# Or via model.fit() (**kwargs flows through to run_nifty_vi):
result = model.fit(flux, noise, vi_config=cfg)
```

Levers that matter most for speed/memory trade-offs:

| Field | Default | Effect |
|-------|---------|--------|
| `use_vmap` | `True` | vmap sample draws → 1 batched XLA call (keep True) |
| `n_samples` | `3` | Fewer = faster/less memory, but noisier KL estimate |
| `draw_linear_kwargs["cg_kwargs"]["maxiter"]` | `30` | Fewer JVPs per CG solve |
| `nonlinearly_update_kwargs["minimize_kwargs"]["maxiter"]` | `3` | Newton-CG steps per update |

VIConfig has no flag to JIT NIFTy's outer optimization loop (that's `vi_native`, which is not
yet complete). The `signal_response` in `jit_engine.py` is intentionally left un-JIT'd so
`jax.jvp` / `jax.vjp` can trace through it for metric_vec and hamiltonian computations.

### Tests added

`tests/unit/test_mode_comparison.py` — 17 tests across 4 classes:

- `TestModeRouting` — compositional kernel builds, auto-mode routes correctly
- `TestModeNumericalAgreement` — compositional bit-identical to exact (rel_diff < 1e-10);
  hybrid within 0.5% of exact; agreement holds across 5 random seeds
- `TestModeSpeedOrdering` — compositional < exact and hybrid < exact in wall-clock time;
  prints µs and speedup for manual inspection
- `TestModeWithStochasticSFH` — all three modes finite and bit-identical for `dpl+field` SFH

All 17 pass. Benchmark numbers printed by the speed tests: compositional 1,542µs (5.9×),
hybrid 33µs (277.8×) vs exact 9,117µs.

### Scripts added

`scripts/test_vi_memory_hybrid.py` — measures peak RSS for MAP + VI across forward model modes.
Run as a separate process per mode for clean measurement:

```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu python scripts/test_vi_memory_hybrid.py auto
JAX_PLATFORMS=cpu python scripts/test_vi_memory_hybrid.py hybrid
```
