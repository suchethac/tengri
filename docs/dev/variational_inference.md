# Variational Inference in tengri

## The Big Picture

You have a galaxy. You have photometry (or a spectrum). You want to know the physical properties: dust, metallicity, star formation history. There are many combinations of parameters that could produce similar-looking data — that's the **posterior**.

Variational inference finds this posterior by searching for the best Gaussian-like approximation. It's much faster than MCMC for high-dimensional problems (D > 50), at the cost of being approximate.

## How It Works — Intuitively

Imagine you're mapping a mountain landscape in fog. You can't see the whole thing at once, but you can:

1. **Drop a pin** on the map (your current best guess = expansion point **m**)
2. **Send scouts** in different directions from the pin (= draw **samples**)
3. Each scout reports back: "the terrain slopes this way" (= **KL gradient**)
4. You **move the pin** to a better location based on the scouts' reports (= **Newton step**)
5. Repeat

The quality of your map depends on two things:
- **Where** the scouts go (sample quality)
- **How** you move the pin (optimization quality)

## Sample Modes — The Scout Strategies

### Linear scouts (MGVI)

The scouts walk in straight lines from the pin. Fast to deploy, but if the landscape curves (the age-dust degeneracy is a banana-shaped valley), they miss the shape.

```
resample: send new scouts each round    → explores well, reports are noisy
sample:   send same scouts each round   → consistent reports, less exploration
```

### Nonlinear scouts (geoVI)

The scouts follow the curvature of the landscape. They walk along the banana instead of in straight lines. More expensive (each scout needs to solve a Newton problem to find the curved path), but they map the posterior shape accurately.

```
resample: fresh scouts, curved paths    → best exploration, some noise
sample:   same scouts, re-curved paths  → deterministic, excellent
update:   same scouts, re-adjust only   → cheapest, best stability
```

### The Staleness Problem

If you keep the same scouts for too many rounds, they become **stale**: the pin has moved, but the scouts are still exploring from the old position. Their curved paths get adjusted, but the base randomness is fixed. Over many iterations, this causes a slow drift.

**Solution**: periodically send fresh scouts (resample) to prevent staleness.

## The Default Strategy

When you call `fitter.run("vi")` (the default), this is what happens internally:

```
Iteration  1:  nonlinear_resample   ← fresh curved scouts (establish)
Iteration  2:  nonlinear_update     ← re-adjust at new pin (refine)
Iteration  3:  nonlinear_update     ← refine
Iteration  4:  nonlinear_update     ← refine
Iteration  5:  nonlinear_update     ← refine
Iteration  6:  nonlinear_resample   ← FRESH scouts (prevent staleness)
Iteration  7:  nonlinear_update     ← refine
Iteration  8:  nonlinear_update     ← refine
Iteration  9:  nonlinear_update     ← refine
Iteration 10:  nonlinear_update     ← refine
Iteration 11:  nonlinear_resample   ← FRESH scouts (prevent staleness)
...
```

Fresh scouts every 5 iterations. Deterministic refinement in between. This gives:
- **Stable convergence** (no oscillation from noisy samples)
- **No staleness** (periodic refresh prevents drift)
- **Good posterior quality** (nonlinear curving captures banana shapes)

## User API

### Simple (recommended)

```python
# Just works. Uses the optimal schedule internally.
# vi (NIFTy geoVI) is the default.
result = fitter.run("vi", n_iterations=15)
```

### With control

The per-iteration geoVI cadence is resolved **inside** the backend
(`inference/backends/vi/nifty.py`), not passed in: `nonlinear_resample` at
iteration 0 and every `resample_every` (5) iterations, `nonlinear_update`
between. What you control is `VIConfig`:

```python
from tengri.inference.vi_config import VIConfig

cfg = VIConfig(
    n_iterations=25,
    n_samples=6,             # -> 12 effective, mirror_samples=True
    evi_linear_fraction=0.5, # EVI only: fraction of iterations run as MGVI first
)
result = fitter.run("vi", vi_config=cfg)

# MGVI (linear) instead of geoVI
result = fitter.run("vi_linear", n_iterations=10)
```

There is no `schedule=` parameter. `OptimizationSchedule` was deleted in #1293
as dead code — nothing ever consumed it, and the cadence it expressed is the
one the backend already applies internally.

### VI Method reference

For the full list of available methods (VI and otherwise), call `tengri.list_inference_methods()`.

The variational inference methods are:

| Method | Tier | What it does | When to use |
|--------|------|-------------|-------------|
| `"vi"` | primary | NIFTy geoVI with resample+update, nonlinear draws (**DEFAULT**) | Almost always; best balance of speed and accuracy |
| `"vi_nonlinear_fast"` | primary | NIFTy geoVI without Python logging | Same as `vi`, slightly faster (logging overhead removed) |
| `"vi_linear"` | experimental | NIFTy MGVI with logging (linear approximation to posterior) | High-D exploration, lower memory than geoVI |
| `"vi_linear_fast"` | experimental | NIFTy MGVI without Python logging | High-D quick look, lower memory |
| `"native_vi_nonlinear"` | broken | JIT-compiled geoVI (internal use only) | [segfaults on some models; do not use] |
| `"native_vi_linear"` | broken | JIT-compiled MGVI (internal use only) | [segfaults on some models; do not use] |

### Backend speeds (empirical)

The `_fast` variants skip Python logging, so they're measurably faster on catalog fits where the logging overhead dominates per-galaxy time. The difference is largest with `vmap` over many galaxies; on single-galaxy fits it's negligible.

For high-dimensional problems (D ≥ 50), `vi_linear` converges faster than `vi` (MGVI is cheaper per iteration than geoVI), at the cost of lower posterior fidelity. Try `vi_linear` first if your model is high-D; measure the posterior quality before committing.

### Deprecated names

The following method names are no longer in the registry. If you encounter them in old notebooks or older code, they have been renamed:

| Old name | Current name | Status |
|----------|-------------|--------|
| `native_geovi`, `geovi`, `vi_nifty`, `nifty_geovi`, `fast_geovi` | `vi` | Consolidated to one entry point (both logging and fast variants available separately) |
| `mgvi`, `evi`, `vi_nifty_linear`, `nifty_mgvi`, `fast_mgvi` | `vi_linear` | Consolidated; use `vi_linear_fast` for non-logging variant |
| `native_vi_nonlinear` | (broken — do not use) | Segfaults on some models; broke during native JIT refactor |
| `native_vi_linear` | (broken — do not use) | Segfaults on some models; broke during native JIT refactor |

### Batch fitting

`fitter.fit_batch(galaxies)` fits multiple galaxies. Default method is `vi`.

**Removed names**: `fit_catalog` -> `fit_batch`.

## The Five Sample Modes (for advanced users)

| Mode | Fresh keys? | Linear draw? | Curve? | Stability | Quality |
|------|:-----------:|:------------:|:------:|:---------:|:-------:|
| `linear_resample` | Yes | Yes | No | Noisy | Low |
| `linear_sample` | No | Yes | No | Stable | Low |
| `nonlinear_resample` | Yes | Yes | Yes | Some noise | High |
| `nonlinear_sample` | No | Yes | Yes | Stable | High |
| `nonlinear_update` | — | No | Re-curve | Most stable | High* |

*Degrades over many iterations without periodic resample.

### What "nonlinear curving" means

The posterior in (dust, metallicity) space is banana-shaped. A Gaussian centered on the mode misses the tails of the banana. geoVI finds a coordinate transform **g** that straightens the banana:

```
Original space:        geoVI space:
   ___                    |
  /   \                   |
 /     \       g(x)       |
|  mode |   -------->     * mode
 \     /                  |
  \___/                   |
  (banana)              (straight)
```

In the straightened space, a Gaussian IS a good approximation. The "curving" step inverts g to map samples from the straight space back to the banana.

### What "nonlinear_update" means

When the pin (expansion point m) moves, the coordinate transform g changes (because g depends on the Jacobian at m). The existing curved samples need to be re-adjusted to the new g. This is cheaper than drawing new samples because:
1. No new random numbers (deterministic)
2. The previous curved position is a good starting point for the Newton solve
3. Only ~3 Newton iterations needed

## Convergence Diagnostics

After fitting, check:

```python
result = fitter.run("vi", n_iterations=15)

# Chi-squared per data point (should be ~1)
print(result.diagnostics["chi2_dof"])

# Posterior predictive check
for i in range(10):
    sample = {k: v[i] for k, v in result.samples.items()}
    pred = model.predict_photometry(sample)
    # pred should bracket the data within noise
```

## Choosing a VI Backend

The two main variational inference backends are:

| Method string | Backend | Notes |
|---|---|---|
| `"vi"` | NIFTy geoVI (`jft.optimize_kl`, Python loop) | **Default.** Best posterior quality; includes Python logging. |
| `"vi_nonlinear_fast"` | NIFTy geoVI, no logging | Same algorithm as `vi`, slightly faster on catalog fits (logging overhead removed). |
| `"vi_linear"` | NIFTy MGVI (`jft.optimize_kl`, Python loop) | Linear approximation to posterior; converges faster in very high D (≥ 50) at cost of fidelity. |
| `"vi_linear_fast"` | NIFTy MGVI, no logging | Same as `vi_linear`, faster on catalog fits. |

**Method names.** Older registrations like `"native_geovi"`, `"geovi"`, `"mgvi"`, `"vi_native"`, and similar do **not exist** and will raise a `ParameterError`. Call `tengri.list_inference_methods()` to see the live registry.

Readers encountering references to `"vi_native"` in older code should substitute `"vi"`. The native backends (`"native_vi_nonlinear"`, `"native_vi_linear"`) still exist in the registry but at `tier=broken`: asking for one by its real name raises `BackendError` naming the tier and the working alternatives — a different failure from the never-registered names above, which raise `ParameterError`. They are not posterior-equivalent to the NIFTy reference on some problems (see `bench/reports/2026-04-17_native_vs_nifty.md` for measurements).

### `n_samples` gotcha

`VIConfig.n_samples` defaults to `3`. With NIFTy's `mirror_samples=True` (default), this produces **6 effective samples per KL iteration** (each sample paired with its negation). Raising `n_samples` above 12 triggers a warning because high sample counts reduce the stochastic regularization that Newton-CG relies on.

## Performance Guide

| Problem | D | Recommended method | Time |
|---------|--:|-------------------|------|
| Smooth SFH, 5 bands | ~7 | `vi` (15 iterations) | ~2-5s |
| Stochastic SFH, 5 bands | ~137 | `vi` (15 iterations) | ~10-30s |
| Stochastic SFH, spectrum | ~200 | `vi_linear` (20 iterations) | ~30-60s |
| Hierarchical, 10 galaxies | ~1,400 | `vi_linear` (25 iterations) | ~10 min |
| Catalog, 100+ galaxies | ~7/galaxy | `vi` via `fit_batch` | ~3-10s total (after compilation) |

## Mathematical Details

See `docs/geovi_jit.md` for the full math: Hamiltonian, metric, coordinate transform, CG solver, Newton-CG, line search.
