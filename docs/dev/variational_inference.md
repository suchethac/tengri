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

When you call `fitter.run("native_geovi")` (the default), this is what happens internally:

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
# native_geovi is the default going forward.
result = fitter.run("native_geovi", n_iterations=15)
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

### Method hierarchy

| Method | What it does | When to use |
|--------|-------------|-------------|
| `"native_geovi"` | JIT geoVI with resample+update, nonlinear draws (**DEFAULT**) | Almost always |
| `"native_mgvi"` / `"native_evi"` | JIT MGVI/EVI | Quick look, very high D |
| `"geovi"` / `"fast_geovi"` | NIFTy OptimizeVI.update tight loop | NIFTy-exact math needed |
| `"mgvi"` / `"fast_mgvi"` | NIFTy MGVI tight loop | NIFTy-exact MGVI |
| `"evi"` / `"fast_evi"` | NIFTy EVI tight loop | NIFTy-exact EVI |
| `"nifty_geovi"` | Full jft.optimize_kl with logging | Debugging |
| `"nifty_mgvi"` | Full NIFTy MGVI with logging | Debugging |
| `"geovi_nuts"` | geoVI optimization + NUTS posterior draws | Best of both worlds |
| `"mgvi_nuts"` | MGVI optimization + NUTS posterior draws | VI init + MCMC samples |
| `"raytrace"` | Exact MCMC (Ray Tracing) | Gold-standard validation |
| `"nuts"` | Exact MCMC (NUTS) | Low-D validation |
| `"map"` | Point estimate only | Initialization |

### Backend hierarchy

| Prefix | Backend | Speed | Accuracy |
|--------|---------|-------|----------|
| `native_` | Pure JIT (XLA-compiled) | 0.03s/galaxy* | **Default** |
| (none) / `fast_` | NIFTy `OptimizeVI.update` tight loop | ~12s/galaxy | Exact NIFTy math |
| `nifty_` | Full `jft.optimize_kl` with logging | ~18s/galaxy | Same, with diagnostics |

*After one-time 56s compilation. Best for catalog fitting (100+ galaxies).

### Internal dispatch

| Internal method | Canonical name | Old names (deprecated) |
|----------------|---------------|----------------------|
| `_run_vi` | `vi` (default) | `geovi`, `vi_nifty`, `nifty_geovi`, `fast_geovi` |
| `_run_vi_linear` | `vi_linear` | `mgvi`, `evi`, `vi_nifty_linear`, `nifty_mgvi`, `fast_mgvi` |
| `_run_nifty_fast_vi` | `vi_nifty_fast` | — |
| `_run_nifty_fast_vi_linear` | `vi_nifty_fast_linear` | — |
| `_run_vi_native` | `vi_native` | `native_geovi` |
| `_run_vi_native_linear` | `vi_native_linear` | `native_mgvi`, `native_evi` |
| `_run_map` | `map` | — |
| `_run_nuts` | `mcmc_nuts` | `nuts` |
| `_run_raytrace` | `mcmc_raytrace` | `raytrace` |
| `_run_nss` | `nss` | `evidence` |

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
result = fitter.run("native_geovi", n_iterations=15)

# Chi-squared per data point (should be ~1)
print(result.diagnostics["chi2_dof"])

# Posterior predictive check
for i in range(10):
    sample = {k: v[i] for k, v in result.samples.items()}
    pred = model.predict_photometry(sample)
    # pred should bracket the data within noise
```

## Choosing an Inference Method

tengri has two VI paths with the same variational objective but different drivers.

| Method string | Driver | Notes |
|---|---|---|
| `"vi"` | NIFTy `jft.optimize_kl` (Python loop) | Reference path; verbose logging. |
| `"vi_linear"` | NIFTy MGVI | Linear-only, debugging high-D problems. |
| `"vi_native"` | Pure-JAX `lax.while_loop` | ~18× faster warm-run; **not posterior-equivalent to `"vi"` in general** (see benchmark below). |
| `"vi_native_linear"` | Pure-JAX MGVI | Same as above, MGVI variant. |

**Method names.** The registered VI backends are `"vi"` (NIFTy geoVI, primary), `"vi_nonlinear_fast"` (geoVI without Python logging, primary), and `"vi_linear"` / `"vi_linear_fast"` (NIFTy MGVI, experimental). Older names — `"geovi"`, `"mgvi"`, `"evi"`, `"native_geovi"`, `"vi_native"` and friends — are **not aliases and do not resolve**; they raise. `tengri.list_inference_methods()` is the live list.

**Important equivalence caveat:** A 2026-04-17 benchmark (`bench/reports/2026-04-17_native_vs_nifty.md`) compared `"vi"` vs `"vi_native"` on a 7-parameter parametric setup (15 KL iterations, 6 samples, matched `init_from="random"`). Native was 18.5× faster on warm run, but converged posterior means disagreed by up to 2.3σ on some parameters. **The two paths target the same objective with different solver details and land in different modes on multi-modal problems.** Treat `"vi_native"` as "fast but not identical" — validate per-problem with NUTS (for D ≤ 20) before trusting its posterior.

### `n_samples` gotcha

`VIConfig.n_samples` defaults to `3`. With NIFTy's `mirror_samples=True` (default), this produces **6 effective samples per KL iteration** (each sample paired with its negation). Raising `n_samples` above 12 triggers a warning because high sample counts reduce the stochastic regularization that Newton-CG relies on.

## Performance Guide

| Problem | D | Recommended method | Time |
|---------|--:|-------------------|------|
| Smooth SFH, 5 bands | ~7 | `native_geovi` (15 iter) | 56s compile + 0.3s |
| Stochastic SFH, 5 bands | ~137 | `native_geovi` (15 iter) | 56s compile + 0.8s |
| Stochastic SFH, spectrum | ~200 | `native_evi` (20 iter) | ~60s |
| Hierarchical, 10 galaxies | ~1,400 | `native_evi` (25 iter) | ~10min |
| Catalog, 100+ galaxies | ~7/galaxy | `native_geovi` via `fit_batch` | 56s compile + 3s |

## Mathematical Details

See `docs/geovi_jit.md` for the full math: Hamiltonian, metric, coordinate transform, CG solver, Newton-CG, line search.
