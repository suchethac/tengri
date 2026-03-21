# EVI Optimization Notes

Notes from discussions with Philipp Frank (geoVI author, MPA Garching), March 2025.
For use in the tengri paper's inference methods section.

## Summary of Improvements

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| **Full inference** (10 iter + 2000 samples) | ~70s | **1.7s** | **41x** |
| optimize_kl (10 iterations) | 20s | 0.04s | 500x |
| Posterior sampling (2000 samples) | 1080s | 0.01s | 100,000x |
| Forward pass (JIT'd) | 0.04ms | 0.04ms | — (not bottleneck) |

## What Changed

### 1. Philipp Frank's optimize_kl Configuration

Default `jft.optimize_kl` kwargs were replaced with Philipp Frank's
recommended settings:

```python
draw_linear_kwargs = {"cg_name": "SL", "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30}}
nonlinearly_update_kwargs = {"minimize_kwargs": {"name": "SN", "xtol": 1e-3,
    "cg_kwargs": {"name": None}, "maxiter": 3}}
kl_kwargs = {"minimize_kwargs": {"name": "M", "absdelta": 1e-3,
    "cg_kwargs": {"name": "MCG"}, "maxiter": 10}}
```

- **n_samples=3** with `mirror_samples=True` (6 effective via antithetic sampling)
- **residual_map=jax.vmap** for parallel sample evaluation
- **EVI schedule**: `linear_resample` (MGVI) for first half, `nonlinear_resample` (geoVI) for second half

### 2. JIT-Compiled Signal Response

The forward model (`signal_response`) is wrapped with `jax.jit` before
being passed to `jft.Model`. This ensures the full SPS pipeline (DSPS
+ dust + photometry) is traced once and compiled to XLA.

### 3. Fully JIT-Compiled Optimization Loop (Key Innovation)

**The major breakthrough**: replacing NIFTy's Python-level `optimize_kl`
with a fully JIT-compiled version using `jax.lax.scan` and `jax.lax.while_loop`.

#### The Problem

Profiling revealed that NIFTy's `optimize_kl` spends **99.8% of its time
in Python overhead** — not in the actual linear algebra:

| Operation | JAX compute | Actual time | Python overhead |
|-----------|------------|-------------|-----------------|
| Forward pass | 0.04ms | 0.04ms | 1x (JIT'd) |
| draw_linear_residual | ~2ms | 540ms | 270x |
| KL minimize (1 iter) | ~4ms | 1350ms | 340x |
| 10 optimize_kl iters | ~40ms | 20,000ms | 500x |

The overhead comes from:
- `partial()` / `Partial()` function wrapping per call
- `assert_arithmetics()` type checking
- `jax.debug.callback()` logging inside CG loops
- Python dictionary manipulation for pytree state
- `_process_point_estimate()` processing

#### The Solution

We implemented the geoVI algorithm (Frank et al. 2021, Algorithm 2)
directly in JAX primitives:

1. **Flat array representation**: All parameters stored as a single
   `jnp.float64[D]` vector instead of pytree dicts. Eliminates all
   pytree manipulation overhead.

2. **`jax.lax.while_loop` CG solver**: The conjugate gradient solve
   (equation 19 in the paper) uses JAX's functional while_loop.
   Includes energy-based convergence, curvature checks, and periodic
   reset (adapted from NIFTy's `_static_cg`).

3. **`jax.lax.scan` optimization loop**: The outer KL iteration loop
   uses `jax.lax.scan`, allowing XLA to compile the entire 10-iteration
   optimization into a single fused computation.

4. **`jax.vmap` over samples**: All sample-level operations (drawing,
   metric evaluation, KL gradient) are vmapped for parallel execution.

The math is identical to NIFTy's implementation:
- Sample: `z = J^T √(N^{-1}) η₁ + η₂`, solve `M·r = z` via CG
- KL: `KL(m) = (1/N) Σᵢ H(m + rᵢ)`
- KL minimize: Newton-CG with metric preconditioning

### 4. JIT-Compiled Posterior Sampling

After the optimization converges, additional posterior samples are drawn
using the same JIT-compiled CG machinery:

```python
draw_samples = jax.jit(jax.vmap(draw_one))  # compile once
residuals = draw_samples(pos_flat, keys)     # 2000 samples in ~10ms
```

This replaces NIFTy's `draw_linear_residual` which took ~540ms per sample
due to the same Python overhead.

## Profiling Methodology

All benchmarks on Apple M-series CPU (no GPU), JAX 0.5+, float64.
Model: stochastic SFH with D=137 parameters, 5 SDSS photometric bands.

### Forward Model Profiling

```
Forward pass (JIT'd):     0.04 ms
Gradient (JIT'd):         0.07 ms  (1.6x forward)
JVP:                      0.05 ms  (1.1x forward)
VJP:                      0.06 ms  (1.5x forward)
```

The forward model is not the bottleneck. The entire SPS pipeline
(DSPS stellar population synthesis + Charlot & Fall dust + photometric
integration) executes in 40 microseconds when JIT-compiled.

### optimize_kl Per-Iteration Profiling (NIFTy)

```
Iter  draw_smp   kl_min    total    mode
   1    1.80s     1.69s     3.49s   MGVI  (JIT compile)
   2    0.00s     1.31s     1.32s   MGVI
   3    0.01s     1.38s     1.38s   MGVI
   6    4.07s     1.36s     5.43s   geoVI (JIT compile)
   7    0.02s     1.39s     1.41s   geoVI
  10    0.02s     1.38s     1.39s   geoVI
```

**KL minimization dominates at ~1.35s/iter** (Newton-CG with Python overhead).
Sample drawing is fast after compilation (0.01-0.02s).

### Custom JIT Pipeline

```
10 EVI iterations (JIT'd):           0.04s
2000 posterior samples (JIT CG):     0.01s
JIT compilation (one-time):          ~3s
Total (second run, cached):          1.7s
```

## Theoretical Background

### EVI = Expansion-point Variational Inference

EVI schedules MGVI (cheap, linear) for early iterations and geoVI
(accurate, nonlinear) for later iterations. From Frank et al. (2021),
MGVI is the first-order linearization of geoVI (their equation 37):
expanding `g⁻¹(y)` to first order in `y` recovers `N(0, M⁻¹)`.

Early iterations have a poor expansion point ξ̄, so the nonlinear
coordinate transformation is inaccurate anyway. Cheap MGVI samples
move ξ̄ to the posterior region quickly; then geoVI refines.

### Why n_samples=3 with Mirror Samples

`mirror_samples=True` creates antithetic pairs: for each sample r*,
the algorithm also uses -r*. Antithetic samples are negatively
correlated, halving the KL gradient variance:

```
KL̂ ≈ (1/N) Σᵢ H(m + rᵢ*)
```

3 real + 3 mirrored = 6 effective samples with better gradient
estimates than 6 independent samples.

### Why geoVI Over Adam + NUTS

geoVI uses the full Fisher metric M = J^T N^{-1} J + I, preserving
off-diagonal parameter correlations (age-dust-metallicity degeneracy).
Adam is diagonal (1st order). NUTS uses an estimated mass matrix but
doesn't exploit the metric during sampling. geoVI's coordinate
transformation absorbs the posterior geometry into the sampling
coordinates, enabling Gaussian approximation even for non-Gaussian
posteriors.

## Implementation Details

### Available Methods

| Method | Command | Backend | Speed |
|--------|---------|---------|-------|
| native_geovi (DEFAULT) | `fitter.run("native_geovi")` | JIT (XLA) | **0.03s/gal** (after 56s compile) |
| native_mgvi / native_evi | `fitter.run("native_mgvi")` | JIT (XLA) | **0.03s/gal** (after compile) |
| geovi / fast_geovi | `fitter.run("geovi")` | NIFTy tight loop | ~12s |
| mgvi / fast_mgvi | `fitter.run("mgvi")` | NIFTy tight loop | ~15s |
| evi / fast_evi | `fitter.run("evi")` | NIFTy tight loop | ~12s |
| nifty_geovi | `fitter.run("nifty_geovi")` | Full NIFTy | ~18s |
| nifty_mgvi | `fitter.run("nifty_mgvi")` | Full NIFTy | ~18s |
| geovi_nuts / mgvi_nuts | `fitter.run("geovi_nuts")` | VI + NUTS | ~20s |
| MAP | `fitter.run("map")` | Adam/optax | ~2s |
| Ray Tracing | `fitter.run("raytrace")` | Custom JAX | ~60s |
| NUTS | `fitter.run("nuts")` | BlackJAX | ~120s |

Batch fitting: `fitter.fit_batch(galaxies)` — default method is `native_geovi`.

### Posterior Sampling Methods

| Method | Command | Speed (2000 samples) |
|--------|---------|---------------------|
| JIT nonlinear (default for geoVI) | `posterior_method="nonlinear"` | ~5ms/sample |
| JIT CG (linear) | `posterior_method="jit"` | **0.01s** |
| BlackJAX NUTS | `posterior_method="blackjax"` | ~6s |
| NIFTy CG | `posterior_method="nifty"` | ~1080s |

### Pre-compilation

```python
fitter = Fitter(model, data, noise)
fitter.compile()  # ~3s one-time cost
result = fitter.run("native_geovi", n_posterior_samples=2000)
```
