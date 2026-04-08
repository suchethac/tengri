# geoVI JIT Engine: Theory and Implementation

## Overview

The JIT geoVI engine implements geometric variational inference (Frank et al. 2021) as a fully XLA-compiled JAX program. It is mathematically identical to NIFTy's `optimize_kl` for the fixed-noise Gaussian likelihood case, but runs ~100-500x faster by eliminating Python overhead.

## Theory

### The Posterior and Standardized Coordinates

All inference operates in standardized latent coordinates where every parameter has prior xi ~ N(0, I). The information Hamiltonian is:

    H(xi | d) = 0.5 * chi2(d, f(xi)) + 0.5 * xi^T xi

where f(xi) is the forward model (SPS + dust + IGM + photometry/spectroscopy pipeline).

### The Posterior Metric

The Gauss-Newton approximation to the Hessian:

    M(xi) = J^T N^{-1} J + I

where J = df/dxi is the forward model Jacobian and N = diag(sigma^2) is the noise covariance. This is the Fisher information metric plus the identity (prior contribution).

### MGVI (Linear Sampling)

Approximates the posterior as N(m, M^{-1}) where m minimizes the sample-averaged KL divergence. Residual samples are:

    r = M^{-1} (J^T sqrt(N^{-1}) eta_lh + eta_pr)

where eta_lh ~ N(0, I) in data-space and eta_pr ~ N(0, I) in parameter-space. The CG solver inverts M.

### geoVI Coordinate Transform

geoVI constructs a nonlinear coordinate transformation g that flattens the posterior:

    g(x; m) = (x - m) + J^T(m) N^{-1} (f(x) - f(m))

In the transformed y = g(x) coordinates, the posterior is approximately Gaussian even when f is highly nonlinear (e.g., the age-dust-metallicity degeneracy).

### Nonlinear Update (Curving)

To get geoVI samples, we need g^{-1}: given a metric sample ms (with covariance M), find x such that g(x) = ms. This requires solving:

    minimize_x  0.5 * ||ms - g(x)||^2

Starting from x0 = m + r_linear (the MGVI sample), Newton-CG converges in ~3 iterations because:
1. The linear sample is already close to the solution
2. g is a first-order approximation, so higher-order corrections are small
3. The Hessian of the curving objective is well-conditioned

### The optimize_kl Loop

Each iteration:
1. **Draw/update samples** (depending on sample_mode)
2. **Minimize KL** with samples held fixed: Newton-CG on KL(m) = avg_i H(m + r_i)
3. **Update expansion point**: m <- m_new, residuals carried forward

### Sample Modes

- **linear_resample**: Fresh MGVI samples (fast, linear approximation)
- **nonlinear_resample**: Fresh MGVI + geoVI curving (captures posterior geometry)
- **nonlinear_update**: Re-curve existing samples at new m (deterministic refinement, best convergence stability)

### Block Gibbs Scheduling

For structured problems, alternate between parameter blocks:
- Physical params (dust, metallicity, mass): geoVI (nonlinear) — captures degeneracies
- SFH field xi: MGVI (linear) — nearly Gaussian conditioned on physical params

The `constants` mechanism freezes a block during KL minimization while still sampling it for uncertainty propagation. The `point_estimates` mechanism additionally zeros out residuals for speed.

## Implementation

### File Structure

- `src/tengri/fitter.py`: `_build_jit_engine()` contains all JIT-compiled functions
- `src/tengri/vi_config.py`: `VIConfig`, `BlockStep`, `BlockSchedule` configuration
- `src/tengri/hierarchical.py`: Hierarchical inference (mirrors fitter.py structure)

### Core Primitives (inside `_build_jit_engine`)

All functions operate on flat arrays (not pytree dicts) for XLA efficiency.

| Function | NIFTy Equivalent | Description |
|----------|-------------------|-------------|
| `transformation_flat(pos)` | `likelihood.transformation(pos)` | sqrt(N^{-1}) * f(pos) |
| `left_sqrt_metric_flat(pos, v)` | `likelihood.left_sqrt_metric(pos, v)` | J^T(pos) * sqrt(N^{-1}) * v |
| `right_sqrt_metric_flat(pos, v)` | `likelihood.right_sqrt_metric(pos, v)` | sqrt(N^{-1}) * J(pos) * v |
| `draw_metric_sample(pos, key)` | `draw_linear_residual(from_inverse=False)` | Sample with covariance M |
| `draw_residuals(pos, keys)` | `draw_linear_residual(from_inverse=True)` | Sample with covariance M^{-1} |
| `curve_residual(m, r, key, sign)` | `nonlinearly_update_residual()` | Invert g via Newton-CG |
| `metric_vec(xi, v)` | `_StandardHamiltonian.metric(xi, v)` | M(xi) @ v |
| `hamiltonian(xi)` | `_StandardHamiltonian.energy(xi)` | 0.5*chi2 + 0.5*||xi||^2 |
| `kl_vg(m, residuals)` | `_kl_vg(likelihood, m, samples)` | Sample-averaged KL + gradient |
| `kl_metric(m, residuals, v)` | `_kl_met(likelihood, m, v, samples)` | Sample-averaged metric * v |

### What Is Identical to NIFTy

1. **Mathematical formulation**: Same Hamiltonian, metric, coordinate transform g
2. **Sample drawing**: Same algorithm (CG inversion of M applied to J^T sqrt(N^{-1}) eta + eta')
3. **Mirror samples**: Both use antithetic pairs (r, -r) for variance reduction
4. **Nonlinear update objective**: Same phi(x) = 0.5 * ||ms - g(x)||^2
5. **Nonlinear update gradient**: Same analytical gradient -ngrad = -(r + lsm(x, rsm(m, r)))
6. **Nonlinear update metric**: Same Hessian approximation
7. **KL minimization**: Same Newton-CG with metric preconditioning
8. **Sample modes**: linear_resample, nonlinear_resample, nonlinear_update all match

### What Differs from NIFTy

1. **CG solver**: JIT uses energy-based convergence with periodic reset (every 20 iters). NIFTy's CG uses gradient-norm convergence with different reset strategy. Both converge to the same solution but may take slightly different paths.

2. **Newton-CG termination**: JIT uses energy decrease < 1e-3 after miniter=3. NIFTy uses xtol (position change) with custom `sampnorm` gradient norm. This can cause 1-2 extra/fewer Newton steps.

3. **Outer loop convergence**: JIT uses relative KL change < kl_rtol with min 5 iterations. NIFTy runs a fixed number of iterations (no early stopping by default).

4. **Flat arrays vs pytrees**: JIT flattens all params into a single 1D array. NIFTy operates on pytree dicts. Mathematically equivalent but numerically, floating-point ordering of operations differs by ~epsilon.

5. **JIT compilation boundary**: JIT compiles the entire outer loop into one XLA program. NIFTy has Python-level iteration with per-step JIT. This means JIT can't print intermediate diagnostics (all iterations run atomically).

6. **No `sampnorm`**: The JIT CG uses energy-based convergence instead of NIFTy's custom `sampnorm` (which computes ||natgrad||^2 + ||L(m) natgrad||^2). Both are valid convergence criteria.

7. **Variable noise**: JIT's nonlinear update currently supports fixed-noise Gaussian only. NIFTy handles variable noise via `VariableCovarianceGaussian`. Extension is straightforward but deferred.

### Expected Numerical Differences

- **Converged expansion point m**: Should agree within ~1e-3 (CG tolerance)
- **Hamiltonian at convergence**: Should agree within ~0.1 (both near the optimum)
- **Posterior standard deviations**: Should agree within ~10-20% (sampling noise)
- **Posterior means**: Should agree within ~0.5 sigma (expansion point difference)

### User API

```python
from tengri import Fitter

# --- Native JIT (default, fully XLA-compiled) ---
result = fitter.run("native_geovi", ...)   # DEFAULT: JIT geoVI with resample+update, nonlinear draws
result = fitter.run("native_mgvi", ...)    # JIT MGVI
result = fitter.run("native_evi", ...)     # JIT EVI

# --- NIFTy optimize_kl (uses OptimizeVI.update in tight loop) ---
result = fitter.run("geovi", ...)    # geoVI (nonlinear, exact NIFTy math)
result = fitter.run("mgvi", ...)     # MGVI (linearized geoVI)
result = fitter.run("evi", ...)      # EVI: MGVI first half, geoVI second half

# --- Full NIFTy (with logging, for debugging) ---
result = fitter.run("nifty_geovi", ...)   # Full jft.optimize_kl with logging
result = fitter.run("nifty_mgvi", ...)    # Full NIFTy MGVI with logging

# --- Hybrid VI + MCMC ---
result = fitter.run("geovi_nuts", ...)    # geoVI optimization + NUTS posterior draws
result = fitter.run("mgvi_nuts", ...)     # MGVI optimization + NUTS posterior draws

# --- Batch fitting ---
results = fitter.fit_batch(galaxies)      # Default method: native_geovi
```

### Architecture

**`native_geovi` is the default going forward.** It uses a fully XLA-compiled loop
(zero Python overhead) with the "geovi" sample mode: resample at iteration 0 and
every 5th iteration, nonlinear_update between (via `jax.lax.cond`). Posterior draws
are nonlinear (geoVI-curved), not linear CG.

The `"geovi"` / `"mgvi"` / `"evi"` methods (without `native_` prefix) use NIFTy's
`OptimizeVI.update` in a tight Python loop with logging/pickling stripped. This gives:
- **Exact same math** as `jft.optimize_kl` (same CG, Newton-CG, line search, sampnorm)
- **~35% faster** than the full `jft.optimize_kl` (no stdout capture, no pickle saves)
- **Stable convergence** (H stays in 4.5-6.5 range for 8 iterations)

### Internal Dispatch

| Internal method | Canonical name | Old names (deprecated) |
|----------------|---------------|----------------------|
| `_run_vi` | `vi` | `geovi`, `vi_nifty`, `nifty_geovi`, `fast_geovi` |
| `_run_vi_linear` | `vi_linear` | `mgvi`, `evi`, `vi_nifty_linear`, `nifty_mgvi`, `fast_mgvi` |
| `_run_nifty_fast_vi` | `vi_nifty_fast` | — |
| `_run_nifty_fast_vi_linear` | `vi_nifty_fast_linear` | — |
| `_run_vi_native` | `vi_native` | `native_geovi` |
| `_run_vi_native_linear` | `vi_native_linear` | `native_mgvi`, `native_evi` |
| `_run_map` | `map` | — |
| `_run_nuts` | `mcmc_nuts` | `nuts` |
| `_run_raytrace` | `mcmc_raytrace` | `raytrace` |
| `_run_nss` | `nss` | `evidence` |

### Block Gibbs API

```python
from tengri.vi_config import BlockSchedule, BlockStep

# Default schedule for individual galaxy
sched = BlockSchedule.individual_geovi()

# Custom schedule
sched = BlockSchedule(blocks=(
    BlockStep("nonlinear_resample", constants=("sfh_field_xi",)),
    BlockStep("linear_resample", constants=()),
))
```

## References

- Frank, P., Leike, R., Ensslin, T.A. (2021). "Geometric Variational Inference." Entropy 23(7):853. arXiv:2105.10470
- Knollmuller, J., Ensslin, T.A. (2019). "Metric Gaussian Variational Inference." arXiv:1901.11033
- Edenhofer, G. et al. (2024). "Re-envisioning Numerical Information Field Theory (NIFTy.re)." arXiv:2402.16683
