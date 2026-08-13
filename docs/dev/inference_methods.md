# Inference Methods in tengri

A comprehensive reference for every inference method available in tengri: mathematical
foundations, implementation details, performance characteristics, and usage patterns.

This document consolidates and expands:
- `docs/variational_inference.md` (intuitive overview, sample modes, schedules)
- `docs/geovi_jit.md` (mathematical theory, JIT engine, NIFTy equivalence)
- `docs/hierarchical_block_gibbs.md` (block Gibbs structure for hierarchical models)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [geoVI: Geometric Variational Inference](#3-geovi-geometric-variational-inference)
4. [The Five Sample Modes](#4-the-five-sample-modes)
5. [The Optimal Resample+Update Schedule](#5-the-optimal-resampleupdate-schedule)
6. [Method Hierarchy](#6-method-hierarchy)
7. [Compilation and Caching](#7-compilation-and-caching)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [Posterior Sampling](#9-posterior-sampling)
10. [Block Gibbs for Hierarchical Models](#10-block-gibbs-for-hierarchical-models)
11. [Convergence Diagnostics](#11-convergence-diagnostics)
12. [Quick Reference](#12-quick-reference)
13. [References](#13-references)

---

## 1. Overview

### The Problem

You have a galaxy. You have photometry (or a spectrum). You want to know the physical
properties: dust, metallicity, star formation history. There are many combinations of
parameters that could produce similar-looking data --- that is the **posterior**.

tengri provides five families of inference methods to explore this posterior:

| Family | Canonical method | Old names (deprecated) | Posterior type | Best for |
|--------|-----------------|----------------------|---------------|----------|
| **MAP** | `map` | — | Point estimate | Initialization, quick look |
| **Variational (linear)** | `vi_linear` | `mgvi`, `native_mgvi`, `fast_mgvi`, `nifty_mgvi` | Approximate Gaussian | Very high D (>10^5), speed |
| **Variational (nonlinear)** | `vi` (default) | `native_geovi`, `geovi`, `fast_geovi`, `native_evi`, `evi`, `nifty_geovi` | Non-Gaussian VI | Most problems |
| **Hybrid** | `vi` → `.refine("mcmc_nuts")` | `geovi_nuts`, `mgvi_nuts` | VI optimization + MCMC samples | Best of both worlds |
| **MCMC (high-D)** | `mcmc_raytrace` | `raytrace` | Exact posterior | Validation, high-D |
| **MCMC (low-D)** | `mcmc_nuts` | `nuts` | Exact posterior | Gold standard, D ≤ 30 |

NIFTy is the computational library implementing geoVI; the `vi` method uses its fast path by default.

### When to Use What

| Problem | D | Recommended | Fallback |
|---------|--:|-------------|----------|
| Smooth SFH, few bands | ~7 | `vi` (15 iter) | `mcmc_raytrace` |
| Stochastic SFH, photometry | ~137 | `vi` (15 iter) | `vi_linear` (20 iter) |
| Stochastic SFH, spectrum | ~200 | `vi_linear` (20 iter) | `vi` (25 iter) |
| Hierarchical, 10 galaxies | ~1,400 | `vi_linear` (25 iter) | `vi` |
| Catalog, 100+ galaxies | ~7/gal | `vi` via `fit_batch` | `vi_linear` |
| Validation (low-D) | <20 | `mcmc_nuts` | `mcmc_raytrace` |
| Validation (high-D) | >20 | `mcmc_raytrace` | `vi` + `.refine("mcmc_nuts")` |

### Usage

All methods are accessed through `Fitter.run()`:

```python
from tengri import SEDModel, Parameters, Fitter

model = SEDModel(spec, ssp)
fitter = Fitter(model, data, noise)

# Simple (vi is the default)
result = fitter.run("vi", n_iterations=15)

# With initialization from MAP
result_map = fitter.run("map", n_steps=1500)
result = fitter.run("vi", init_from=result_map)

# Batch fitting (default method: vi)
results = fitter.fit_batch(galaxies)

# Access results
print(result.params)            # posterior mean (physical space)
print(result.samples)           # dict of arrays, shape (n_samples, ...)
print(result.diagnostics)       # chi2_dof, n_iterations, etc.
```

---

## 2. Mathematical Foundations

### 2.1 Standardized Coordinates

All inference in tengri operates in **standardized latent coordinates** where every
parameter has prior xi ~ N(0, I). Physical parameters with bounded priors (e.g.,
Uniform(lo, hi)) are mapped to unbounded space via a sigmoid transform:

```
u = to_unbounded(x, lo, hi)    # physical -> unbounded
x = to_bounded(u, lo, hi)      # unbounded -> physical
```

The GP latent vector `psd_xi` is already standardized (prior N(0, I)) and needs no
transform.

### 2.2 The Information Hamiltonian

The posterior in standardized coordinates is characterized by the information Hamiltonian:

```
H(xi | d) = 0.5 * chi2(d, f(xi)) + 0.5 * xi^T xi
```

where:
- `f(xi)` is the full forward model pipeline (SPS + dust + IGM + photometry/spectroscopy)
- `chi2(d, f(xi)) = (d - f(xi))^T N^{-1} (d - f(xi))` is the chi-squared
- `N = diag(sigma^2)` is the noise covariance
- `0.5 * xi^T xi` is the standard normal prior contribution

The first term pulls toward data-fitting solutions; the second regularizes toward the
prior. The posterior mode minimizes H.

### 2.3 The Posterior Metric (Fisher Information)

The Gauss-Newton approximation to the Hessian of H gives the posterior metric:

```
M(xi) = J^T N^{-1} J + I
```

where `J = df/dxi` is the Jacobian of the forward model. This is the Fisher information
metric plus the identity (prior contribution).

The metric defines the local curvature of the posterior:
- **Large eigenvalues** = tightly constrained directions (data-dominated)
- **Eigenvalues near 1** = unconstrained directions (prior-dominated)

In tengri, `J` is computed via JAX automatic differentiation (`jax.jvp` for forward-mode,
`jax.vjp` for reverse-mode). The metric-vector product `M(xi) @ v` never materializes
the full Jacobian:

```python
def metric_vec(xi, v):
    """M(xi) @ v = (J/sigma)^T (J/sigma) v + v."""
    _, Jv = jax.jvp(signal_response, (xi,), (v,))       # forward: J @ v
    _, vjp_fn = jax.vjp(signal_response, xi)
    return vjp_fn(whiten(whiten(Jv, noise), noise))[0] + v   # reverse: J^T @ (N^{-1} J v) + v
```

Note the spelling: **whitened twice, never `noise_inv * Jv`**. Algebraically the
same, but `N^{-1} = 1/sigma^2` is ~1e59 at a real flux uncertainty — outside
float32 entirely — so the direct form returned `inf`, or `NaN` wherever `Jv` was
exactly zero, and geoVI drew NaN posterior samples while reporting a perfectly
healthy energy (#1588). `whiten` lives in `tengri.utils.scale` and carries an
`optimization_barrier`, without which XLA re-associates the two divisions back
into the overflowing product under `jit`.

This implicit matrix-vector product is the foundation of all VI methods.

### 2.4 Variable Noise Extension

When a noise model is active (calibration uncertainty, Student-t likelihood), the
Hamiltonian includes a log-determinant term and the metric becomes:

```
M(xi) = J_full^T H_noise J_full + I
```

where `J_full` includes derivatives of both the signal and noise model, and `H_noise`
is the Hessian of the noise-model likelihood. The `variable_noise_metric_vec` function
handles this generalization.

### 2.5 The Same Metric for HMC (`precondition=True`)

The metric above is what makes the VI methods work: geoVI and MGVI re-evaluate it at the
current position on every iteration, so the geometry they see is always locally white.
NIFTy does the same thing (`nifty8/re/evi.py`), and standardization exists precisely so
the prior's contribution is exactly `I` and only `J^T N^{-1} J` has to be computed.

A Hamiltonian sampler cannot do that. Its mass matrix is estimated once during warmup
and then frozen, so it gets no benefit from the standardized prior when the likelihood
curvature varies. On the correlated-field posterior the gap is wide — measured at
n_grid=16 with emission lines (D=25):

```
cond(grad^2 H) at the MAP        1.1e5      (eigenvalues 0.81 ... 9.0e4)
```

No diagonal mass matrix covers that, and a dense one estimated from warmup draws is both
noisy and memory-hungry.

`precondition` (**opt-in — default off**; pass `True` to enable) supplies the metric
analytically instead. It is a **linear change of variables**, not a mass matrix:

```
G = -grad^2 log p   at the initial point, eigenvalue MAGNITUDES floored at 1.0
G_alpha = V diag(lambda^alpha) V^T        spectrum capped at MAX_METRIC_CONDITION
G_alpha = L L^T,  A = L^{-T}              so  A A^T = G^{-alpha}

sample  H(A zeta)   instead of   H(xi),   then map draws back with   xi = A zeta
```

`alpha` is the **whitening strength** — `precondition=True` uses
`DEFAULT_WHITENING_STRENGTH` (0.5), `precondition=1.0` is full whitening, and any float
in `[0, 1]` is accepted. §2.5.1 is why the default is not 1.

The magnitude (`max(|lambda|, 1.0)`, saddle-free Newton) matters at a non-stationary
expansion point: a steeply *negative* direction is treated as steep, not flat — a
signed floor left one unconverged-MAP fit mis-scaled by 51x. At a true stationary
point all eigenvalues are positive and the two are identical.

Because the map is linear its Jacobian is constant, so the posterior is unchanged — only
the geometry the integrator sees. The eigenvalue floor of 1.0 is the prior's exact
contribution: the Gauss-Newton likelihood term is positive semi-definite, so anything
below 1 is residual curvature (the term Gauss-Newton drops).

The metric is formally position-dependent, so a single one built at the MAP is an
approximation. Measured, it holds across the region a chain actually visits:

| point | preconditioned cond |
|---|---|
| at the MAP | 1.23 |
| timescale latents at +2 sd | 2.62 (raw cond there: 1.2e4) |
| random 1-sd jitter | 2.05 |

#### 2.5.1 Why the default strength is 0.5, not 1

Full whitening has no floor under it. Measured on single-galaxy photometry fits, it
ranged from **0.10x to 5.76x ESS/s across seeds of the same model** — a 58x spread, above
1 in exactly half of usable pairs, with the sign flipping *within* one configuration
across seeds. The post-warmup step size explains the outcome at **r = +0.92** (log-log),
and 7 of 9 fits had preconditioning *lower* it.

The mechanism is exact rather than statistical. Write the true precision as `H` and the
metric actually used as `G = H^gamma`, where `gamma = 1` means a perfect metric. For any
`A` with `A A^T = G^{-alpha}`, the eigenvalues of the whitened precision `A^T H A` are the
generalized eigenvalues of the pencil `(H, G^alpha)` — they do not depend on which root
was chosen — so

```
cond_whitened = cond(H) ** |1 - alpha*gamma|
```

Preconditioning is therefore worse than doing nothing exactly when `gamma > 2/alpha`:

| `gamma` | `alpha = 1` (full) | `alpha = 0.5` (default) |
|---|---|---|
| 1 (perfect metric) | **1.0** | `cond^0.5` |
| 2 | `cond` — *identical to no preconditioning* | **1.0** |
| 3 | `cond^2` — *worse than the original problem* | `cond^0.5` |

There is no plateau: past `gamma = 2` full whitening amplifies as `cond^(gamma-1)`,
unbounded. And `gamma != 1` is the **normal** case, not a pathological one — a
single-point Hessian at the MAP is a *modal* curvature estimate, and wherever the
posterior is non-Gaussian that is not the curvature of the bulk.

Halving the exponent doubles the tolerated misspecification (`gamma <= 4`) and costs only
a `cond^0.5` residual when the metric happens to be exact. Ill-conditioning here runs
`1e5`–`3e8`, so `cond^0.5` is still a 300x–17000x improvement on the raw problem.

`MAX_METRIC_CONDITION` (1e8) is a backstop, not the mechanism: the smallest eigenvalues
are the least reliably estimated and the most damaging when wrong.

**`gamma` is measurable, and it sits just under the cliff.** Whiten fully, then evaluate
the curvature one posterior standard deviation away in the whitened frame; were the metric
exact that would be `I`. Inverting the law gives `gamma ~ 1 + log(S)/log(cond)`. Measured
(median over 6 jitter directions, one seed per configuration):

| config | D | cond @ MAP | whitened @ 1 sd | `gamma` | optimal `alpha` = 1/`gamma` |
|---|---|---|---|---|---|
| `d7` | 7 | 7.8e4 | 8.9e3 | 1.81 | 0.55 |
| `d8dust` | 8 | 7.9e4 | 4.6e3 | 1.75 | 0.57 |
| `d8met` | 8 | 8.0e4 | 5.1e3 | 1.76 | 0.57 |

This is the quantitative account of the sign instability. At `gamma ~ 1.8` full whitening
leaves `cond^0.8` — barely better than doing nothing — and any seed-to-seed variation in
the effective `gamma` crosses 2, where the sign flips to amplification. Full whitening was
not unlucky; it was parked on a knife edge. At `alpha = 0.5` the exponent is `|1 - 0.9|`
= 0.1, well inside the safe region.

It also argues **against** per-problem auto-tuning: `gamma` is stable at 1.75–1.81 across
every configuration measured, so a tuner would return ~0.56 every time. The fixed default
is the right shape of fix, not a simplification.

**Practical notes.**

- Pass `init_from` a MAP result so the metric is built where the chain will be.
- Cost is not a constraint below the cap: the dense Hessian is **flat at ~2 s** from
  D=25 to D=521 (`jax.hessian` is `jacfwd(jacrev)`, which vectorizes rather than taking
  D sequential backward passes), and `eigh` + Cholesky is 0.11 s / 2.2 MB at D=521.
  Only the `O(D^3)` factorization grows — hence the 1024 cap. "Easy low-dimensional"
  posteriors turned out not to exist here: the simplest configuration measured
  (double-power-law + photometry, D=7) already had raw cond 8.5e4.
- **Why it is opt-in.** Not because of [#1397](https://github.com/suchethac/tengri/issues/1397),
  despite that being the original stated reason here. Those notebook failures were
  caused by a **sub-band node gradient underflowing to NaN** inside the model;
  preconditioning was the first thing to notice, not the cause, and with the root fix
  `notebooks/07` returns to its pre-preconditioning R-hat. The reason that survives is
  measured on the *fixed* code: on `recipes.mock_recovery_minimal()` (D=7), 4 seeds of
  4 converge without preconditioning (R-hat 0.997-1.007) and none converge with it at
  full strength (1.055-2.689), at 4x to 25x worse ESS/s. The whitening strength
  ([#1442](https://github.com/suchethac/tengri/issues/1442)) bounds that damage but does
  not overturn the conclusion — see the sweep below.
- **The conditioning win is not a demonstrated sampling win — at any strength.** Three
  arms, one frozen source (`b17b319d8`), 3 configs x 3 seeds x {off, `alpha=1`,
  `alpha=0.5`}, 27 fits, 0 failures. Restricted to the 8 cells where **off itself
  converged**:

  | arm | broke convergence | aggregate ESS/s vs off* |
  |---|---|---|
  | `alpha = 1` (full) | **5 of 8** | **0.57x** |
  | `alpha = 0.5` (default) | **2 of 8**, both marginal (1.063, 1.064) | **0.79x** |

  \* an unusable fit scores 0 rather than being dropped. **Gating on convergence and then
  taking a median is survivorship bias**: full whitening's gated median is 1.05x, computed
  over only the 3 cells where it did not break the fit. The gate is right for comparing
  sampling efficiency and wrong for deciding whether to enable something.

  Partial whitening is strictly better on both axes, and repairs the catastrophic cases —
  `d8dust` seed 3 went from R-hat **1.437** at a 0.04x step size (full) to R-hat **1.004**
  at 1.13x (half). But **neither arm beats `off`**, which converged in 8 of 9 cells and
  had the best aggregate throughput. Even at the measured-optimal strength the geometry
  win does not become a sampling win: NUTS's warmup adaptation is already doing that work,
  and a frozen metric partly competes with it. This is why the feature is opt-in.
- The `D=8` degradation is **dimensional, not metallicity-specific**: a second `D=8` cell
  varying the dust slope instead (`d8dust`) agreed with free metallicity (`d8met`).
- **Never benchmark this on one seed.** The pre-fix numbers in the history of this
  document (1.87x at D=7, 0.84x at D=8) came from single-arm medians on a source tree
  that still carried the sub-band gradient bug
  ([#1420](https://github.com/suchethac/tengri/issues/1420)) and from too few seeds to see
  the spread. Gate on *both* arms converging before computing any ratio, and print the
  excluded pairs — an exclusion is often the most informative row.
- **Where it is most likely to pay.** All three HMC backends force `use_dense = False`
  above D=30, so above that threshold the competitor is a *diagonal* mass matrix, which
  cannot represent rotation at all. Below D=30 the competitor is an empirical dense
  covariance from warmup — a far stronger baseline, and where the 0.84x was measured.
- Which backends accept the flag is a **declared registry capability**
  (`register_backend(..., accepts_precondition=True)`), not a list maintained by hand.
  Today that is the Hamiltonian family; gradient-free and self-tuning kernels
  (`mcmc_raytrace`, `mcmc_ess`, `mcmc_mclmc`) do not declare it, because whitening the
  integrator's metric is meaningless for them. Passing `precondition=True` to a backend
  that has not declared it raises at dispatch and names the ones that have.

---

## 3. geoVI: Geometric Variational Inference

### 3.1 Why Not Just a Gaussian?

The posterior in SED fitting is often **banana-shaped** due to the age-dust-metallicity
degeneracy. A Gaussian centered on the mode (which is what MGVI provides) misses the
curved tails:

```
Original space:          Gaussian approximation:
   ___                        .  .
  /   \                     .      .
 /     \                   .   *    .
|  mode |                  .      .
 \     /                    .  .
  \___/
  (banana)                 (misses tails)
```

### 3.2 The geoVI Coordinate Transform

geoVI (Frank et al. 2021) constructs a nonlinear coordinate transformation g that
**straightens the banana**:

```
g(x; m) = (x - m) + J^T(m) N^{-1} (f(x) - f(m))
```

In the transformed coordinates y = g(x), the posterior is approximately Gaussian even
when f is highly nonlinear. The key insight: g incorporates the nonlinearity of the
forward model into the coordinate system itself.

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

In the straightened space, a Gaussian IS a good approximation. The "curving" step inverts
g to map samples from the straight space back to the banana.

### 3.3 How geoVI Differs from MGVI

**MGVI** (Metric Gaussian Variational Inference, Knollmuller & Ensslin 2019) approximates
the posterior as N(m, M^{-1}). It draws samples via:

```
r = M^{-1} (J^T sqrt(N^{-1}) eta_lh + eta_pr)
```

where `eta_lh ~ N(0, I)` in data-space and `eta_pr ~ N(0, I)` in parameter-space. The
CG solver computes the M^{-1} inversion. These are **linear** samples: straight-line
displacements from the expansion point m.

**geoVI** takes each linear sample and applies the nonlinear curving to follow the
posterior geometry. The result is a sample that lives on the banana, not just near it.

| Aspect | MGVI | geoVI |
|--------|------|-------|
| Sample geometry | Straight lines from m | Curved paths following posterior |
| Posterior shape | Gaussian (symmetric) | Non-Gaussian (captures skewness) |
| Cost per sample | 1 CG solve | 1 CG solve + ~3 Newton iterations |
| Best for | High-D, nearly Gaussian | Nonlinear degeneracies |

### 3.4 The Nonlinear Curving Step

To get a geoVI sample, we need g^{-1}: given a metric sample ms (with covariance M),
find x such that g(x) = ms. This is solved via Newton-CG minimization:

```
minimize_x  phi(x) = 0.5 * ||ms - g(x)||^2
```

The gradient and Hessian of phi are:

```
grad_phi(x) = -(r + L^T(x) L(m) r)      where r = ms - g(x)
hess_phi(x, v) = L^T(m) L(x) v + v + L^T(x) L(m) (L^T(m) L(x) v + v)
```

where `L(x) = sqrt(N^{-1}) J(x)` (right sqrt metric) and `L^T(x) = J^T(x) sqrt(N^{-1})`
(left sqrt metric).

Starting from `x0 = m + r_linear` (the MGVI sample), Newton-CG converges in ~3
iterations because:
1. The linear sample is already close to the solution
2. g is a first-order approximation, so higher-order corrections are small
3. The Hessian of the curving objective is well-conditioned

The curving uses NIFTy's `sampnorm` for gradient convergence:

```python
def sampnorm(natgrad):
    fpp = right_sqrt_metric_flat(m, natgrad)
    return sqrt(dot(natgrad, natgrad) + dot(fpp, fpp))
```

---

## 4. The Five Sample Modes

Each KL iteration in the optimize_kl loop has two stages:
1. **Draw or update samples** (the "scouts")
2. **Minimize KL** with samples held fixed (Newton-CG on the expansion point m)

The sample mode controls stage 1. The scout analogy helps build intuition.

### 4.1 The Scout Analogy

Imagine mapping a mountain landscape in fog. You drop a pin on the map (your current
best guess = expansion point **m**), then send scouts in different directions. Each scout
reports the terrain slope (= KL gradient), and you move the pin based on their reports
(= Newton step).

The quality of your map depends on:
- **Where** the scouts go (sample quality)
- **How** you move the pin (optimization quality)

### 4.2 Mode Reference Table

| Mode | Fresh keys? | Linear draw? | Curve? | Stability | Quality |
|------|:-----------:|:------------:|:------:|:---------:|:-------:|
| `linear_resample` | Yes | Yes | No | Noisy | Low |
| `linear_sample` | No | Yes | No | Stable | Low |
| `nonlinear_resample` | Yes | Yes | Yes | Some noise | High |
| `nonlinear_sample` | No | Yes | Yes | Stable | High |
| `nonlinear_update` | --- | No | Re-curve | Most stable | High* |

*Degrades over many iterations without periodic resample.

### 4.3 linear_resample

**What it does**: Draw fresh random samples from the MGVI approximation N(m, M^{-1}).
New random keys every iteration.

**Scout analogy**: Send brand-new scouts in random directions each round. They walk in
straight lines from the pin.

**When to use**: `fitter.run("vi_linear")`. Fast, good for very high-D problems
where the posterior is nearly Gaussian. Also used for the "cheap" first phase of EVI.

**Implementation**: CG solve of `M @ r = J^T sqrt(N^{-1}) eta_lh + eta_pr` with fresh
`eta_lh, eta_pr ~ N(0, I)`.

### 4.4 linear_sample

**What it does**: Redraw samples from the MGVI approximation, but reuse the same PRNG
keys as the previous iteration. The randomness is the same, but the CG solve uses the
updated expansion point m.

**Scout analogy**: Same scouts, same random directions, but the pin has moved so their
CG paths change slightly.

**When to use**: When you want deterministic MGVI refinement without resampling noise.
Rarely used directly; `nonlinear_update` is usually preferred for geoVI.

### 4.5 nonlinear_resample

**What it does**: Draw fresh MGVI samples (new random keys), then apply geoVI nonlinear
curving to each. This is the standard geoVI sample mode.

**Scout analogy**: Fresh scouts that follow the curvature of the landscape. They walk
along the banana instead of in straight lines.

**Implementation**:
1. Draw linear residuals via CG (same as `linear_resample`)
2. For each residual r and its mirror -r, solve `phi(x) = 0.5 * ||ms - g(x)||^2`
   via Newton-CG to get the curved residual
3. Return 2*n_samples curved residuals (n positive + n mirror)

**When to use**: First iteration of geoVI (establish initial samples), and periodic
refresh iterations (prevent staleness). This is the "most correct" geoVI mode but
also the most expensive and noisiest.

### 4.6 nonlinear_sample

**What it does**: Reuse the same PRNG keys as the previous iteration, but redraw linear
samples and re-curve them. The randomness is fixed, but CG and curving use the updated
expansion point.

**Scout analogy**: Same scouts, re-curved paths at the new pin location.

**When to use**: Deterministic geoVI refinement. Less noise than `nonlinear_resample`,
but more expensive than `nonlinear_update` because it re-solves the full CG.

### 4.7 nonlinear_update

**What it does**: Take the existing curved residuals from the previous iteration and
re-adjust them at the new expansion point m. Does NOT draw new random numbers or re-solve
the CG. Only re-runs the Newton-CG curving step with the previous curved position as
the starting point.

**Scout analogy**: Same scouts, same base randomness, but their curved paths get
adjusted to the new pin location. Cheapest option because the previous curved position
is a good starting point.

**Why it is fast**:
1. No new random numbers (deterministic)
2. The previous curved position is a warm start for the Newton solve
3. Only ~3 Newton iterations needed (often fewer, since the adjustment is small)

**When to use**: The workhorse of converged geoVI. Use between periodic resample
refreshes. Best convergence stability of any mode.

**The staleness problem**: If you use `nonlinear_update` for too many iterations
without resampling, the base randomness becomes stale: the pin has moved far from
where the scouts were originally drawn, and the re-curved adjustments accumulate
error. This manifests as slow drift in the KL value.

---

## 5. The Optimal Resample+Update Schedule

### 5.1 Why Pure Resample Oscillates

Using `nonlinear_resample` every iteration introduces fresh randomness at each step.
While this prevents staleness, the stochastic noise in the KL gradient causes the
expansion point to oscillate rather than converge smoothly. The KL value bounces
around rather than decreasing monotonically.

### 5.2 Why Pure Update Goes Stale

Using `nonlinear_update` every iteration (after an initial resample) gives beautiful
smooth convergence for the first ~5-8 iterations. But then the samples become stale:
they were drawn at an expansion point that is now far away, and the re-curving
adjustment cannot fully compensate. The KL value plateaus or slowly drifts.

### 5.3 The Solution: Periodic Refresh

The optimal strategy combines both: deterministic refinement (update) for stability,
with periodic fresh samples (resample) to prevent staleness.

When you call `fitter.run("vi")` (the default), this is what happens internally:

```
Iteration  1:  nonlinear_resample   <-- fresh curved scouts (establish)
Iteration  2:  nonlinear_update     <-- re-adjust at new pin (refine)
Iteration  3:  nonlinear_update     <-- refine
Iteration  4:  nonlinear_update     <-- refine
Iteration  5:  nonlinear_update     <-- refine
Iteration  6:  nonlinear_resample   <-- FRESH scouts (prevent staleness)
Iteration  7:  nonlinear_update     <-- refine
Iteration  8:  nonlinear_update     <-- refine
Iteration  9:  nonlinear_update     <-- refine
Iteration 10:  nonlinear_update     <-- refine
Iteration 11:  nonlinear_resample   <-- FRESH scouts (prevent staleness)
...
```

Fresh scouts every 5 iterations. Deterministic refinement in between. This gives:
- **Stable convergence** (no oscillation from noisy samples)
- **No staleness** (periodic refresh prevents drift)
- **Good posterior quality** (nonlinear curving captures banana shapes)

The refresh interval of 5 is the default (`_RESAMPLE_EVERY = 5` in `fitter.py`). It can
is fixed at 5 inside `backends/vi/nifty.py`; it is not a user parameter (`OptimizationSchedule` was deleted as dead code, #1293).

---

## 6. Method Hierarchy

tengri provides the same mathematical algorithms through multiple backends that trade
off between diagnostic richness and raw speed.

### Internal Dispatch

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

**Removed names**: `fit_catalog` → `fit_batch`.
Hybrid `geovi_nuts`/`mgvi_nuts` replaced by chaining: `result.refine("mcmc_nuts")`.

### 6.1 vi (Default)

Uses NIFTy's `optimize_kl` for geoVI. Old names (`geovi`, `vi_nifty`) still work but emit a `DeprecationWarning`.

```python
result = fitter.run("vi", n_iterations=15)
```

**Backend**: Pure JAX, fully XLA-compiled. The entire optimization loop runs inside
`jax.lax.while_loop` with zero Python overhead.

**What it does**: JIT-compiled geoVI with the "geovi" sample mode: resample at
iteration 0 and every 5th iteration, nonlinear_update between (via `jax.lax.cond`).

**Posterior samples**: Draws **nonlinear** (geoVI-curved) posterior samples. This
captures non-Gaussian shapes that linear CG samples miss.

**Speed**: ~0.03s/galaxy AFTER one-time ~56s compilation. Best for all use cases,
especially catalog fitting via `fitter.fit_batch(galaxies)`.

**Key differences from NIFTy backends**:
- CG solver uses energy-based convergence with periodic reset (every 20 iterations),
  vs NIFTy's gradient-norm convergence with custom reset strategy
- Newton-CG termination uses energy decrease < 1e-3 after miniter=3, vs NIFTy's xtol
  with `sampnorm` gradient norm
- No `sampnorm`: uses standard L1 gradient norm instead
- Compiles the ENTIRE outer loop into one XLA program (no intermediate Python)
- Supports automatic early stopping when relative KL change < `kl_rtol`

**Expected numerical differences vs NIFTy**:
- Converged expansion point m: agreement within ~1e-3
- Hamiltonian at convergence: agreement within ~0.1
- Posterior standard deviations: agreement within ~10-20% (sampling noise)
- Posterior means: agreement within ~0.5 sigma

**Multi-seed support**: Supports `n_seeds` for running from multiple random starting
points. The best result (lowest Hamiltonian) is returned. Seeds that disagree by >10%
in Hamiltonian trigger a multimodality warning.

### 6.2 vi_linear

Formerly `native_mgvi` / `native_evi`. Old names still work but emit a `DeprecationWarning`.

```python
result = fitter.run("vi_linear", n_iterations=15)
```

Same JIT-compiled backend as `vi` but uses linear resampling throughout. Faster per
iteration; use when geoVI nonlinear resampling is too expensive or for very large N.

### 6.3 vi_nifty

Formerly `nifty_geovi` / `fast_geovi` / `geovi`. Old names still work but emit a `DeprecationWarning`.

```python
result = fitter.run("vi_nifty", n_iterations=15)
```

**Backend**: NIFTy's `OptimizeVI.update` in a tight Python loop.

**What it does**: Calls NIFTy's exact CG, Newton-CG, line search, and sampnorm
implementations. Strips logging, pickling, stdout capture, and callbacks for ~35%
speedup over the full `jft.optimize_kl`.

**Math**: Identical to NIFTy. Same CG convergence criteria, same `sampnorm`, same
line search with steepest-descent reset.

**Sample schedule**: Uses the optimal resample+update schedule (Section 5.3). The
`sample_mode` is resolved as a callable `f(i) -> str` that returns `"nonlinear_resample"`
at iteration 0 and every 5 iterations, `"nonlinear_update"` otherwise.

**Posterior samples**: By default draws **nonlinear** (geoVI-curved) posterior samples
via the JIT engine. This captures non-Gaussian shapes that linear CG samples miss.

**Speed**: ~12s/galaxy for smooth SFH (D~7), ~30s for stochastic SFH (D~137).

### 6.4 vi_nifty_linear

Formerly `nifty_mgvi` / `fast_mgvi` / `mgvi`. Old names still work but emit a `DeprecationWarning`.

```python
result = fitter.run("vi_nifty_linear", n_iterations=15)
```

Same as `vi_nifty` but uses `linear_resample` every iteration. Faster per iteration
(no Newton-CG curving), but the Gaussian approximation misses nonlinear degeneracies.

Posterior samples are linear CG draws (Gaussian).

### 6.5 vi (Expansion-point schedule)

The `vi` method automatically uses an expansion-point schedule: runs linear resampling
for the first half of iterations, then switches to nonlinear geoVI for the second half.

Rationale: early iterations explore far from the optimum, where nonlinear curving adds
cost but little benefit. Once the expansion point is roughly converged, the nonlinear
correction becomes valuable.

The transition point is controlled by `VIConfig.evi_linear_fraction` (default 0.5).

### 6.6 vi_nifty (full logging)

```python
result = fitter.run("vi_nifty", n_iterations=15)
```

**Backend**: Full `jft.optimize_kl` with all NIFTy diagnostics, logging, and minisanity
checks enabled.

**When to use**: Debugging, validating against reference NIFTy behavior, or when you
need detailed per-iteration diagnostics that the fast backend strips.

**Speed**: ~18s/galaxy (roughly 35% slower than the fast backend due to Python overhead).

### 6.7 VI → MCMC chaining (formerly geovi_nuts)

```python
result = fitter.run("vi", n_iterations=10)
result_exact = result.refine("mcmc_nuts", n_posterior_samples=500)
```

**What it does**: Runs `vi` optimization to find the posterior mode and approximate
covariance, then draws posterior samples via BlackJAX NUTS starting from the converged
position.

**When to use**: When you want exact (MCMC) posterior samples but need a good starting
point. The VI phase provides initialization and mass matrix for NUTS, dramatically
reducing warmup time.

**Posterior samples**: Independent MCMC samples via NUTS. Not restricted to the Gaussian
or geoVI-curved approximation.

### 6.8 mcmc_raytrace

Formerly `raytrace`. Old name still works but emits a `DeprecationWarning`.

```python
result = fitter.run("mcmc_raytrace", n_burnin=100, n_steps=500, n_leapfrog_steps=10)
```

**Ray Tracing Sampler** (Behroozi 2025). Propagates light rays through a medium where
the refractive index `n(x) = L(x)^{1/(D-1)}`, using Snell's law to bend rays toward
high-likelihood regions. An exact MCMC sampler that is particularly resilient to
stochastic gradients.

**Key parameters**:
- `step_size`: Integration step size. Default `0.03 * sqrt(D)` for D<=10, `0.01` for D>10.
  For stochastic SFH models (D~137), there is a sharp viability cliff at step_size~0.06
  where acceptance drops from ~98% to 0%. Use `step_size=0.05, n_leapfrog_steps=50,
  n_steps=2000`.
- `n_leapfrog_steps`: Leapfrog integration steps per trajectory.
- `refresh_rate`: Partial momentum refresh. 0 = pure ray tracing.

**Convergence diagnostics**:
- Acceptance rate: 30-70% ideal; >90% means barely moving
- ESS (bulk): >100 per parameter, >400 total (Vehtari et al. 2021)

### 6.9 mcmc_nuts

Formerly `nuts`. Old name still works but emits a `DeprecationWarning`.

```python
result = fitter.run("mcmc_nuts", n_warmup=500, n_samples=1000)
```

**NUTS** (No-U-Turn Sampler) via BlackJAX. The gold standard for low-dimensional
validation. Uses window adaptation to tune step size and mass matrix during warmup.

**Phases**:
1. **Warmup** (adaptation): tunes step size and mass matrix
2. **Burn-in** (optional): additional post-warmup steps, discarded
3. **Sampling**: posterior samples collected

**Warning**: NUTS is computationally expensive for stochastic SFH models (D~137). A
warning is issued when `spec.stochastic` is True.

**Convergence diagnostics**:
- Acceptance rate: ~80% target
- Divergences: 0 ideal; >5% = serious problem
- ESS (bulk): >100 per parameter

### 6.10 map

```python
result = fitter.run("map", n_steps=1500, optimizer="adam", learning_rate=0.02)
```

**MAP** (Maximum A Posteriori) point estimate via gradient descent. No posterior samples.

**Optimizers**: `"adam"` (default), `"adamw"`, `"sgd"`, or any pre-built optax optimizer.

**Features**:
- Early stopping: halts when loss doesn't improve by `rtol` over `patience` steps
- Returns `loss_history` for convergence diagnostics

**Use case**: Initialization for MCMC or VI methods. A quick MAP run provides a good
starting point that dramatically improves convergence:

```python
result_map = fitter.run("map", n_steps=1000)
result = fitter.run("vi", init_from=result_map, n_iterations=10)
```

---

## 7. Compilation and Caching

### 7.1 When JAX Recompiles

JAX traces and compiles a function the first time it is called with a new combination
of **static arguments**. In tengri, the following are static:

| Argument | Effect |
|----------|--------|
| `sample_mode` (str) | Separate XLA program per mode |
| `n_iterations` (int) | Recompiles if changed |
| `n_samples` (int) | Recompiles if changed |

This means:
- `fitter.run("vi", n_iterations=15, n_samples=3)` compiles once
- Calling again with the same settings is instant
- Calling with `n_iterations=20` triggers recompilation

### 7.2 XLA Persistent Cache

tengri enables the XLA persistent compilation cache at import time:

```
~/.cache/tengri_jax_cache
```

Compiled XLA programs are stored on disk and survive Python restarts. The first
invocation in a new session may still take a few seconds to deserialize from cache,
but this is much faster than recompilation.

### 7.3 Ahead-of-Time Compilation via fitter.compile()

For catalog fitting or interactive notebooks, you can pre-compile all needed modes:

```python
fitter = Fitter(model, data, noise)

# Compile default modes (MGVI + geoVI update): ~3s
fitter.compile()

# Compile all modes including full geoVI: ~60s
fitter.compile(
    modes=("linear_resample", "nonlinear_update", "nonlinear_resample"),
    n_iterations=15,
    n_samples=3,
    n_posterior_samples=200,
)

# Now all runs are instant
result = fitter.run("vi")  # no compilation delay
```

`fitter.compile()` pre-compiles:
- The optimization loop for each specified `sample_mode`
- The MGVI optimizer (old path)
- The posterior sample draw function

### 7.4 Compilation Costs by Mode

| What compiles | Time | Notes |
|---------------|------|-------|
| JIT engine init (`_build_jit_engine`) | ~2s | Builds all closures and NIFTy objects |
| `linear_resample` optimization | ~0.03s | Simple CG path |
| `nonlinear_update` optimization | ~3s | Includes curving Newton-CG |
| `nonlinear_resample` optimization | ~56s | Full geoVI with CG + curving |
| Posterior draw (200 samples) | ~1s | CG-based residual drawing |
| `"geovi"` mode (lax.cond) | ~56s | Traces both resample and update branches |

The `"geovi"` sample mode (used by `vi`, the default) uses `jax.lax.cond`
to dynamically choose between resample and update. This traces both branches, incurring
the full 56s cost. The fast/NIFTy backends avoid this by dispatching in Python.

### 7.5 MCMC Compilation Architecture and Design Philosophy

#### Why `lax.scan` (not a Python for-loop)

```python
# BAD: Python for-loop — unrolls the entire graph at trace time
for k in keys:
    state, info = kernel(k, state, ld, step_size, inv_mass_matrix, max_doublings)
    positions.append(state.position)

# GOOD: lax.scan — compiles the loop body once, executes N times
_, (positions, divergent) = jax.lax.scan(_step, state, chain_keys)
```

`jax.lax.scan` compiles the loop body (`_step`) exactly once into an XLA `while` primitive.
The resulting XLA program is compact and runs at hardware speed. A Python for-loop at trace
time produces a giant unrolled graph: compile time and memory scale linearly with N, and
the BlackJAX NUTS tree-builder (`lax.while_loop`) gets compiled N times — once per
iteration. For 1000 samples this is catastrophically slow.

#### Why `data_args` is a traced tensor (not closed over)

The log-posterior is structured as:

```python
def log_posterior_flat_2arg(position, data_args):
    return logdensity_2arg(unravel_fn(position), data_args)
```

`data_args` flows into the JIT-compiled program as a **traced** argument — it is part
of the input values, not burned into the XLA bytecode. This means:

- Switching to a new galaxy with the same model shape **never** triggers recompilation.
- The compiled XLA program is galaxy-agnostic: only the model structure (parameter count,
  filter count, SFH grid size) matters.
- Benchmark evidence: `A1_new_gal mcmc_nuts = 0.08s` — a new galaxy runs at cached speed.

#### Why `logdensity_fn_2arg` is a static argument

```python
@functools.partial(jax.jit, static_argnums=(3, ...))
def _nuts_full_scan(init_flat, warmup_key, chain_keys, logdensity_fn_2arg, data_args, ...):
```

JAX uses **function identity** (Python object `id`) as part of the XLA cache key for
static arguments. Marking `logdensity_fn_2arg` as static ensures:

1. The XLA program is model-specific — two different models with the same D get separate
   compiled programs (correct behavior).
2. The same model reuses the same compiled program — no recompilation when `data_args` change.
3. `functools.cache` on `_get_nuts_kernel()` guarantees kernel object identity across calls.

#### Previous design (3 JITs) vs current design (1–2 JITs)

Old architecture per cold NUTS run:

| Phase | Function | JIT trigger |
|-------|----------|-------------|
| Warmup | `blackjax.window_adaptation.run()` | Internal `lax.scan`, compiles NUTS kernel |
| Burn-in | `_nuts_burnin_scan()` | `@jax.jit`, recompiles NUTS kernel |
| Sampling | `_nuts_sample_scan()` | `@jax.jit`, recompiles NUTS kernel |

The NUTS tree-builder (`lax.while_loop`) compiled **3×** per cold run.

New architecture:

| Path | Function | JIT trigger |
|------|----------|-------------|
| Cold (no cached adaptation) | `_nuts_full_scan()` | Single outer `jax.jit` — kernel compiled **once** |
| Warm (cached adaptation) | `_nuts_chain_scan()` | Single outer `jax.jit` — kernel compiled **once** |

`_nuts_full_scan` wraps `window_adaptation.run()` **and** the chain scan inside a single
outer `jax.jit`. Because JAX traces the entire body as one program, `blackjax`'s internal
`lax.scan` (warmup) and the chain `lax.scan` see the same compiled NUTS kernel — the
`lax.while_loop` tree-builder is emitted once into the XLA bytecode.

#### XLA persistent disk cache

XLA compiled programs are serialized to `~/.cache/tengri_jax_cache` on first run. On
subsequent Python interpreter restarts, JAX deserializes the cached bytecode rather than
recompiling. This means:

- **First run** (cold, no disk cache): ~23s for NUTS D=7 (down from ~69s with 3 JITs).
- **Second run** (disk cache hit): <1s deserialization.
- The cache survives across Python restarts as long as JAX version and function identity
  (`logdensity_fn_2arg`) are stable.

#### Pre-warming with `fitter.compile()`

```python
# Pre-warm the MCMC XLA cache before interactive notebook use:
fitter.compile(
    mcmc_methods=["nuts"],   # or ["hmc", "dynamic_hmc", "ghmc"]
    n_warmup=300,
    n_burnin=100,
    n_mcmc_samples=100,
)
# Now restart Python — first NUTS call deserializes in <1s.
result = fitter.run("mcmc_nuts", n_warmup=300, n_burnin=100, n_samples=1000)
```

**Cache key contract**: The XLA program is keyed on `logdensity_fn_2arg` identity,
`n_warmup`, `n_burnin`, `n_mcmc_samples` (chain length = n_burnin + n_samples), `use_dense`,
and `max_doublings`. Use the same values in `compile()` as in `run()` to guarantee a hit.

#### Expected performance impact

| Scenario | Before | After |
|----------|--------|-------|
| Cold NUTS D=7, no XLA cache | ~69s (3 kernel compiles) | ~23s (1 compile) |
| Warm NUTS (cached adaptation) | 2 JIT calls | 1 JIT call |
| Fresh kernel restart, XLA cache populated | ~69s | <1s (deserialize) |
| After `fitter.compile(mcmc_methods=["nuts"])` | cold on first use | instant on first use |
| New galaxy, same model | 0.08s (traced data_args) | 0.08s (unchanged) |

---

## 8. Performance Benchmarks

All benchmarks on MacBook Pro M-series, CPU (`JAX_PLATFORMS=cpu`).

### 8.1 Forward Model

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|-------------|-------------------|
| Forward model | 140 us | 356 us |
| Gradient | 56 us | 63 us |

### 8.2 Inference Methods

| Method | D=7 (smooth) | D=137 (stochastic) | Notes |
|--------|-------------|-------------------|-------|
| `map` (1000 steps) | ~3s | ~5s | Adam optimizer |
| `vi_nifty` (10 iter) | ~12s | ~30s | NIFTy exact math |
| `vi_nifty` full logging (10 iter) | ~18s | ~45s | Full NIFTy with logging |
| `vi` (10 iter) | 56s compile + 0.3s run | 56s compile + 0.8s run | First call only |
| `vi_linear` (10 iter, 2000 samp) | 11s | 14s | JIT-compiled |
| `mcmc_raytrace` (500 steps) | ~10s | ~120s | Depends on step_size |
| `mcmc_nuts` (500+1000) | ~30s | ~300s+ | Expensive for high-D |

### 8.3 Catalog Fitting Estimates

For fitting N galaxies sequentially with the native JIT backend:

| N galaxies | Compile (one-time) | Per-galaxy run | Total |
|-----------|-------------------|---------------|-------|
| 1 | 56s | 0.03s | 56s |
| 10 | 56s | 0.03s | 56.3s |
| 100 | 56s | 0.03s | 59s |
| 1,000 | 56s | 0.03s | 86s |
| 10,000 | 56s | 0.03s | 356s (~6 min) |

The native backend amortizes the compilation cost across galaxies. For 100+ galaxies,
the per-galaxy cost dominates and the effective throughput is ~30ms/galaxy.

### 8.4 Posterior Sampling Speed

| Method | Speed | Notes |
|--------|-------|-------|
| JIT CG (linear) | ~0.2ms/sample | Default for MGVI |
| JIT geoVI (nonlinear) | ~5ms/sample | Default for geoVI |
| NIFTy draw_linear_residual | ~540ms/sample | Slow Python-loop CG |
| BlackJAX NUTS | ~2ms/sample | After warmup |

---

## 9. Posterior Sampling

After the optimization phase converges, posterior samples are drawn from the approximate
posterior. The choice of sampling method determines posterior quality.

### 9.1 Linear CG Draws (MGVI)

The default for MGVI methods. Draws samples from N(m, M^{-1}) by solving:

```
M @ residual = J^T sqrt(N^{-1}) eta_lh + eta_pr
```

where `eta_lh, eta_pr ~ N(0, I)`. The sample is `m + residual`.

**Pros**: Fast (~0.2ms/sample), exact for Gaussian posteriors.
**Cons**: Misses non-Gaussian features (banana shapes, skewness, multimodality).

### 9.2 Nonlinear geoVI-Curved Draws

The default for geoVI methods (`posterior_method="nonlinear"`). Draws linear residuals
via CG, then applies the geoVI coordinate curving to each:

1. Draw linear residual r via CG
2. Compute metric sample ms = draw_metric_sample(m, key)
3. Solve `phi(x) = 0.5 * ||ms - g(x)||^2` for x (Newton-CG, ~3 iterations)
4. Return x - m as the curved residual

**Pros**: Captures non-Gaussian posterior shapes. Respects the age-dust-metallicity
banana. Typically 10-50x better coverage of the true posterior tails.
**Cons**: Slower (~5ms/sample). Still limited to the geoVI approximation (cannot
discover disconnected modes).

### 9.3 NUTS Posterior Samples

Available via `geovi_nuts` and `mgvi_nuts`. After VI optimization, runs BlackJAX NUTS
with 200 warmup steps starting from the converged position.

**Pros**: Exact MCMC samples, not limited by any VI approximation. Can discover features
that geoVI misses.
**Cons**: Slower warmup (~2s), sequential sampling. Expensive for high-D.

### 9.4 Why Nonlinear Matters

For a concrete example: in the dust_tau_bc vs met_logzsol plane, the true posterior is
a curved banana due to the age-dust-metallicity degeneracy. With 200 samples:

- **Linear CG**: Samples form an ellipse. The 95% credible region misses the curved
  tails, underestimating uncertainty by 20-30%.
- **Nonlinear geoVI**: Samples follow the banana. The 95% credible region matches
  the NUTS ground truth within sampling noise.

This is why `geovi` defaults to `posterior_method="nonlinear"`.

---

## 10. Block Gibbs for Hierarchical Models

### 10.1 The Problem

Hierarchical inference jointly fits shared population parameters (PSD amplitude, PSD
timescale) and per-galaxy parameters (dust, metallicity, SFH). The total parameter
dimension scales as:

```
D_total = D_shared + N_gal * (D_phys + D_xi)
```

For 100 galaxies: D_total = 2 + 100 * (7 + 130) = 13,702 parameters.

Fitting all parameters jointly is inefficient because:
- The PSD parameters affect all galaxies (global coupling)
- Per-galaxy physical parameters have nonlinear degeneracies (need geoVI)
- Per-galaxy SFH fields are nearly Gaussian conditioned on physical params (MGVI suffices)

### 10.2 The Three-Block Structure

Block Gibbs cycling updates parameter blocks in rotation, holding others fixed:

#### Block 1: Shared PSD Parameters (highest priority)

- **Updated**: `psd_sigma_u`, `psd_tau_u` (2 parameters)
- **Frozen** (`constants`): all per-galaxy physical params + all per-galaxy xi
- **Point estimates** (`point_estimates`): per-galaxy params (zero residuals for speed)
- **Sample mode**: `nonlinear_resample` (PSD response is nonlinear)
- **n_samples**: 6 (more samples for precise population constraint)
- **Why first**: PSD parameters set the prior for every galaxy's SFH. Getting them
  right first propagates correct information downstream.

#### Block 2: Per-Galaxy Physical Parameters (medium priority)

- **Updated**: all `gal_phys` across all galaxies (N_gal x ~7 params)
- **Frozen** (`constants`): `psd_sigma_u`, `psd_tau_u`, all `gal_xi`
- **Sample mode**: `nonlinear_resample` (age-dust degeneracy needs geoVI)
- **n_samples**: 3
- **Why second**: Physical params interpret each galaxy's data given the current SFH
  prior. Conditioned on PSD + xi, the per-galaxy physical problems are independent.

#### Block 3: Per-Galaxy SFH Fields (lowest priority)

- **Updated**: all `gal_xi` across all galaxies (N_gal x n_grid params)
- **Frozen** (`constants`): `psd_sigma_u`, `psd_tau_u`, all `gal_phys`
- **Sample mode**: `linear_resample` (MGVI --- GP response is nearly linear)
- **n_samples**: 2 (cheap, high-D)
- **Why last**: SFH fields are the "leaf" of the hierarchy. Nearly Gaussian conditioned
  on PSD + physical params.

### 10.3 Parameter Layout

The hierarchical flat array has this layout:

```
[psd_sigma_u, psd_tau_u, gal0_phys..., gal0_xi..., gal1_phys..., gal1_xi..., ...]
 <--- shared --->  <-------------- per-galaxy (N_gal repeats) ------------------>
```

### 10.4 Constants vs Point Estimates

Both mechanisms freeze parameters during KL minimization (their gradient is zeroed),
but they differ in sampling:

| Mechanism | Gradient | Residual | Speed | Uncertainty propagation |
|-----------|----------|----------|-------|------------------------|
| `constants` | Zeroed | Kept (sampled) | Normal | Yes (uncertainty of frozen params propagates) |
| `point_estimates` | Zeroed | Zeroed | Faster | No (treats frozen params as known exactly) |

**Use `constants`** for parameters whose uncertainty matters for the block being updated.
Example: when updating per-galaxy physical params, the shared PSD params are frozen as
`constants` so their uncertainty propagates into the per-galaxy posteriors.

**Use `point_estimates`** for parameters whose uncertainty is less relevant for the block
being updated, and speed matters. Example: when updating the 2D shared PSD params, the
per-galaxy params (13,700D) are frozen as `point_estimates` because their individual
uncertainties barely affect the population-level PSD constraint.

### 10.5 Resample+Update Within Blocks

Each block follows the same resample+update pattern as individual geoVI:

```
Outer cycle 0: nonlinear_resample for blocks 1+2, linear_resample for block 3
Outer cycles 1-4: nonlinear_update for blocks 1+2, linear_sample for block 3
Outer cycle 5: resample again (prevent staleness)
```

### 10.6 Usage

```python
from tengri import HierarchicalFitter

hfitter = HierarchicalFitter(
    model_factory=lambda sigma, tau: Model(spec, ssp, psd_sigma=sigma, psd_tau_myr=tau),
    galaxies=[{"flux_obs": f, "noise": n} for f, n in zip(fluxes, noises)],
)

# Default: native_geovi with CorrelatedFieldMaker
result = hfitter.run("native_geovi", n_iterations=25)

# EVI (MGVI first, then geoVI)
result = hfitter.run("native_evi", n_iterations=30)
```

### 10.7 Expected Performance (Hierarchical)

For N=100 galaxies, D_total = 13,702:

| Block | Dimension | Per-iteration cost |
|-------|-----------|-------------------|
| Block 1 (shared PSD) | 2 | ~0.01ms |
| Block 2 (physical) | 700 | ~0.5ms |
| Block 3 (SFH xi) | 13,000 | ~2ms (MGVI, no curving) |
| **Total per outer cycle** | | **~2.5ms** |

25 outer iterations x 3 blocks = 75 total steps: ~190ms.
Compile time: ~60s (one-time, cached to XLA disk cache).

---

## 11. Convergence Diagnostics

### 12.1 Chi-squared per Degree of Freedom

The most basic diagnostic. After fitting:

```python
result = fitter.run("native_geovi", n_iterations=15)
print(result.diagnostics["chi2_dof"])
```

| chi2/dof | Interpretation |
|----------|---------------|
| ~1.0 | Good fit (residuals consistent with noise) |
| <0.5 | Overfitting or overestimated noise |
| 2-5 | Mediocre fit, may need more iterations |
| >5 | Poor fit: model mismatch or bad initialization |

The fitter automatically warns if chi2/dof > 5.0.

### 12.2 Parameters at Bounds

The fitter checks if any parameter's posterior median is within 2% of its prior bounds.
If so, a diagnostic warning is issued:

```
Parameters near bounds: dust_tau_bc, met_logzsol. Consider widening the prior.
```

This often indicates that the data prefers values outside the prior range.

### 12.3 Posterior Predictive Check

Generate model predictions from posterior samples and verify they bracket the data:

```python
result = fitter.run("native_geovi", n_iterations=15)

for i in range(10):
    sample = {k: v[i] for k, v in result.samples.items()}
    pred = model.predict_photometry(sample)
    # pred should bracket the data within noise
```

### 12.4 Ray Tracing Diagnostics

| Diagnostic | Threshold | Meaning |
|-----------|-----------|---------|
| Acceptance rate | 30-70% | Fraction of proposals accepted |
| Acceptance > 90% | Bad | Chain barely moving (step_size too small) |
| Acceptance < 10% | Bad | Chain stuck (step_size too large) |
| ESS (bulk) | >100/param, >400 total | Effective sample size |

### 12.5 NUTS Diagnostics

| Diagnostic | Threshold | Meaning |
|-----------|-----------|---------|
| Divergences | 0 ideal, >5% serious | Tree hit pathological curvature |
| Acceptance rate | ~80% target | Tuned by adaptation |
| ESS (bulk) | >100/param | Effective sample size |

### 12.6 Known Difficult Parameters

`dust_tau_bc`, `dust_tau_diff`, and `met_logzsol` consistently have low ESS across
all methods due to the age-dust-metallicity degeneracy. This is a physical limitation
of SED fitting, not a sampler bug. The posterior in these parameters is highly
correlated and banana-shaped.

### 12.7 geoVI/MGVI Diagnostics

For variational methods, check:
- **KL convergence**: the Hamiltonian should decrease monotonically and plateau
- **Multi-seed agreement**: `n_seeds > 1` in the native backend; seeds that disagree
  in Hamiltonian by >10% suggest multimodality
- **Comparison to MCMC**: for validation, run `raytrace` or `nuts` and compare posteriors

For geoVI/MGVI: check KL convergence across iterations and compare to RT posteriors
when possible. Use `convergence_check()` or `convergence_table()` from
`notebooks/_plot_style.py` for standardized diagnostics (Vehtari et al. 2021;
Stan/ArviZ/BlackJAX thresholds).

---

## 12. Quick Reference

### Method Selection Cheat Sheet

```
Need a point estimate?          --> fitter.run("map")
Need speed, D < 50?             --> fitter.run("native_geovi")   (default)
Need speed, D > 1000?           --> fitter.run("native_mgvi")
Need accuracy, D < 20?          --> fitter.run("nuts")
Need accuracy, D > 20?          --> fitter.run("raytrace")
Need speed + accuracy?          --> fitter.run("native_geovi")   (default, nonlinear draws)
Need exact samples + speed?     --> fitter.run("geovi_nuts")
Catalog of 100+ galaxies?       --> fitter.compile(); fitter.fit_batch(galaxies)
Hierarchical (shared PSD)?      --> hfitter.run("native_geovi")
NIFTy-exact math needed?        --> fitter.run("geovi")          (NIFTy tight loop)
Debugging NIFTy behavior?       --> fitter.run("nifty_geovi")    (full logging)
```

### VIConfig Defaults (Philipp Frank's Recommendations)

```python
VIConfig(
    n_samples=3,                  # 3 samples x 2 (mirror) = 6 effective
    n_iterations=10,              # KL iterations
    draw_linear_kwargs={
        "cg_name": "SL",
        "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30},
    },
    nonlinearly_update_kwargs={
        "minimize_kwargs": {
            "name": "SN", "xtol": 1e-3,
            "cg_kwargs": {"name": None}, "maxiter": 3,
        },
    },
    kl_kwargs={
        "minimize_kwargs": {
            "name": "M", "absdelta": 1e-3,
            "cg_kwargs": {"name": "MCG"}, "maxiter": 10,
        },
    },
)
```

### Common Gotchas

- `n_samples > 12` triggers a warning: high sample counts reduce stochastic
  regularization and cause Newton-CG overshooting. Use 3-6.
- `n_iterations > 100` with `kl_rtol=0` triggers a warning: risk of divergence.
- Never create `Model`/`ParamSpec` inside a JAX gradient tape.
- The native backend compiles the entire loop atomically --- no intermediate diagnostics.
- Ray Tracing step_size has a sharp cliff at ~0.06 for D~137.

---

## 13. References

- Frank, P., Leike, R., Ensslin, T.A. (2021). "Geometric Variational Inference."
  Entropy 23(7):853. arXiv:2105.10470
- Knollmuller, J., Ensslin, T.A. (2019). "Metric Gaussian Variational Inference."
  arXiv:1901.11033
- Edenhofer, G. et al. (2024). "Re-envisioning Numerical Information Field Theory
  (NIFTy.re)." arXiv:2402.16683
- Behroozi, P. (2025). "Ray Tracing Sampler." Apache 2.0 license.
- Vehtari, A. et al. (2021). "Rank-normalization, folding, and localization."
  Bayesian Analysis 16(2):667-718. (ESS/Rhat diagnostics)
