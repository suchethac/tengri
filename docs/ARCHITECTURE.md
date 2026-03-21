# diffsed Architecture Guide

> For AI agents and developers making code changes. Read this before modifying inference, distributions, or the forward model.

## Core Principle: Full Standardization

Every parameter — physical, PSD, and latent field — is represented by a standardized latent variable `ξ ~ N(0, I)`. The prior is absorbed into the forward model via differentiable transforms. The loss is always:

```
H(ξ) = ½ χ²(data, f(ξ)) + ½ ξᵀξ
```

No separate prior penalty terms. No per-distribution special cases.

## Dependency Layers

```
Layer 0 (always): JAX, DSPS, h5py, matplotlib, scipy
  → Forward model, mock generation, SFH prediction, plotting

Layer 1 (MAP): + optax
  → Gradient descent optimization

Layer 2 (Ray Tracing): no extra deps
  → raytrace_jax.py (Behroozi 2025, Apache 2.0)

Layer 3 (NUTS): + blackjax
  → Hamiltonian Monte Carlo

Layer 4 (geoVI/MGVI/hierarchical): + nifty8.re
  → Variational inference, CorrelatedFieldMaker, optimize_kl
```

**Rule:** Never import nifty8.re at module level. Always use lazy imports inside methods that need it. The forward model, distributions, and non-VI samplers must work without NIFTy installed.

## Module Map

```
src/diffsed/
├── distributions.py        # Prior distributions with standardized transforms
│                           #   Each has: sample(), log_prob(), unstandardize(ξ→θ), standardize(θ→ξ)
│                           #   Built-in: Uniform, Gaussian, LogUniform, LogNormal, StudentT, Fixed
│                           #   Custom: implement unstandardize() — must be JAX-differentiable
│
├── param_spec.py           # ParamSpec: single source of truth for all parameters
│                           #   Separates free vs fixed, validates, provides sampling
│
├── model.py                # Model: high-level forward model
│                           #   predict_photometry(), predict_spectrum(), predict_sfh()
│                           #   predict_derived(), mock(), mock_batch()
│                           #   Wraps ForwardModel with clean parameter names
│                           #   Accepts _correlated_field key to bypass internal GP
│
├── standardized.py         # StandardizedForwardModel: ξ → observables
│                           #   Absorbs priors into forward model via Distribution.unstandardize()
│                           #   build_standardized_loss(): H = ½χ² + ½ξᵀξ
│                           #   build_hierarchical_loss(): shared params across N galaxies
│                           #   Accepts custom PSD model (DRW default, user-swappable)
│
├── fitter.py               # Fitter: inference engine
│                           #   run("map"), run("raytrace"), run("nuts"), run("geovi"), run("mgvi")
│                           #   All methods return Posterior objects
│                           #   Burns in properly (n_burnin for RT and NUTS)
│
├── raytrace_jax.py         # Ray Tracing Sampler (Behroozi 2025)
│                           #   Snell's law MCMC: n(x) = L(x)^{1/(D-1)}
│                           #   250× more gradient-noise resilient than HMC
│
├── posterior.py             # Posterior: results + diagnostics
│                           #   summary(), effective_sample_size(), autocorrelation()
│                           #   plot_corner() with overlay support, to_arviz(), resample()
│
├── hierarchical.py         # HierarchicalFitter: population-level PSD
│                           #   Uses CorrelatedFieldMaker for joint PSD learning (geoVI)
│                           #   Also supports flat-vector RT and geoVI
│
├── forward_model.py        # Low-level ForwardModel (internal, legacy API)
├── models/                 # Physics modules (SFH, dust, SPS, observation)
└── utils/                  # Transforms, grid, cosmology, precomputation
```

## Parameter Flow

```
User defines ParamSpec:
  sfh_alpha = Uniform(0.5, 3.0)    ← Distribution object
  met_logzsol = Gaussian(-0.5, 0.3)
  psd_sigma = LogNormal(0.0, 0.8)  ← can be free or fixed
  dust_tau_diff = 0.3              ← Fixed (scalar shorthand)

                    ↓

StandardizedForwardModel builds ξ→θ mapping:
  ξ_alpha   → sigmoid(ξ)·(3.0-0.5)+0.5     [Uniform.unstandardize]
  ξ_met     → clip(-0.5+0.3·ξ, -2, 0)      [Gaussian.unstandardize]
  ξ_sigma   → exp(0.0+0.8·ξ)               [LogNormal.unstandardize]
  ξ_field   → IFFT(√P(σ,τ)·ξ_field)        [correlated field]

                    ↓

Model.predict_photometry(params) → flux array

                    ↓

Loss: H = ½ Σ((data-flux)/noise)² + ½ ξᵀξ
```

## Distribution Protocol

To add a new prior, implement a Distribution subclass with these methods:

```python
class MyPrior(Distribution):
    @property
    def bounds(self) -> tuple[float, float]: ...
    def sample(self, key) -> jnp.ndarray: ...          # for mock generation
    def log_prob(self, x) -> jnp.ndarray: ...          # for diagnostics
    def unstandardize(self, xi) -> jnp.ndarray: ...    # ξ~N(0,1) → θ  [MUST be JAX-differentiable]
    def standardize(self, theta) -> jnp.ndarray: ...   # θ → ξ  [for initialization]
```

**Critical:** `unstandardize()` must be differentiable via `jax.grad`. This is used inside the forward model during inference. Test with:
```python
g = jax.grad(lambda xi: dist.unstandardize(xi))(jnp.array(0.5))
assert jnp.isfinite(g)
```

## NIFTy.re Integration

When geoVI/MGVI is used, we leverage NIFTy's infrastructure:

| Our code | NIFTy equivalent | How we use it |
|----------|-----------------|---------------|
| `Distribution.unstandardize()` | `jft.NormalPrior`, `jft.LogNormalPrior`, etc. | Same concept — both map ξ→θ |
| `StandardizedForwardModel` | `jft.Model` | Wrap ours as jft.Model for optimize_kl |
| Chi² likelihood | `jft.Gaussian` | Use theirs directly |
| `optimize_kl` | `jft.optimize_kl` | Use theirs directly |
| `draw_linear_residual` | `jft.draw_linear_residual` | Use theirs for cheap posterior samples |
| Correlated field | `jft.CorrelatedFieldMaker` | Use theirs for joint PSD learning |

**Rule:** For geoVI/MGVI inference, use NIFTy's implementations — they are optimized and battle-tested. Our code wraps them with a user-friendly API.

## Correlated Field and PSD

The SFH fluctuation field `x(t)` is generated as:
```
x(t) = IFFT(√P(σ_PSD, τ_PSD) · ξ_field)
```
where `ξ_field ~ N(0, I)` and `P` is the power spectral density.

**When PSD is fixed:** `√P` is precomputed once. The field depends only on `ξ_field`.

**When PSD is free:** `√P` depends on `ξ_sigma` and `ξ_tau` through their unstandardize transforms. The field depends on `(ξ_sigma, ξ_tau, ξ_field)` — this is the natural PSD-field coupling.

**PSD model is swappable:** The `StandardizedForwardModel` accepts a `psd_model` callable:
```python
def my_psd(sigma, tau_yr, n_grid, log_ages):
    # Extended Regulator, Flex-PSD, Matérn, etc.
    return sqrt_power_array
smodel = StandardizedForwardModel(model, psd_model=my_psd)
```

## Inference Method Hierarchy

| Method | Module | Dependency | Exact? | Best D |
|--------|--------|------------|--------|--------|
| MAP | fitter.py | optax | No (point est) | Any |
| Ray Tracing | raytrace_jax.py | — | Yes | ≤300 |
| NUTS | fitter.py | blackjax | Yes | ≤20 |
| geoVI | fitter.py | nifty8.re | Approximate | ≤10⁵ |
| MGVI | fitter.py | nifty8.re | Approximate | ≤10⁶ |

**Ray Tracing and geoVI are equal-priority primaries.** NUTS validates. MAP initializes.

**Step sizes for Ray Tracing:**
- D ≤ 10: `0.03 * sqrt(D)` (Behroozi 2025 recommendation)
- D > 10: `0.01`
- D > 100 (hierarchical): `0.005`

**NIFTy geoVI best practices (from literature):**
- Use 4-12 samples per KL iteration, not 80
- Convergence: mean squared weighted deviation < 1.05 for 3 consecutive iterations
- Use `nonlinear_resample` for geoVI, `linear_resample` for MGVI

## Hierarchical Inference

Shares PSD hyperparameters across N galaxies. Each galaxy has its own `ξ_field_i` and `ξ_phys_i`.

**Three approaches available:**

1. **CorrelatedFieldMaker + native_geovi** (`hfitter.run("native_geovi")`) — Recommended (default). PSD hyperparameters are part of the generative model. JIT-compiled with resample+update schedule.

2. **native_mgvi** (`hfitter.run("native_mgvi")`) — Same but faster per iteration. For very large N.

3. **Ray Tracing** (`hfitter.run("raytrace")`) — Flat vector, MAP initialization per galaxy. Works for small N.

Batch fitting: `fitter.fit_batch(galaxies)` — default method is `native_geovi`.

## Key Gotchas

1. **`hash()` overflow:** `jax.random.fold_in(key, hash(string))` fails. Use `abs(hash(x)) % (2**31)`.

2. **No Model creation inside gradient tape:** `ParamSpec.__init__` with JAX-traced values fails. Pre-build models outside differentiable functions.

3. **Metallicity offset:** SSP grid is `log10(Z)` absolute. User param `met_logzsol` is solar-relative. Offset: `LOG10_ZSUN = -1.848`.

4. **Photometry precomputation:** Auto-activates when redshift fixed + filters present. 21.6× gradient speedup. Check `model._precomp is not None`.

5. **Corner plot axes:** `fig.axes` returns flat list. Reshape: `np.array(axes).reshape(n, n)`.

6. **Notebook editing:** Use Python JSON manipulation for `.ipynb`, not text editing.

7. **`_correlated_field` key:** When `StandardizedForwardModel` passes a pre-computed correlated field to `Model.predict_photometry()`, it uses the `_correlated_field` key in the params dict. The Model uses this directly instead of recomputing `√P · ξ`.

## File Locations

- **Code:** `~/Projects/diffsed/`
- **Paper draft:** `~/writing-workspace/projects/differentiable_psd_sed_fitting/`
- **SSP data:** `~/Projects/diffsed/data/` (64 MB HDF5, not in git)
- **Design spec:** `docs/specs/2026-03-15-standardized-model-redesign.md`
- **Analysis scripts:** `analysis/` (figure generation for paper)
- **Paper figures:** `analysis/figures/` and paper `figures/` directory

## Code Name

`diffsed` is a working name. Final public release name TBD.

## Paper Scope

- **Paper I (this paper):** Methods + mock recovery tests (individual + hierarchical PSD recovery)
- **Paper II (future):** Real data application (SDSS, JWST)
