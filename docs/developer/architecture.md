# Architecture

This page describes tengri's package structure, dependency layers, data flow,
and the Information Field Theory (IFT) framework that underpins the SFH model.

## Package structure

```
src/tengri/
├── __init__.py              # Public API re-exports
├── distributions.py         # Uniform, Gaussian, LogUniform, LogNormal, StudentT, Fixed
├── plotting.py              # Visualization utilities
├── simulate.py              # SED-from-SFH utilities
│
├── core/                    # Forward model
│   ├── model.py             # Model class (thin orchestrator)
│   ├── param_spec.py        # Parameters: parameter definitions + validation
│   ├── param_translate.py   # Public → internal param mapping + unit conversion
│   ├── emission_helpers.py   # Shared emission physics (nebular, shock, AGN, dust IR, radio, xray, IGM)
│   ├── fused_kernels.py     # JIT kernel factory — calls emission_helpers
│   ├── sed_pipeline.py      # Non-fused SED engine — calls emission_helpers
│   ├── prediction.py        # Lazy Prediction object
│   ├── noise.py             # Noise model handling
│   └── mock.py              # Mock galaxy generation
│
├── inference/               # All fitting + results
│   ├── fitter.py            # Fitter: MAP, Ray Tracing, NUTS, geoVI, MGVI
│   ├── hierarchical.py      # PopulationFitter: shared PSD
│   ├── standardized.py      # StandardizedForwardModel: xi → observables
│   ├── posterior.py          # Posterior: summary, corner, ESS
│   ├── raytrace.py          # Ray Tracing Sampler (Behroozi 2025)
│   ├── map_optimizer.py     # MAP optimization (adam/adamw/sgd/custom optax)
│   ├── nuts.py              # NUTS sampler (blackjax wrapper)
│   ├── geovi.py             # geoVI inference (nifty8.re wrapper)
│   ├── geovi_nuts.py        # geoVI warm-start → NUTS refinement
│   ├── vi_config.py         # VI settings and sample scheduling
│   └── common.py            # Shared inference utilities
│
├── models/                  # Physics modules
│   ├── sfh/                 # SFH models, PSD, GP generation
│   ├── dust/                # Two-component attenuation + IR emission
│   ├── agn/                 # AGN disc + torus models
│   ├── nebular/             # Nebular emission (BakedIn, CLOUDY, Cue)
│   ├── sps/                 # DSPS wrapper, SSP loading
│   ├── observation/         # Photometry, spectroscopy, filters
│   ├── igm.py               # IGM transmission (Inoue+2014)
│   ├── radio.py             # Radio continuum
│   └── xray.py              # X-ray emission
│
├── diagnostics/             # Analysis tools
│   ├── fisher.py            # Fisher information matrix
│   ├── green_functions.py   # Green's function analysis
│   └── saliency.py          # Gradient saliency maps
│
├── profiling/               # Performance measurement
│   ├── timers.py            # Timing utilities
│   ├── pipeline.py          # Component-level pipeline profiling
│   └── memory.py            # Memory usage tracking
│
└── utils/                   # Grid, cosmology, transforms
```

## Dependency layers

tengri uses a layered dependency model. Each layer adds optional packages on
top of the core JAX stack. Higher layers never import lower-layer-only code
at module level.

```
Layer 0 (always): JAX, DSPS, h5py, matplotlib, scipy
  → Forward model, mock generation, SFH prediction, plotting

Layer 1 (MAP): + optax
  → Gradient descent optimization

Layer 2 (Ray Tracing): no extra deps
  → inference/raytrace.py (Behroozi 2025, Apache 2.0)

Layer 3 (NUTS): + blackjax
  → Hamiltonian Monte Carlo

Layer 4 (geoVI/MGVI/hierarchical): + nifty8.re
  → Variational inference, CorrelatedFieldMaker, optimize_kl
```

```{important}
Never import `nifty8.re` at module level. Always use lazy imports inside the
methods that need it. The forward model, distributions, and non-VI samplers
must work without NIFTy installed.
```

## Class naming conventions

Each class suffix has a specific meaning — these are enforced across the codebase:

| Suffix | Meaning | Examples |
|--------|---------|---------|
| *(none)* | Data container / physics object | `Photometry`, `Spectroscopy`, `Observation` |
| `Model` | A physical forward model | `Model`, `NoiseModel` |
| `Parameters` | Prior specification — what gets fitted | `Parameters` |
| `Fitter` | Runs inference | `Fitter`, `PopulationFitter` |
| `Posterior` | Result of inference (samples + diagnostics) | `Posterior`, `PopulationPosterior` |
| `Config` | Static structural choice (not a fitted param) | `AGNConfig`, `VIConfig` |
| `Backend` | Interchangeable computation engine | `CueBackend`, `CloudyGridBackend` |

## Module dependency direction

Dependencies flow in one direction:

```
core → models → inference
```

Never reverse this. Modules in `core/` never import from `inference/`.
Modules in `models/` never import from `inference/`. The `inference/` layer
depends on both `core/` and `models/`.

## Layered architecture

```
┌──────────────────────────────────────────────────────┐
│  User API:  Parameters → Model → Fitter → Posterior   │
├──────────────────────────────────────────────────────┤
│  Forward Model: SFH → SPS → Dust → Photometry       │
├──────────────────────────────────────────────────────┤
│  Low-level JAX functions: PSD, GP, SFH, attenuation   │
├──────────────────────────────────────────────────────┤
│  JAX runtime: JIT, vmap, grad, autodiff              │
└──────────────────────────────────────────────────────┘
```

**Layer 1 -- Low-level functions.** Pure JAX functions with no state. Each
implements one physical component: PSD models, GP generation, mean SFH,
dust attenuation, SPS integrals, photometry. These are `@jax.jit`-decorated,
accept arrays and scalars, and compose via standard function calls.

**Layer 2 -- Forward model.** `Model` composes the low-level functions into
a single differentiable pipeline (see [Data flow](#data-flow) below).

**Layer 3 -- User API.** Four classes provide the high-level interface:
`Parameters`, `Model`, `Fitter`, and `Posterior`. Each has a `.summary()`
method for quick inspection.

## Core principle: full standardization

Every parameter -- physical, PSD, and latent field -- is represented by a
standardized latent variable `xi ~ N(0, I)`. The prior is absorbed into the
forward model via differentiable transforms. The loss is always:

```
H(xi) = 1/2 chi^2(data, f(xi)) + 1/2 xi^T xi
```

No separate prior penalty terms. No per-distribution special cases.

## Emission physics architecture

Both the non-fused pipeline (`sed_pipeline.py`) and the fused JIT kernel
(`fused_kernels.py`) call the same pure functions from `emission_helpers.py`.
This guarantees identical physics across code paths:

```
emission_helpers.py          ← single source of truth
    nebular_emission()         wNE SSP detection + SFR-based Q_H fallback
    attenuate_emission()       dust on nebular/shock; returns L_absorbed
    shock_emission()           MAPPINGS V line emission
    agn_emission()             K&D 3-zone + SKIRTOR + polar dust
    dust_ir_emission()         DL07/DL14/Dale/MBB (energy-balanced)
    radio_emission()           SF synchrotron + AGN jets + free-free
    xray_emission()            XRBs + AGN corona
    igm_absorption()           Inoue+2014

sed_pipeline.py              ← non-fused orchestrator
    branching + component tracking + return dict

fused_kernels.py             ← fused JIT orchestrator
    closure captures at build time, calls same helpers inside @jax.jit
```

**Adding a new parameter:** update the helper function signature in
`emission_helpers.py` → both paths inherit the change automatically.

**Energy conservation:** `L_absorbed = L_absorbed_stellar + L_absorbed_nebular`.
The `attenuate_emission()` helper returns the absorbed luminosity as a side
output, which feeds into the dust IR energy balance.

**Nebular dust attenuation** (configurable via `Parameters(neb_dust=...)`):

| Mode | Birth-cloud | Diffuse ISM | Use case |
|------|:-----------:|:-----------:|----------|
| `"bc"` (default) | Yes | Yes | Charlot & Fall (2000) |
| `"diff"` | No | Yes | CLOUDY grids with internal HII dust |
| `"neb"` | Custom law | Yes | Different grain properties in HII regions |
| `"none"` | No | No | Debugging |

## Data flow

The full parameter-to-observable pipeline:

```
User defines Parameters:
  sfh_alpha = Uniform(0.5, 3.0)         # Distribution object
  met_logzsol = Gaussian(-0.5, 0.3)
  psd_sigma = LogNormal(0.0, 0.8)       # free or fixed
  dust_tau_diff = 0.3                    # Fixed (scalar shorthand)

                    |
                    v

StandardizedForwardModel builds xi -> theta mapping:
  xi_alpha  -> sigmoid(xi) * (3.0 - 0.5) + 0.5    [Uniform.unstandardize]
  xi_met    -> clip(-0.5 + 0.3 * xi, -2, 0)        [Gaussian.unstandardize]
  xi_sigma  -> exp(0.0 + 0.8 * xi)                 [LogNormal.unstandardize]
  xi_field  -> IFFT(sqrt(P(sigma, tau)) * xi_field) [correlated field]

                    |
                    v

param_translate.py maps public -> internal names:
  met_logzsol           -> log_z_abs      (+ LOG10_ZSUN offset)
  dust_tau_bc           -> tau_bc
  dust_tau_diff         -> tau_diff
  sfh_field_psd_sigma   -> psd_sigma
  sfh_field_psd_tau_myr -> psd_tau_yr     (x 1e6 to convert Myr -> yr)

                    |
                    v

Model.predict_photometry(params) -> flux array

                    |
                    v

Loss: H = 1/2 sum((data - flux) / noise)^2 + 1/2 xi^T xi
```

In more detail, the forward model evaluates:

```
xi, theta_PSD -> sqrt(P) * FFT(xi) -> x(t)                    [GP realization]
             -> mean_SFH(t) * exp(x(t) - sigma^2/2)           [lognormal SFH]
             -> interpolate to SSP ages -> CSP weights          [mass history]
             -> sum(weights * SSP_flux * dust_atten) * LSUN_ERG  [composite SED, erg/s/Hz]
             -> + AGN / radio / X-ray / nebular components       [all erg/s/Hz]
             -> filter convolution * (1+z) / (4 pi d_L^2)       [observed flux, erg/s/cm²/Hz]
```

The lognormal correction `exp(-sigma^2/2)` ensures that `E[SFR] = mean_SFR`
regardless of the GP amplitude, so the parametric envelope directly controls
the average star formation history.

## The IFT framework

Information Field Theory (Ensslin 2019) treats the SFH as a continuous field
`x(t)` to be inferred from finite data.

### Correlated field model

The log-SFR fluctuation `x(t)` is a Gaussian process defined via:

```
x = IFFT(sqrt(P) * xi),    xi ~ N(0, I)
```

where `P(omega)` is the power spectral density and `xi` is a vector of
standardized latent variables. This separates the prior structure (encoded in
`P`) from the stochastic realization (encoded in `xi`).

### PSD as prior

The DRW (damped random walk) PSD has two parameters:

- **sigma_burst** (`psd_sigma`): amplitude of SFR fluctuations in dex
- **tau_burst** (`psd_tau_myr`): memory timescale -- how long a burst/quench
  episode lasts

Different physical mechanisms produce different (sigma, tau) combinations:

| Timescale | Physical mechanism |
|-----------|-------------------|
| 1--10 Myr | Stellar winds, supernovae |
| 20--50 Myr | Supernova feedback cycles |
| 100--300 Myr | Gas accretion, halo response |

### Standardized latent space

All samplers operate in xi-space, which is simple (standard normal). The
physics is encoded in the amplitude operator `sqrt(P)`. This makes
gradient-based inference efficient: the sampler explores a well-conditioned
space while the forward model maps to complex SFH shapes.

### Information Hamiltonian

The loss function (negative log-posterior) is:

```
H(xi|d) = 1/2 sum_k ((d_k - m_k(theta)) / sigma_k)^2  +  1/2 xi^T xi
           |----------- data fit (chi^2) ------------|     |-- GP prior --|
```

Minimizing H gives MAP; sampling from `exp(-H)` gives the full posterior.

### Swappable PSD models

The `StandardizedForwardModel` accepts a `psd_model` callable:

```python
def my_psd(sigma, tau_yr, n_grid, log_ages):
    # Extended Regulator, Flex-PSD, Matern, etc.
    return sqrt_power_array

smodel = StandardizedForwardModel(model, psd_model=my_psd)
```

When the PSD is fixed, `sqrt(P)` is precomputed once. When PSD parameters
are free, `sqrt(P)` depends on `xi_sigma` and `xi_tau` through their
unstandardize transforms, creating a natural PSD-field coupling.

## Design principles

1. **Pure JAX functions.** All model components are stateless pure functions.
   No global state, no side effects.
2. **JIT-compatible.** Everything inside the forward model compiles to a
   single XLA graph via `@jax.jit`.
3. **Immutable arrays.** Never mutate arrays. Use `jnp.ndarray.at[].set()`
   for updates.
4. **End-to-end differentiability.** Every computation from PSD through SPS
   to filter convolution is differentiable via `jax.grad`.
5. **Lazy imports.** Optional dependencies (nifty, blackjax, optax, arviz)
   are imported inside the methods that use them, never at module level.

## Stochastic vs parametric mode

tengri supports two SFH modes controlled by `Parameters`:

**Parametric** (`stochastic=False`): The SFH is a smooth double power law
with 4 parameters. Low-dimensional (7--11 free parameters), suitable for
broadband photometry fitting at catalog scale.

**Stochastic** (`stochastic=True`): Adds a GP-correlated burstiness field
on top of the parametric envelope. Higher-dimensional (`n_grid` + physical
parameters), but captures realistic SFH variability. Best used with geoVI
or Ray Tracing for efficient posterior exploration.

## Distribution protocol

To add a new prior, implement a `Distribution` subclass:

```python
class MyPrior(Distribution):
    @property
    def bounds(self) -> tuple[float, float]: ...
    def sample(self, key) -> jnp.ndarray: ...          # for mock generation
    def log_prob(self, x) -> jnp.ndarray: ...          # for diagnostics
    def unstandardize(self, xi) -> jnp.ndarray: ...    # xi~N(0,1) -> theta
    def standardize(self, theta) -> jnp.ndarray: ...   # theta -> xi
```

```{important}
`unstandardize()` must be differentiable via `jax.grad`. Test with:

    g = jax.grad(lambda xi: dist.unstandardize(xi))(jnp.array(0.5))
    assert jnp.isfinite(g)
```

## NIFTy.re integration

When geoVI/MGVI is used, tengri wraps NIFTy's optimized implementations:

| tengri | NIFTy equivalent | Usage |
|--------|-----------------|-------|
| `Distribution.unstandardize()` | `jft.NormalPrior`, etc. | Same concept -- both map xi to theta |
| `StandardizedForwardModel` | `jft.Model` | Wrapped as `jft.Model` for `optimize_kl` |
| Chi-squared likelihood | `jft.Gaussian` | Used directly |
| `optimize_kl` | `jft.optimize_kl` | Used directly |
| Correlated field | `jft.CorrelatedFieldMaker` | Used for joint PSD learning |
