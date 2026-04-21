# Architecture

This page describes tengri's package structure, dependency layers, data flow,
and the Information Field Theory (IFT) framework that underpins the SFH model.

## Package structure

```
src/tengri/
├── __init__.py              # Public API re-exports
├── defaults.toml            # Default configuration values
│
├── parameters/              # Parameter definitions, priors, translation, defaults
│   ├── _param_defs.py       # Parameter name registry and default values
│   ├── _prior_defs.py       # Prior specifications per domain
│   ├── _param_translate.py  # Public → internal param name and unit conversion
│   └── parameters.py        # Parameters class: spec + validation
│
├── components/              # Physics modules — pure JAX functions
│   ├── sfh/                 # SFH models, PSD, GP generation
│   ├── sps/                 # DSPS wrapper, SSP loading
│   ├── dust/                # Two-component attenuation + IR emission
│   ├── nebular/             # Nebular emission (BakedIn, CLOUDY, Cue)
│   ├── agn/                 # AGN disc + torus (K&D, SKIRTOR, polar dust)
│   ├── igm/                 # IGM transmission (Inoue+2014)
│   ├── radio/               # Radio continuum (SF + AGN)
│   └── xray/                # X-ray emission (XRBs + AGN)
│
├── forward/                 # Forward SED model and pipeline
│   ├── sed_model.py         # SEDModel orchestrator
│   ├── pipeline.py          # Non-fused SED engine + component tracking
│   ├── emission_helpers.py  # Shared emission functions
│   ├── kernels/             # Fused JIT kernels
│   └── precompute/          # Precomputation registry
│
├── observation/             # Data handling
│   ├── filters.py, photometry.py, spectroscopy.py  # Data containers
│   ├── noise_model.py       # Noise handling
│   └── mock.py              # Mock generation
│
├── inference/               # Parameter estimation
│   ├── fitter.py            # Fitter orchestrator
│   ├── posterior.py         # Posterior samples + diagnostics
│   ├── loss.py              # Loss function (chi^2 + prior)
│   └── backends/            # Inference engines
│       ├── map_dispatch.py  # MAP, Laplace, Pathfinder
│       ├── vi/              # Variational inference
│       ├── mcmc/            # MCMC (NUTS, Ray Tracing, ESS)
│       └── nested/          # Nested sampling
│
├── analysis/                # Post-fitting analysis and visualization
│   ├── diagnostics/         # Posterior diagnostics (ESS, R-hat, etc.)
│   ├── plotting/            # Visualization (corner, SED, SFH, etc.)
│   └── simulate.py          # SED generation from SFH
│
├── config/                  # Configuration and exceptions
│   ├── settings.py          # Model configuration (DustConfig, NebularConfig, etc.)
│   ├── exceptions.py        # Exception hierarchy (TengriError, etc.)
│   ├── display.py           # Display formatting utilities
│   └── deprecation.py       # Deprecation warnings
│
├── profiling/               # Performance measurement
│   ├── timers.py            # Timing utilities
│   ├── pipeline.py          # Component-level pipeline profiling
│   └── memory.py            # Memory usage tracking
│
└── utils/                   # Shared utilities (no dependencies on physics)
    ├── cosmology.py         # Cosmological calculations
    ├── conversions.py       # Unit conversions
    ├── interpolation.py     # Interpolation utilities
    ├── physics_constants.py # Physical constants (IAU 2015)
    └── wavelength.py        # Wavelength utilities
```

## Class naming conventions

Each class suffix has a specific meaning — these are enforced across the codebase:

| Suffix | Meaning | Examples |
|--------|---------|---------|
| *(none)* | Data container / physics object | `Photometry`, `Spectroscopy`, `Observation` |
| `Model` | A physical forward model | `SEDModel`, `NoiseModel` |
| `Parameters` | Prior specification — what gets fitted | `Parameters` |
| `Fitter` | Runs inference | `Fitter`, `PopulationFitter` |
| `Posterior` | Result of inference (samples + diagnostics) | `Posterior`, `PopulationPosterior` |
| `Config` | Static structural choice (not a fitted param) | `DustConfig`, `NebularConfig`, `VIConfig` |
| `Backend` | Interchangeable computation engine | `CueBackend`, `CloudyGridBackend` |

## Optional dependencies (layer stacking)

tengri layers optional packages on top of core JAX:

```
Layer 0 (always): JAX, DSPS, h5py, scipy
  → Forward model, SFH, plotting

Layer 1 (MAP): + optax
  → Gradient descent optimization

Layer 2 (Ray Tracing): no extra deps
  → inference/backends/mcmc/raytrace.py

Layer 3 (NUTS): + blackjax
  → inference/backends/mcmc/nuts.py

Layer 4 (VI): + nifty8.re
  → Variational inference, optimize_kl
```

```{important}
Never import `nifty8.re` at module level; use lazy imports inside methods
that need it. The forward model and non-VI samplers must work without NIFTy.
```

Module dependencies flow downward:

```
parameters/ → components/ → forward/ → observation/ → inference/ → analysis/
config/ and utils/ (available at all layers)
```

**Critical rule:** Never reverse. `components/` never imports from `forward/`
or `inference/`. `forward/` never imports from `inference/`.

## Layered computation architecture

```
┌──────────────────────────────────────────────────────┐
│  User API:  Parameters → SEDModel → Fitter → Posterior │
├──────────────────────────────────────────────────────┤
│  Forward SED: SFH → SPS → Dust → AGN → IGM → Photo  │
├──────────────────────────────────────────────────────┤
│  Low-level JAX: PSD, GP, SFH, attenuation, filters   │
├──────────────────────────────────────────────────────┤
│  JAX runtime: JIT, vmap, grad, autodiff              │
└──────────────────────────────────────────────────────┘
```

**Layer 1 -- Low-level functions.** Pure JAX functions in `components/`. Each
implements one physical component: PSD, GP generation, SFH models, dust
attenuation, SPS integrals. These are `@jax.jit`-decorated, accept arrays and
scalars, compose via standard function calls.

**Layer 2 -- Forward model.** `SEDModel` in `forward/` orchestrates the
low-level functions into a single differentiable pipeline (see [Data flow](#data-flow)).

**Layer 3 -- User API.** Four classes provide the high-level interface:
`Parameters`, `SEDModel`, `Fitter`, and `Posterior`. Each has a `.summary()`
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

Both the non-fused pipeline (`pipeline.py`) and fused JIT kernels
(`kernels/`) call the same pure functions from `emission_helpers.py`.
This guarantees identical physics across code paths:

```
emission_helpers.py          ← single source of truth
    compute_nebular_sed()      wNE SSP detection + SFR-based Q_H fallback
    apply_dust_attenuation()   dust on nebular/shock; returns L_absorbed
    compute_shock_sed()        MAPPINGS V line emission
    compute_agn_sed()          K&D 3-zone + SKIRTOR + polar dust
    compute_dust_ir_sed()      DL07/DL14/Dale/MBB (energy-balanced)
    compute_radio_sed()        SF synchrotron + AGN jets + free-free
    compute_xray_sed()         XRBs + AGN corona
    apply_igm_transmission()   Inoue+2014

pipeline.py                  ← non-fused orchestrator
    branching + component tracking + return dict

kernels/*.py                 ← fused JIT orchestrators
    closure captures at build time, calls same helpers inside @jax.jit
```

**Adding a new parameter:** update the helper function signature in
`emission_helpers.py` → both paths inherit the change automatically.

**Energy conservation:** `L_absorbed = L_absorbed_stellar + L_absorbed_nebular`.
The `apply_dust_attenuation()` helper returns the absorbed luminosity as a side
output, which feeds into the dust IR energy balance.

## Data flow

The full parameter-to-observable pipeline:

```
User defines Parameters:
  sfh_alpha = Uniform(0.5, 3.0)         # Distribution object
  met_logzsol = Gaussian(-0.5, 0.3)
  sfh_field_psd_sigma = LogNormal(0.0, 0.8)  # free or fixed
  dust_tau_diff = 0.3                    # Fixed (scalar shorthand)

                    |
                    v

Parameters.unstandardize() builds xi -> theta mapping:
  xi_alpha  -> sigmoid(xi) * (3.0 - 0.5) + 0.5    [Uniform.unstandardize]
  xi_met    -> clip(-0.5 + 0.3 * xi, -2, 0)        [Gaussian.unstandardize]
  xi_sigma  -> exp(0.0 + 0.8 * xi)                 [LogNormal.unstandardize]
  xi_field  -> IFFT(sqrt(P(sigma, tau)) * xi_field) [correlated field]

                    |
                    v

parameters/_param_translate.py maps public -> internal names:
  met_logzsol           -> log_z_abs      (+ LOG10_ZSUN offset)
  dust_tau_bc           -> tau_bc
  dust_tau_diff         -> tau_diff
  sfh_field_psd_sigma   -> psd_sigma
  sfh_field_psd_tau_myr -> psd_tau_yr     (x 1e6 to convert Myr -> yr)

                    |
                    v

SEDModel.predict_photometry(params) -> flux array

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

Information Field Theory treats the SFH as a continuous field `x(t)`.

**Correlated field:** Log-SFR fluctuations follow a GP via `x = IFFT(sqrt(P) * xi)`
where `P` is the power spectral density (PSD) and `xi ~ N(0, I)`.

**PSD parameters:**
- `psd_sigma`: amplitude of SFR fluctuations (dex)
- `psd_tau_myr`: burst/quench memory timescale (Myr)

**Loss function (information Hamiltonian):**
```
H(xi|d) = 1/2 sum_k ((d_k - m_k(xi)) / sigma_k)^2 + 1/2 xi^T xi
```

MAP: minimize H. Posterior samples: sample from exp(-H).

### Swappable PSD models

The forward model accepts a `psd_model` callable:

```python
def my_psd(sigma, tau_yr, n_grid, log_ages):
    # Extended Regulator, Flex-PSD, Matern, etc.
    return sqrt_power_array

model = SEDModel(parameters, psd_model=my_psd)
```

When the PSD is fixed, `sqrt(P)` is precomputed once. When PSD parameters
are free, `sqrt(P)` depends on `xi_sigma` and `xi_tau` through their
unstandardize transforms, creating a natural PSD-field coupling.

## Design principles

1. **Pure JAX.** All components are stateless pure functions (no side effects).
2. **JIT-compilable.** Forward model compiles to single XLA graph.
3. **Immutable arrays.** Use `jnp.ndarray.at[].set()`, never in-place mutation.
4. **End-to-end differentiable.** Every computation supports `jax.grad`.
5. **Lazy imports.** Optional deps (nifty, blackjax, optax) imported inside methods only.

## Modes

**Parametric** (`stochastic=False`): Smooth double power law (4 params, 7–11 DOF).
For broadband photometry at catalog scale.

**Stochastic** (`stochastic=True`): GP-correlated burstiness on parametric
envelope. Higher-dimensional, captures realistic SFH variability. Best with `vi`
or `mcmc_raytrace`.

## Key reading

- `docs/dev/NAMING_CONTRACT.md` — Class and function naming conventions
- `docs/dev/design_philosophy.md` — Architecture philosophy (immutability, JIT, pure functions)
