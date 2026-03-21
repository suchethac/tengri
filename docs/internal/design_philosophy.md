# tengri: Design Philosophy and Architecture

> **Code name:** `tengri` is a working name. The final public release name is TBD.

## Overview

tengri is a fully differentiable galaxy SED fitting code built on JAX. It models star formation histories (SFHs) as continuous correlated random fields using Information Field Theory (IFT), enabling gradient-based inference that is 10-100x faster than traditional gradient-free samplers.

The durable contribution is not any particular SFH parametrization, dust model, or inference algorithm — those are modular and configurable. It is the combination of:
1. A **correlated-field SFH prior** (PSD → GP → log-SFR fluctuations) that encodes physically motivated temporal correlations
2. A **fully differentiable forward model** from latent variables through SPS to observed photometry/spectra
3. A **multi-method inference pipeline** where Ray Tracing and geoVI serve as equal-priority primary samplers, NUTS validates, and MAP initializes

## Core Principle: End-to-End Differentiability

Every computation in tengri — from the power spectral density (PSD) of the SFH through stellar population synthesis, dust attenuation, and filter convolution — is implemented as a composition of differentiable JAX functions. This enables:

1. **Gradient-based optimization** (MAP via Adam) in seconds per galaxy
2. **Ray tracing MCMC** (Behroozi 2025) for gradient-directed posterior exploration
3. **Hamiltonian Monte Carlo** (NUTS via BlackJAX) for exact posteriors in minutes
4. **Variational inference** (geoVI via NIFTy.re) for approximate posteriors at scale
5. **Fisher information analysis** via autodiff Jacobians
6. **Gradient saliency maps** showing which spectral features constrain which parameters

## Architecture: Layered Design

```
┌──────────────────────────────────────────────────────┐
│  User API:  ParamSpec → Model → Fitter → Posterior   │
├──────────────────────────────────────────────────────┤
│  Forward Model: SFH → SPS → Dust → Photometry       │
├──────────────────────────────────────────────────────┤
│  Low-level JAX functions: PSD, GP, DPL, Charlot+Fall │
├──────────────────────────────────────────────────────┤
│  JAX runtime: JIT, vmap, grad, autodiff              │
└──────────────────────────────────────────────────────┘
```

### Layer 1: Low-Level Functions

Pure JAX functions with no state. Each implements one physical component:

- **PSD models** (`psd_drw`, `psd_matern`): Power spectral densities for SFH burstiness
- **GP generation** (`gp_from_xi`): Deterministic GP from standardized latent vector ξ
- **Mean SFH** (`double_powerlaw`): Smooth parametric envelope
- **Dust** (`charlot_fall`): Wavelength-dependent attenuation
- **SPS** (`compute_csp_sed`): Composite stellar population via DSPS/FSPS templates
- **Photometry** (`compute_flux_density`): Filter convolution with redshift

These are `@jax.jit`-decorated, accept arrays and scalars, and compose via standard function calls.

### Layer 2: Forward Model

`ForwardModel` composes the low-level functions into a single differentiable pipeline:

```
ξ, θ_PSD → sqrt(P) · FFT(ξ) → x(t)                    [GP realization]
       → mean_SFH(t) · exp(x(t) - σ²/2)               [lognormal SFH]
       → interpolate to SSP ages → CSP weights          [mass history]
       → Σ weights · SSP_flux · dust_atten              [composite SED]
       → filter convolution × (1+z)/(4π d_L²)          [observed flux]
```

The lognormal correction `exp(-σ²/2)` ensures that `E[SFR] = mean_SFR` regardless of the GP amplitude, so the parametric envelope directly controls the average star formation history.

### Layer 3: User API

The high-level API provides four classes:

**`ParamSpec`** — Defines all model parameters in one place:
```python
spec = ParamSpec(
    sfh_alpha        = Uniform(0.5, 3.0),      # free, uniform prior
    sfh_tau_peak_gyr = Gaussian(4.0, 1.0, lo=0.5, hi=10.0),  # Gaussian prior
    dust_slope       = -0.7,                    # fixed
    redshift         = 0.1,                     # fixed (or free if given a distribution)
    stochastic       = True,                    # enable GP burstiness
    n_grid           = 128,                     # GP grid size
)
```

Each parameter is either `Fixed(value)`, `Uniform(lo, hi)`, `Gaussian(mu, sigma)`, or `LogUniform(lo, hi)`. This single object controls mock generation (sampling), inference (priors), and parameter fixing.

**`Model`** — The forward model with clean parameter names:
```python
model = Model(spec, ssp_data, filters=filters)
photometry = model.predict_photometry(params)
sfh = model.predict_sfh(params)
derived = model.predict_derived(params)  # stellar mass, SFR, sSFR
```

The Model translates between user-facing names (`sfh_tau_peak_gyr` in Gyr) and internal names (`tau_sfh` in yr) automatically. It supports both parametric-only SFHs (no GP) and stochastic SFHs (with GP).

**`Fitter`** — Separates inference from the model:
```python
fitter = Fitter(model, data, noise, data_type="photometry")
result_map = fitter.run("map", n_steps=1500)
result_nuts = fitter.run("nuts", init_from=result_map, n_warmup=500)
```

The Fitter builds the loss function from the Model's predictions and the ParamSpec's priors. It handles the unbounded↔physical parameter transformation, supports MAP→NUTS chaining, and warns about dimensionality when using NUTS with stochastic SFHs.

**`Posterior`** — Results with resampling and diagnostics:
```python
posterior.summary()             # median ± 68% CI
posterior.derived["stellar_mass"]  # derived quantities per sample
posterior.resample(key, n=50)   # draw from posterior
posterior.to_param_spec()       # convert to ParamSpec for mock generation
posterior.to_arviz()            # ArviZ InferenceData for diagnostics
```

## The IFT Framework for SFHs

Information Field Theory (Enßlin 2019) treats the SFH as a continuous field `x(t)` to be inferred from finite data. The key components:

### Correlated Field Model

The log-SFR fluctuation `x(t)` is a Gaussian process defined via:

```
x = IFFT(√P · ξ),    ξ ~ N(0, I)
```

where `P(ω)` is the power spectral density and `ξ` is a vector of standardized latent variables. This separates the prior structure (encoded in `P`) from the stochastic realization (encoded in `ξ`).

### PSD as Prior

The DRW (damped random walk) PSD has two parameters:

- **σ_burst** (`psd_sigma`): amplitude of SFR fluctuations in dex
- **τ_burst** (`psd_tau_myr`): memory timescale — how long a burst/quench episode lasts

Different physical mechanisms produce different (σ, τ) combinations:
- τ ~ 1-10 Myr: stellar winds, supernovae
- τ ~ 20-50 Myr: supernova feedback cycles
- τ ~ 100-300 Myr: gas accretion, halo response

### Standardized Latent Space

All samplers operate in ξ-space, which is simple (standard normal). The physics is encoded in the amplitude operator √P. This makes gradient-based inference efficient: the sampler explores a well-conditioned space while the forward model maps to complex SFH shapes.

### Information Hamiltonian

The loss function (negative log-posterior) is:

```
H(ξ|d) = ½ Σ_k ((d_k - m_k(θ))/σ_k)² + ½ ξᵀξ + prior_penalties(θ)
         ├── data fit (χ²) ──┤   ├ GP prior ┤  ├── param priors ──┤
```

Minimizing H gives MAP; sampling from exp(-H) gives the full posterior.

## Ray Tracing Sampler

In addition to MAP, NUTS, and geoVI, tengri integrates the Ray Tracing Sampler of Behroozi (2025, arXiv:2510.25824). This physics-inspired MCMC method propagates "rays" through parameter space using an analogy to Snell's law of refraction:

- **Refractive index:** `n(x) = L(x)^{1/(D-1)}`, where `L(x)` is the likelihood and `D` is the number of parameters. Rays bend toward higher-likelihood regions just as light bends toward denser media.
- **Resilience to stochastic gradients:** Unlike HMC/NUTS, which rely on energy conservation and can fail when gradients are noisy, ray tracing uses only gradient *direction* (not magnitude) to compute refraction angles. This makes it robust to noisy or approximate likelihoods.
- **Barrier crossing:** Rays can traverse low-likelihood valleys between modes because refraction (unlike Hamiltonian dynamics) does not conserve an energy that would trap the sampler in a single basin.
- **Integration into Fitter:** The sampler is available as `fitter.run("raytrace", init_from=result_map, n_steps=500)`. It accepts MAP results as initialization and returns a `Posterior` object compatible with all downstream analysis (summary, corner plots, ArviZ export).

The Ray Tracing Sampler and geoVI are **equal-priority** primary inference methods. Ray Tracing provides exact, asymptotically unbiased posteriors and handles stochastic gradients well; geoVI provides fast approximate posteriors and scales to hierarchical problems with >10^5 parameters. NUTS serves as the gold-standard validation tool for low-dimensional problems.

## Hierarchical PSD Inference

The defining science application is **population-level PSD parameter recovery**: sharing (σ_PSD, τ_PSD) across N galaxies while each galaxy retains its own latent field ξ_i and physical parameters. The total dimensionality is 2 + N × (n_grid + n_phys).

**Current implementation:** `HierarchicalFitter` flattens all parameters and runs RT or geoVI on the joint vector, with per-galaxy MAP initialization.

**Planned improvement:** Use NIFTy's `CorrelatedFieldMaker` to learn PSD hyperparameters jointly inside the generative model, as demonstrated in production IFT applications (Eberle+2025, Roth+2024, Terveer+2026). The PSD shape should be part of the generative model with its own learned hyperparameters, not external flat parameters. Key practices from the literature:
- 8 samples (4 antithetic pairs) per KL iteration, not 80
- Convergence criterion: mean squared weighted deviation < 1.05 for 3 consecutive iterations
- Major/minor cycle scheme for approximate + exact likelihood correction

## Stochastic vs Parametric Mode

tengri supports two SFH modes:

**Parametric** (`stochastic=False`): The SFH is a smooth double power law with 4 parameters (α, β, τ_peak, SFR_peak). Fast, low-dimensional (7-11 free parameters), suitable for broadband photometry fitting at catalog scale.

**Stochastic** (`stochastic=True`): The SFH adds a GP-correlated burstiness field on top of the parametric envelope. Higher-dimensional (N_grid + 10 physical parameters), but captures realistic SFH variability. Best used with geoVI for efficient approximate posteriors.

## Precomputation

For photometric fitting at fixed redshift, the SSP fluxes integrated through each filter can be precomputed once, eliminating the wavelength-level integral from the MCMC inner loop. This provides a 30-50x speedup (following Zacharegkas+2025). tengri handles this automatically based on the ParamSpec:

- Fixed redshift → precompute at that z
- Free redshift → precompute on a grid, interpolate during inference

## Parallel Mock Generation

Mock galaxy catalogs are generated by sampling from the ParamSpec and evaluating the forward model:

```python
param_batch = spec.sample_batch(key, n=10000)
mock_catalog = model.mock_batch(param_batch, snr=20.0, key=noise_key)
```

The batch generation uses JAX's vmap for efficient GPU parallelization.

## Parameter Names and Units

All parameters use descriptive, prefixed names with explicit units:

| Parameter | Description | Units |
|-----------|-------------|-------|
| `sfh_alpha` | DPL falling slope (cosmic time) | dimensionless |
| `sfh_beta` | DPL rising slope (cosmic time) | dimensionless |
| `sfh_tau_peak_gyr` | DPL turnover time | Gyr |
| `sfh_peak_sfr` | Peak SFR | M☉/yr |
| `psd_sigma` | GP PSD amplitude | dex |
| `psd_tau_myr` | GP damping timescale | Myr |
| `met_logzsol` | Metallicity | log₁₀(Z/Z☉) |
| `dust_tau_bc` | Birth cloud optical depth | dimensionless |
| `dust_tau_diff` | Diffuse ISM optical depth | dimensionless |
| `dust_slope` | Dust power-law index | dimensionless |
| `redshift` | Source redshift | dimensionless |

## Dependencies

**Core:** JAX, DSPS, h5py, matplotlib
**Optional:** BlackJAX (NUTS), NIFTy.re (geoVI), optax (MAP), ArviZ (diagnostics)
