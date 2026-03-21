# diffsed

**Differentiable SED fitting with Information Field Theory star formation history priors.**

A modular, fully differentiable JAX code for galaxy spectral energy distribution (SED) fitting. The star formation history is modeled as a continuous correlated field governed by a power spectral density (PSD) kernel, enabling physically motivated burstiness priors. End-to-end gradients flow through the entire pipeline -- from PSD parameters through stellar population synthesis to predicted photometry and spectroscopy -- enabling fast variational and gradient-based inference.

> **Status:** v0.1.0, active development. Core pipeline fully functional with 808 tests. Paper in preparation.

## Highlights

- **IFT-based SFH model**: Star formation history as a continuous 1D field reconstructed from noisy SED data via Information Field Theory (Ensslin 2019). The PSD encodes the amplitude and timescale of burstiness.
- **Fully differentiable**: Pure JAX from PSD parameters through to predicted photometry. Gradients via autodiff enable HMC, variational inference, and gradient-based optimization.
- **GPU-native**: All operations are JIT-compiled and run on GPU/TPU. Designed for catalog-scale inference following the approach of [Zacharegkas, Hearin & Benson (2025)](https://arxiv.org/abs/2506.19919).
- **Modular forward model**: Every component (SFH, dust, SPS, AGN, nebular, observation) is a swappable pure function. The forward model is the primary product; inference is one application of it.
- **Multiple inference backends**: MAP, Ray Tracing (Behroozi 2025), NUTS ([BlackJAX](https://github.com/blackjax-devs/blackjax)), geoVI/MGVI ([NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty)), and hierarchical population fitting.
- **DSPS-powered SPS**: Differentiable stellar population synthesis via [DSPS](https://github.com/ArgonneCPAC/dsps) (Hearin et al. 2023). Accepts any SSP template in HDF5 format.

## Installation

```bash
# Basic install (JAX + DSPS)
pip install -e .

# With all inference backends
pip install -e ".[all]"

# Development (adds pytest, ruff, jupytext)
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, JAX >= 0.4.20, DSPS >= 0.3

## Quick start

```python
from diffsed import Model, ParamSpec, Fitter, Posterior, Uniform, Gaussian
from diffsed import load_ssp_data, load_filter_set

# Load SSP templates and filters
ssp = load_ssp_data("data/ssp.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# Define the model: parametric SFH + stochastic GP field
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0, 4),
    redshift=0.1,
)

# Build the model (auto-precomputes photometry kernels at fixed redshift)
model = Model(spec, ssp, filters=filters)

# Fit observed data
fitter = Fitter(model, data, noise)
result = fitter.run("geovi")        # or "raytrace", "nuts", "map"
print(result.summary_table())

# Posterior analysis
posterior = Posterior(result, model)
posterior.corner()
posterior.plot_sfh()
```

## The model

The star formation rate is decomposed into a smooth secular component and stochastic fluctuations:

```
ln SFR(t) = ln SFR_mean(t) + x(t) - K(0)/2
```

where:
- **SFR_mean(t)** is a parametric mean SFH (truncated skew-normal by default, 8+ options)
- **x(t) ~ GP(0, K)** is a Gaussian Process with covariance determined by the PSD
- **-K(0)/2** is the lognormal correction preserving the linear-SFR mean

The GP is generated via the IFT correlated field model:

```
x = IFFT(sqrt(P) * xi),    xi ~ N(0, I)
```

where P(omega) is the power spectral density. The default is a damped random walk (DRW):

```
P(omega) = sigma_PS^2 * tau_PS / (1 + (tau_PS * omega)^2)
```

The two PSD parameters have direct physical meaning:
- **sigma_PS**: amplitude of SFR fluctuations (higher = burstier)
- **tau_PS**: characteristic damping timescale (sets the coherence time of bursts)

## SFH model types

Parametric mean SFH functions can be selected via prefix in `ParamSpec`. These can be composed with a GP field (e.g., `"tsnorm+field"`) or stacked (e.g., `"tsnorm+burst+field"`).

| Model | Prefix | Description |
|-------|--------|-------------|
| Truncated skew-normal | `tsnorm` | Default. Flexible peak location, skewness, width |
| Double power law | `dpl` | Rising + falling power laws with smooth turnover |
| Skew-normal | `snorm` | Gaussian-like with skewness |
| Normal | `norm` | Symmetric Gaussian |
| Log-normal | `lnorm` | Log-normal peak |
| Constant | `const` | Flat SFH |
| Exponential | `exp` | Exponentially declining |
| Delayed exponential | `dexp` | t * exp(-t/tau) |
| Burst | `burst` | Triweight kernel for recent bursts |

## Physics modules

**Stellar population synthesis**
- DSPS differentiable CSP integral with metallicity interpolation
- Any SSP template set in HDF5 format (BC03, BPASS, FSPS, ProGeny)
- [Pre-formatted SSP templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) publicly available
- IMF-based mass remaining fractions

**Dust attenuation**
- Two-component Charlot & Fall (2000) model (birth cloud + diffuse ISM)
- Pluggable attenuation laws: power_law, calzetti, kriek_conroy, smc, cardelli, salim

**Dust emission**
- Modified blackbody (MBB)
- Dale et al. (2014) templates
- Draine & Li (2007) tabulated templates

**AGN**
- 6 models: simple power law, standard thin disc, kubota_done, QSOGen (Temple+2021), SKIRTOR (Stalevski+2016) tabulated torus templates, unified NLR+BLR
- Broad and narrow line region emission

**Nebular emission**
- 3 backends: baked-in SSP emission lines, CLOUDY photoionization grid, Cue neural emulator (Li+2024, JAX re-implementation)

**Other**
- IGM absorption (Inoue+2014)
- Radio continuum (q_IR parametrization)
- X-ray emission

**Observation**
- Broadband photometry with filter convolution
- Pixel-level spectroscopy with Chebyshev calibration polynomials
- Emission line marginalization

## Inference methods

| Method | Command | Best for |
|--------|---------|----------|
| MAP | `fitter.run("map", optimizer="adam")` | Point estimates. Optimizer: adam/adamw/sgd/custom optax |
| Ray Tracing | `fitter.run("raytrace", n_burnin=100, n_steps=300)` | Exact MCMC, stochastic-gradient resilient (Behroozi 2025) |
| NUTS | `fitter.run("nuts", n_warmup=500, n_burnin=50)` | Gold-standard validation (low-D only) |
| geoVI | `fitter.run("geovi", n_iterations=15)` | Non-Gaussian posteriors, moderate D (Frank+2021) |
| MGVI | `fitter.run("mgvi", n_iterations=15)` | Fastest VI, very large D (>10^5) |
| Hierarchical | `HierarchicalFitter(models, data).run()` | Shared PSD across galaxy populations |

## Package structure

```
src/diffsed/
├── __init__.py              # Public API re-exports
├── distributions.py         # Uniform, Gaussian, LogUniform, LogNormal, Fixed, StudentT
├── plotting.py              # SED, SFH, corner, and diagnostic plots
├── simulate.py              # Mock data generation utilities
├── core/
│   ├── model.py             # Model: high-level interface (spec + SSP + filters -> predict)
│   ├── param_spec.py        # ParamSpec: parameter definitions, priors, validation
│   ├── param_translate.py   # High-level <-> internal parameter mapping
│   ├── prediction.py        # Prediction, SFHQuantities, SEDQuantities, DerivedQuantities
│   ├── noise.py             # Noise models (Gaussian, Student-t, variable noise)
│   ├── mock.py              # MockData generation
│   ├── fused_kernels.py     # Fused JIT kernels (weights + Z-interp + dust + einsum)
│   └── sed_pipeline.py      # Low-level SED pipeline
├── inference/
│   ├── fitter.py            # Fitter: unified interface to all inference methods
│   ├── posterior.py          # Posterior: summary tables, corner, SFH plots, ESS, R-hat
│   ├── hierarchical.py      # HierarchicalFitter: shared PSD via CorrelatedFieldMaker
│   ├── raytrace.py          # Ray Tracing sampler (Behroozi 2025, Apache 2.0)
│   ├── geovi.py             # geoVI wrapper (NIFTy.re)
│   ├── nuts.py              # NUTS wrapper (BlackJAX)
│   ├── map_optimizer.py     # MAP optimization (optax)
│   ├── vi_config.py         # VIConfig for geoVI/MGVI tuning
│   ├── standardized.py      # Standardized latent space transforms
│   ├── common.py            # Shared inference utilities
│   └── geovi_nuts.py        # Combined geoVI initialization + NUTS refinement
├── models/
│   ├── sfh/
│   │   ├── mean_sfh.py      # 8+ parametric SFH functions
│   │   ├── psd_models.py    # DRW, Matern, Extended Regulator PSDs
│   │   ├── gp_sfh.py        # IFT correlated field: IFFT(sqrt(P) * xi)
│   │   └── registry.py      # SFH + field model registry (composable SFH strings)
│   ├── dust/
│   │   ├── attenuation.py   # Two-component Charlot & Fall + pluggable laws
│   │   └── emission.py      # MBB, Dale+2014, Draine & Li 2007 dust emission
│   ├── agn/
│   │   ├── disc.py          # Accretion disc models (simple, standard, kubota_done)
│   │   ├── torus.py         # Torus emission
│   │   ├── qsogen.py        # QSOGen (Temple+2021) + Balmer continuum
│   │   ├── skirtor.py       # SKIRTOR tabulated torus templates (Stalevski+2016)
│   │   ├── blr.py           # Broad line region
│   │   ├── nlr.py           # Narrow line region
│   │   └── unified.py       # Unified NLR+BLR model
│   ├── nebular/
│   │   ├── baked_in.py      # SSP-embedded emission lines
│   │   ├── cloudy_grid.py   # CLOUDY photoionization grid interpolation
│   │   └── cue.py           # Cue neural emulator (Li+2024, JAX re-impl)
│   ├── sps/
│   │   ├── dsps_wrapper.py  # DSPS CSP integral, metallicity interpolation, SSP loading
│   │   ├── precompute.py    # Precomputed photometry/spectroscopy kernels
│   │   └── mass_remaining.py # IMF-based stellar mass remaining fractions
│   ├── observation/
│   │   ├── filters.py       # Filter loading (sedpy) and filter set management
│   │   ├── photometry.py    # Broadband filter convolution
│   │   ├── spectroscopy.py  # Pixel-level spectra + Chebyshev calibration
│   │   ├── calibration.py   # Spectroscopic calibration polynomials
│   │   └── eline_marginalization.py  # Emission line marginalization
│   ├── igm.py               # IGM absorption (Inoue+2014)
│   ├── radio.py             # Radio continuum (q_IR parametrization)
│   └── xray.py              # X-ray emission
├── utils/
│   ├── transforms.py        # Bounded <-> unbounded parameter bijections
│   ├── grid.py              # Log-age grid construction
│   ├── cosmology.py         # Flat LCDM distances, ages, lookback times
│   ├── sed_quantities.py    # Derived SED quantities (stellar mass, SFR, sSFR)
│   ├── devices.py           # JAX device management
│   └── optimizations.py     # Performance utilities
└── diagnostics/
    ├── fisher.py            # Fisher information matrix
    ├── saliency.py          # Parameter saliency maps
    └── green_functions.py   # Green's function response analysis
```

## Performance

Forward model timings on Apple M-series CPU:

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|-------------|-------------------|
| Forward model | 140 us | 356 us |
| Gradient | 56 us | 63 us |

Key optimizations:
- **Fused JIT kernels**: Single `@jax.jit` scope for weights + metallicity interpolation + dust + einsum, eliminating intermediate array materializations
- **Precomputed photometry**: SSPs through filters computed once at fixed redshift (21.6x speedup, following Zacharegkas+2025)
- **Precomputed spectroscopy**: SSPs pre-interpolated to observed wavelength grid
- **Precomputed dust age weights**: Sigmoid(log10(age)) computed once at Model init
- **Mixed precision**: `Model(spec, ssp, forward_dtype="float32")` halves memory, ~1.5x speed, <0.1% error
- **XLA compilation cache**: Persistent cache at `/tmp/diffsed_jax_cache` avoids recompilation across sessions

## Design principles

1. **Forward model first, inference second.** The core is a differentiable function mapping parameters to predicted SEDs. It can be used standalone for mock generation, plugged into any inference framework, or used to train emulators.

2. **Modular and composable.** Each physical component (SFH, dust, SPS, AGN, nebular, filters) is an independent pure JAX function. Adding a new dust law or AGN model = writing one function. No changes to inference.

3. **Standardized latent space.** All latent variables are mapped to xi ~ N(0, I) via differentiable bijections (following NIFTy.re and [Zacharegkas+2025](https://arxiv.org/abs/2506.19919) Appendix C). Samplers explore unconstrained space; the forward model handles the mapping to physical parameters.

4. **GPU-parallel by design.** The forward model is `jax.vmap`-able over galaxies. Following Zacharegkas+2025, independent HMC/NUTS chains can run in parallel across GPU threads, enabling ~1000 posteriors per GPU-minute.

5. **SSP-agnostic.** Any SSP template set in DSPS-compatible HDF5 format works. A [repository of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) from BC03, BPASS, FSPS, and ProGeny is publicly available for SPS uncertainty testing.

## Testing

```bash
# Run all tests (808 tests, ~60s)
pytest tests/ -q

# Run with coverage
pytest tests/ --cov=src/diffsed --cov-report=term-missing

# Lint and format (must pass before committing)
ruff check src/ tests/
ruff format --check src/ tests/

# Cross-validation against bagpipes/FSPS (not run by default)
pytest -m crossval tests/crossval/
```

808 tests across three tiers:
- **Unit tests** (`tests/unit/`): Pure function tests, no SSP data needed. PSD integrals, GP statistics, dust attenuation, parameter transforms, SFH shapes, fused kernels, noise models.
- **Integration tests** (`tests/integration/`): End-to-end pipeline tests with SSP data. Forward model, inference methods, mock recovery, precomputed photometry. Skips gracefully if SSP files are missing.
- **Cross-validation tests** (`tests/crossval/`): Numerical agreement against bagpipes and python-fsps. Excluded from default `pytest` runs; requires `bagpipes` or `SPS_HOME` for FSPS.

## Roadmap

- [x] Core GP + PSD machinery with DRW kernel
- [x] Mean SFH models (8+ parametric forms: tsnorm, dpl, snorm, norm, lnorm, const, exp, dexp, burst)
- [x] Composable SFH strings (e.g., "tsnorm+field", "dpl+burst+field")
- [x] Two-component Charlot & Fall dust attenuation with pluggable laws
- [x] Dust emission (MBB, Dale+2014, Draine & Li 2007 tabulated)
- [x] DSPS wrapper for CSP integral and metallicity interpolation
- [x] Photometry (filter convolution) and spectroscopy (pixel-level + calibration)
- [x] Precomputed photometry kernels (Zacharegkas+2025, 21.6x speedup)
- [x] Forward model composing all components
- [x] Parameter transforms (bounded/unbounded) and standardized latent space
- [x] Fused JIT kernels and mixed precision support
- [x] NIFTy.re inference (geoVI / MGVI via `optimize_kl`)
- [x] BlackJAX NUTS/HMC with GPU-parallel chains
- [x] Ray Tracing sampler (Behroozi 2025)
- [x] MAP optimization with optax (adam/adamw/sgd)
- [x] Mock generation and recovery test pipeline
- [x] Hierarchical population model (shared PSD parameters)
- [x] AGN models (disc, torus, QSOGen, SKIRTOR, BLR, NLR, unified)
- [x] Nebular emission (baked-in, CLOUDY grid, Cue neural emulator)
- [x] IGM absorption, radio, X-ray
- [x] Noise models (Gaussian, Student-t, variable noise)
- [x] Diagnostics (Fisher information, saliency, Green's functions)
- [x] Plotting utilities (SED fits, SFH, corner plots, diagnostics)
- [x] Comprehensive test suite (808 tests)
- [ ] Paper figures (in progress)
- [ ] Sphinx documentation
- [ ] Public release and PyPI package

## Dependencies

| Package | Role | Required |
|---------|------|----------|
| [JAX](https://github.com/google/jax) | Autodiff, JIT, GPU | Yes |
| [DSPS](https://github.com/ArgonneCPAC/dsps) | Differentiable SPS | Yes |
| [NumPy](https://github.com/numpy/numpy) | Array utilities | Yes |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | Plotting | Yes |
| [h5py](https://github.com/h5py/h5py) | SSP template I/O | Yes |
| [NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty) | geoVI / MGVI inference | Optional |
| [BlackJAX](https://github.com/blackjax-devs/blackjax) | NUTS / HMC sampling | Optional |
| [optax](https://github.com/google-deepmind/optax) | MAP optimization | Optional |
| [sedpy](https://github.com/bd-j/sedpy) | Filter transmission curves | Optional |

## References

- Ensslin, T. A. (2019). *Information field theory.* [arXiv:1804.03350](https://arxiv.org/abs/1804.03350)
- Frank, P. et al. (2021). *Geometric Variational Inference.* [arXiv:2105.10470](https://arxiv.org/abs/2105.10470)
- Edenhofer, G. et al. (2024). *Re-envisioning Numerical Information Field Theory (NIFTy.re).* [arXiv:2402.16683](https://arxiv.org/abs/2402.16683)
- Hearin, A. P. et al. (2023). *DSPS: Differentiable Stellar Population Synthesis.* [arXiv:2112.08423](https://arxiv.org/abs/2112.08423)
- Zacharegkas, G., Hearin, A. & Benson, A. (2025). *Bayesian Posteriors with Stellar Population Synthesis on GPUs.* [arXiv:2506.19919](https://arxiv.org/abs/2506.19919)
- Munoz, J. B. et al. (2026). *Relatively Fast and Reasonably Furious.* [arXiv:2601.07912](https://arxiv.org/abs/2601.07912)
- Wan, J. et al. (2024). *Stochastic prior for non-parametric SFHs.* [arXiv:2404.14494](https://arxiv.org/abs/2404.14494)
- Charlot, S. & Fall, S. M. (2000). *A Simple Model for the Absorption of Starlight by Dust in Galaxies.* [ApJ, 539, 718](https://ui.adsabs.harvard.edu/abs/2000ApJ...539..718C)
- Inoue, A. K. et al. (2014). *Updated analytic model for attenuation by the intergalactic medium.* [MNRAS, 442, 1805](https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I)

## License

MIT

## Acknowledgments

This code builds on the DSPS ecosystem developed by Andrew Hearin and collaborators at Argonne National Laboratory, and the NIFTy framework developed by the Information Field Theory group at the Max Planck Institute for Astrophysics. The Ray Tracing sampler implementation is based on Behroozi (2025), released under the Apache 2.0 license.
