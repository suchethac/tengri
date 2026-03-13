# diffsed

**Differentiable SED fitting with Information Field Theory star formation history priors.**

A modular, fully differentiable JAX code for galaxy spectral energy distribution (SED) fitting. The star formation history is modeled as a continuous correlated field governed by a power spectral density (PSD) kernel, enabling physically motivated burstiness priors. End-to-end gradients flow through the entire pipeline, enabling fast variational and gradient-based inference.

> **Status:** Active development (v0.1.0). Core modules implemented and tested. Paper in preparation.

## Highlights

- **IFT-based SFH model**: Star formation history as a continuous 1D field reconstructed from noisy SED data via Information Field Theory (Ensslin 2019). The PSD encodes the amplitude and timescale of burstiness.
- **Fully differentiable**: Pure JAX from PSD parameters through to predicted photometry. Gradients via autodiff enable HMC, variational inference, and gradient-based optimization.
- **GPU-native**: All operations are JIT-compiled and run on GPU/TPU. Designed for catalog-scale inference following the approach of [Zacharegkas, Hearin & Benson (2025)](https://arxiv.org/abs/2506.19919).
- **Modular forward model**: Every component (SFH, dust, SPS, observation) is a swappable pure function. The forward model is the primary product; inference is one application of it.
- **Multiple inference backends**: geoVI/MGVI ([NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty)), NUTS/HMC ([BlackJAX](https://github.com/blackjax-devs/blackjax)), MAP optimization ([optax](https://github.com/google-deepmind/optax)).
- **DSPS-powered SPS**: Differentiable stellar population synthesis via [DSPS](https://github.com/ArgonneCPAC/dsps) (Hearin et al. 2023). Accepts any SSP template in HDF5 format.

## Installation

```bash
# Basic install (JAX + DSPS)
pip install -e .

# With all inference backends
pip install -e ".[all]"

# Development (adds pytest)
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, JAX >= 0.4.20, DSPS >= 0.3

## Quick start

```python
import jax
import jax.numpy as jnp
from diffsed.models.sfh.psd_models import psd_drw, drw_variance
from diffsed.models.sfh.gp_sfh import gp_from_xi, compute_sqrt_power_drw
from diffsed.models.sfh.mean_sfh import double_powerlaw
from diffsed.utils.grid import make_log_age_grid, log_age_to_age_yr, grid_spacing

# Set up the log-age grid (1 Myr to 13.8 Gyr, 256 points)
log_age_grid = make_log_age_grid(256)
d_log_age = grid_spacing(log_age_grid)
age_yr = log_age_to_age_yr(log_age_grid)

# Define DRW PSD parameters
sigma_ps = 1.5   # burstiness amplitude
tau_ps = 30e6    # 30 Myr damping timescale

# Pre-compute amplitude operator (with Jacobian correction for log-age grid)
sqrt_power = compute_sqrt_power_drw(256, float(d_log_age), sigma_ps, tau_ps)

# Draw a GP realization from standardized latent variables
key = jax.random.PRNGKey(42)
xi = jax.random.normal(key, shape=(256,))   # xi ~ N(0, I)
gp_x = gp_from_xi(xi, sqrt_power, 256)      # correlated field

# Combine with smooth mean SFH (BAGPIPES-style double power law)
sfr_mean = double_powerlaw(age_yr, alpha=1.5, beta=0.8, tau=2e9, norm=5.0)

# Full SFH with lognormal correction
k0_half = drw_variance(sigma_ps) / 2.0
sfr = sfr_mean * jnp.exp(gp_x - k0_half)

# Gradients flow through the entire pipeline
grad_fn = jax.grad(lambda xi: jnp.sum(
    sfr_mean * jnp.exp(gp_from_xi(xi, sqrt_power, 256) - k0_half)
))
grad_xi = grad_fn(xi)  # shape (256,), all finite
```

## Package structure

```
src/diffsed/
├── models/
│   ├── sfh/
│   │   ├── psd_models.py        # DRW, Matern, Extended Regulator PSDs
│   │   ├── gp_sfh.py            # IFT correlated field: IFFT(sqrt(P) * xi)
│   │   └── mean_sfh.py          # Double power law, delayed-tau, constant
│   ├── dust/
│   │   └── charlot_fall.py      # Charlot & Fall (2000), smooth + hard
│   ├── sps/
│   │   └── dsps_wrapper.py      # DSPS CSP integral, metallicity interpolation
│   └── observation/
│       ├── photometry.py        # Filter convolution
│       └── spectroscopy.py      # Pixel-level spectra + Chebyshev calibration
├── utils/
│   ├── transforms.py            # Bounded <-> unbounded parameter transforms
│   ├── grid.py                  # Log-age grid construction
│   └── cosmology.py             # Flat LCDM distances, ages
├── forward_model.py             # Full pipeline: params -> SED/photometry
├── inference/                   # [planned] geoVI, NUTS, MAP wrappers
└── diagnostics/                 # [planned] PSD recovery, posterior checks
```

## The model

The star formation rate is decomposed into a smooth secular component and stochastic fluctuations:

```
ln SFR(t) = ln SFR_mean(t) + x(t) - K(0)/2
```

where:
- **SFR_mean(t)** is a parametric mean SFH (double power law by default)
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

## Design principles

1. **Forward model first, inference second.** The core is a differentiable function mapping parameters to predicted SEDs. It can be used standalone for mock generation, plugged into any inference framework, or used to train emulators.

2. **Modular and composable.** Each physical component (SFH, dust, SPS, filters) is an independent pure JAX function. Adding a new dust model = writing one function. No changes to inference.

3. **Standardized latent space.** All latent variables are mapped to xi ~ N(0, I) via differentiable bijections (following NIFTy.re and [Zacharegkas+2025](https://arxiv.org/abs/2506.19919) Appendix C). Samplers explore unconstrained space; the forward model handles the mapping to physical parameters.

4. **GPU-parallel by design.** The forward model is `jax.vmap`-able over galaxies. Following Zacharegkas+2025, independent HMC/NUTS chains can run in parallel across GPU threads, enabling ~1000 posteriors per GPU-minute.

5. **SSP-agnostic.** Any SSP template set in DSPS-compatible HDF5 format works. A [repository of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) from BC03, BPASS, FSPS, and ProGeny is publicly available for SPS uncertainty testing.

## Computational tricks

Inspired by [Zacharegkas, Hearin & Benson (2025)](https://arxiv.org/abs/2506.19919):

- **Approximate photometry**: Pre-compute SSP broadband fluxes c_SSP on the (age, Z) grid. Galaxy photometry reduces to a weighted sum with dust evaluated at the effective wavelength. Speedup of 30-50x with <0.1% error. *(planned)*
- **Bounded parameters via sigmoid**: All physical parameters are bounded via differentiable sigmoid transforms, ensuring gradient-based samplers never leave the physical domain.
- **JIT-compiled forward model**: The entire pipeline from xi to predicted photometry is a single JIT-compiled function with no Python overhead at runtime.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/diffsed --cov-report=term-missing
```

72 tests covering:
- **T1**: PSD integral = sigma_PS^2 / 2 (Wiener-Khinchin theorem)
- **T2**: GP periodogram shape matches input PSD
- **T3**: GP variance scales correctly with PSD level
- Mean SFH properties (peak location, normalization, gradients)
- Dust attenuation (bounds, wavelength dependence, age dependence, gradients)
- Parameter transform roundtrips (bounded <-> unbounded)
- JIT compatibility and autodiff for all core functions

## Roadmap

- [x] Core GP + PSD machinery with DRW kernel
- [x] Mean SFH models (double power law, delayed-tau, constant)
- [x] Charlot & Fall dust attenuation (smooth sigmoid variant)
- [x] DSPS wrapper for CSP integral and metallicity interpolation
- [x] Photometry (filter convolution) and spectroscopy (pixel-level + calibration)
- [x] Forward model composing all components
- [x] Parameter transforms (bounded/unbounded)
- [x] Comprehensive test suite (72 tests)
- [ ] Approximate photometry (Zacharegkas+2025 Section 3)
- [ ] NIFTy.re inference wrapper (geoVI / MGVI via `optimize_kl`)
- [ ] BlackJAX NUTS/HMC wrapper with GPU-parallel chains
- [ ] Mock generation and recovery test pipeline
- [ ] Hierarchical population model (shared PSD parameters)
- [ ] Nebular emission, dust emission, AGN components
- [ ] Example notebooks and Sphinx documentation
- [ ] Paper figures

## Dependencies

| Package | Role | Required |
|---------|------|----------|
| [JAX](https://github.com/google/jax) | Autodiff, JIT, GPU | Yes |
| [DSPS](https://github.com/ArgonneCPAC/dsps) | Differentiable SPS | Yes |
| [NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty) | geoVI / MGVI inference | Optional |
| [BlackJAX](https://github.com/blackjax-devs/blackjax) | NUTS / HMC sampling | Optional |
| [optax](https://github.com/google-deepmind/optax) | MAP optimization | Optional |

## References

- Ensslin, T. A. (2019). *Information field theory.* [arXiv:1804.03350](https://arxiv.org/abs/1804.03350)
- Frank, P. et al. (2021). *Geometric Variational Inference.* [arXiv:2105.10470](https://arxiv.org/abs/2105.10470)
- Edenhofer, G. et al. (2024). *Re-envisioning Numerical Information Field Theory (NIFTy.re).* [arXiv:2402.16683](https://arxiv.org/abs/2402.16683)
- Hearin, A. P. et al. (2023). *DSPS: Differentiable Stellar Population Synthesis.* [arXiv:2112.08423](https://arxiv.org/abs/2112.08423)
- Zacharegkas, G., Hearin, A. & Benson, A. (2025). *Bayesian Posteriors with Stellar Population Synthesis on GPUs.* [arXiv:2506.19919](https://arxiv.org/abs/2506.19919)
- Munoz, J. B. et al. (2026). *Relatively Fast and Reasonably Furious.* [arXiv:2601.07912](https://arxiv.org/abs/2601.07912)
- Wan, J. et al. (2024). *Stochastic prior for non-parametric SFHs.* [arXiv:2404.14494](https://arxiv.org/abs/2404.14494)

## License

MIT

## Acknowledgments

This code builds on the DSPS ecosystem developed by Andrew Hearin and collaborators at Argonne National Laboratory, and the NIFTy framework developed by the Information Field Theory group at the Max Planck Institute for Astrophysics.
