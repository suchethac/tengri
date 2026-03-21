# Tengri

A fast, modular JAX framework for Bayesian galaxy SED fitting. Scalable from individual posteriors to hierarchical population inference across thousands of dimensions. Built to be the comprehensive, actively developed tool for extracting the fullest physical picture of galaxies from multiwavelength observations.

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from the all-encompassing God of Heaven in traditional Turkic, Mongolic, and other Central Asian nomadic religions. Tengri is the supreme sky deity in Tengrism, the eternal source of order in the natural world. A fitting name for a code that models the light of galaxies across cosmic time. This name is chosen with respect for the cultural and spiritual traditions it originates from; no religious claim or appropriation is intended.*

---

> **Status:** v0.1.0, active development. Core pipeline fully functional with 808 tests. Paper in preparation.

## Highlights

- **IFT-based SFH model**: Star formation history as a continuous 1D field reconstructed from noisy SED data via Information Field Theory (Ensslin 2019). The PSD encodes the amplitude and timescale of burstiness.
- **Fully differentiable**: Pure JAX from PSD parameters through to predicted photometry. Gradients via autodiff enable HMC, variational inference, and gradient-based optimization.
- **GPU-native**: All operations are JIT-compiled and run on GPU/TPU. Designed for catalog-scale inference following the approach of [Zacharegkas, Hearin & Benson (2025)](https://arxiv.org/abs/2506.19919).
- **Modular forward model**: Every component (SFH, dust, SPS, AGN, nebular, observation) is a swappable pure function. The forward model is the primary product; inference is one application of it.
- **Multiple inference backends**: MAP, Ray Tracing (Behroozi 2025), NUTS ([BlackJAX](https://github.com/blackjax-devs/blackjax)), geoVI/MGVI ([NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty)), and hierarchical population fitting.
- **DSPS-powered SPS**: Differentiable stellar population synthesis via [DSPS](https://github.com/ArgonneCPAC/dsps) (Hearin et al. 2023). Accepts any SSP template in HDF5 format.

## Quick start

```python
from tengri import Model, ParamSpec, Fitter, Posterior, Uniform, Gaussian
from tengri import load_ssp_data, load_filter_set

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

## Documentation

- {doc}`Installation <install>` — setup and requirements
- {doc}`Tutorials <tutorials/index>` — learn tengri step by step
- {doc}`Demonstrations <demonstrations/index>` — science workflows (spectroscopy, catalogs, hierarchical)
- {doc}`Reference <reference/index>` — physics deep-dives (PSD, dust, AGN, nebular, noise)
- {doc}`Observation Guide <observation/index>` — unified Observation API
- {doc}`Performance <performance/index>` — benchmarks, optimization, profiling
- {doc}`Advanced <advanced/index>` — convergence diagnostics, batch fitting, extending tengri
- {doc}`API Reference <api/index>` — auto-generated from docstrings
- {doc}`Developer Guide <developer/index>` — architecture, contributing, internals

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

## License

MIT

```{toctree}
:maxdepth: 2
:hidden:

install
tutorials/index
demonstrations/index
reference/index
observation/index
performance/index
advanced/index
api/index
developer/index
changelog
```
