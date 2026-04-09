# Tengri

A fast, modular JAX framework for Bayesian galaxy SED fitting. Scalable from individual posteriors to hierarchical population inference across thousands of dimensions. Built to be the comprehensive, actively developed tool for extracting the fullest physical picture of galaxies from multiwavelength observations.

**Documentation (HTML):** [https://suchethac.github.io/tengri/](https://suchethac.github.io/tengri/) · **Notebooks:** [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks)

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from the all-encompassing God of Heaven in traditional Turkic, Mongolic, and other Central Asian nomadic religions. Tengri is the supreme sky deity in Tengrism, the eternal source of order in the natural world. A fitting name for a code that models the light of galaxies across cosmic time. This name is chosen with respect for the cultural and spiritual traditions it originates from; no religious claim or appropriation is intended.*

---

> **Status:** v0.1.0, active development. Core pipeline fully functional with 2000+ unit and integration tests. Paper in preparation.

## Highlights

- **IFT-based SFH model**: Star formation history as a continuous 1D field reconstructed from noisy SED data via Information Field Theory (Ensslin 2019). The PSD encodes the amplitude and timescale of burstiness.
- **Fully differentiable**: Pure JAX from PSD parameters through to predicted photometry. Gradients via autodiff enable HMC, variational inference, and gradient-based optimization.
- **GPU-native**: All operations are JIT-compiled and run on GPU/TPU. Designed for catalog-scale inference.
- **Modular forward model**: Every component (SFH, dust, SPS, AGN, nebular, observation) is a swappable pure function.
- **Inference as first-class**: Exact autodiff through the full model powers **geoVI** (`vi`), **Ray Tracing**, **NUTS**, evidence sampling, and hierarchical fits on the **same** standardized loss—not a bolt-on optimizer per method.
- **Multiple inference backends**: MAP, Ray Tracing (Behroozi 2025), NUTS ([BlackJAX](https://github.com/blackjax-devs/blackjax)), geoVI/MGVI ([NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty)), Laplace, Pathfinder (Zhang+2022), Elliptical Slice Sampling (Murray+2010), Nested Slice Sampling (Yallup+2026) for Bayesian evidence, and hierarchical population fitting.
- **DSPS-powered SPS**: Differentiable stellar population synthesis via [DSPS](https://github.com/ArgonneCPAC/dsps) (Hearin et al. 2023). Accepts any SSP template in HDF5 format.

## Installation

```bash
pip install -e .              # core install (JAX, DSPS, NIFTy)
pip install -e ".[all]"       # + BlackJAX (NUTS) + optax (MAP)
pip install -e ".[dev]"       # + pytest, ruff, jupytext
```

**Requirements:** Python >= 3.10, JAX >= 0.4.20, DSPS >= 0.3, NIFTy.re >= 8.5

## SSP grids

tengri requires pre-computed Simple Stellar Population (SSP) grids in DSPS-compatible HDF5 format. A [repository of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) from BC03, BPASS, FSPS, and ProGeny is publicly available.

```bash
# Download an SSP grid (e.g., FSPS v3.2)
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/ssp_fsps_v3.2.h5 -P data/
```

Any SSP template set can be used — the only requirement is the DSPS HDF5 schema (age, metallicity, wavelength, spectra arrays). This enables SPS uncertainty testing across different stellar libraries and IMFs.

## How it works

```
┌─────────────────────────────────────────────────────────┐
│                    tengri architecture                  │
│                                                         │
│   Observations ──────────────────► Fitter ─► Posterior │
│                                       ▲                 │
│                                  Model(spec, ssp)       │
│                                  ▲           ▲          │
│                              Parameters    SSP grid     │
│                              (physics)    (templates)   │
└─────────────────────────────────────────────────────────┘
```

**Parameters** declares the free parameters and their priors (SFH shape, dust, metallicity, redshift). **SSP grid** holds the pre-computed stellar population spectra (any DSPS-compatible HDF5 file). **Model** combines them into a differentiable forward model that maps physical parameters to predicted photometry or spectra. **Fitter** runs inference (MAP, VI, MCMC) and returns a **Posterior** with posterior samples, summary statistics, and convergence diagnostics.

Forward-model hub (Sphinx): [Forward model](docs/forward_model/index.md).

## Documentation map

| Section | Role |
|---------|------|
| [Getting started](docs/getting_started/index.md) | Install, first notebook, pointer to the spine |
| [Forward model](docs/forward_model/index.md) | Physics modules and links to galleries **01–06**, **13** |
| [Inference](docs/inference/index.md) | `Fitter`, methods, diagnostics, spine **00**, **07–08**, **14**, **09**, **11** |
| [Fitting workflows](docs/fitting/index.md) | Photometry, spectra, degeneracies, real data |
| [Advanced](docs/advanced/index.md) | Convergence, batch vs population, extending, hierarchical |
| [Examples](docs/examples.md) | Sphinx gallery: one figure per script (supplementary sweeps) |
| [Developer](docs/developer/index.md) · [API](docs/api/index.md) | Internals and reference |

## Reader spine (notebooks)

Canonical tutorials are **Jupytext** percent-format `.py` files at the **repository root**
[`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) (run locally or sync to `.ipynb` with `jupytext`).

**Suggested teaching order:** **00 → 01 → 02 → 13 → 03–06 → 07–12**, then **14** after **08** when you want joint photometry+spectroscopy corners. (Files **13** and **14** are numbered for insertion; the table below lists all spine notebooks in **numeric** order.)

| # | Notebook | Topic |
|---|----------|-------|
| 00 | [`00_quickstart.py`](https://github.com/suchethac/tengri/blob/main/notebooks/00_quickstart.py) | Panchromatic SED + spectrum fits (`vi`, NUTS, Ray Tracing) |
| 01 | [`01_sed_anatomy.py`](https://github.com/suchethac/tengri/blob/main/notebooks/01_sed_anatomy.py) | Wavelength ↔ physics, component build-up |
| 02 | [`02_sfh_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/02_sfh_gallery.py) | SFH models, PSD / GP burstiness theory |
| 03 | [`03_dust_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/03_dust_gallery.py) | Dust attenuation and IR emission |
| 04 | [`04_nebular_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/04_nebular_gallery.py) | Nebular backends, lines, shocks |
| 05 | [`05_agn_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/05_agn_gallery.py) | AGN discs, torus, unified models |
| 06 | [`06_multiwavelength_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/06_multiwavelength_gallery.py) | IGM, radio, X-ray |
| 07 | [`07_fitting_photometry.py`](https://github.com/suchethac/tengri/blob/main/notebooks/07_fitting_photometry.py) | Photometry fitting, batch / catalog |
| 08 | [`08_fitting_spectra.py`](https://github.com/suchethac/tengri/blob/main/notebooks/08_fitting_spectra.py) | Spectroscopy, phot vs spec |
| 09 | [`09_degeneracies.py`](https://github.com/suchethac/tengri/blob/main/notebooks/09_degeneracies.py) | Degeneracies, Fisher / correlations |
| 10 | [`10_real_data.py`](https://github.com/suchethac/tengri/blob/main/notebooks/10_real_data.py) | Real survey-style workflow |
| 11 | [`11_population.py`](https://github.com/suchethac/tengri/blob/main/notebooks/11_population.py) | Hierarchical / population inference |
| 12 | [`12_extending_tengri.py`](https://github.com/suchethac/tengri/blob/main/notebooks/12_extending_tengri.py) | Custom priors, modules, registries |
| 13 | [`13_tabulated_sfh_to_mock_sed.py`](https://github.com/suchethac/tengri/blob/main/notebooks/13_tabulated_sfh_to_mock_sed.py) | Tabulated (t, SFR) → SED + mock photometry (`simulate`) |
| 14 | [`14_joint_photometry_spectroscopy.py`](https://github.com/suchethac/tengri/blob/main/notebooks/14_joint_photometry_spectroscopy.py) | Joint observation, phot / spec / joint posteriors |

More on inference API: [Inference](docs/inference/index.md) · [Convergence](docs/advanced/convergence.md).

**Gallery scripts** ([Examples](docs/examples.md)) complement the spine: short `examples/*.py` snippets that each render one thumbnail figure (useful for parameter sweeps). They do not replace the narrative notebooks above.

## Quick start

```python
from tengri import Model, Parameters, Fitter, Uniform, Gaussian
from tengri import Observation, Photometry, load_ssp_data

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0, 4),
    redshift=0.1,
)

model = Model(spec, ssp, observation=obs)
# obs_flux, obs_noise = ...  # your data vectors, same units as model.predict_photometry
fitter = Fitter(model, obs_flux, obs_noise)
result = fitter.run("vi")  # or "mcmc_raytrace", "mcmc_nuts", "map", "laplace", "evidence"
print(result.summary_table())
```

## Inference methods

| Method | Command | Best for |
|--------|---------|----------|
| `map` | `fitter.run("map")` | Point estimates |
| `vi` | `fitter.run("vi")` | **Default.** geoVI via NIFTy, nonlinear posteriors |
| `vi_linear` | `fitter.run("vi_linear")` | Linear MGVI, very high D or catalog fitting |
| `mcmc_raytrace` | `fitter.run("mcmc_raytrace")` | Exact MCMC, stochastic-gradient resilient |
| `mcmc_nuts` | `fitter.run("mcmc_nuts")` | Gold-standard validation (D ≲ 30) |
| `laplace` | `fitter.run("laplace")` | Instant Gaussian posterior from Hessian at MAP |
| `pathfinder` | `fitter.run("pathfinder")` | Fast approximate posterior, good NUTS initializer |
| `mcmc_ess` | `fitter.run("mcmc_ess")` | Exact MCMC for Gaussian-prior latent models |
| `evidence` | `fitter.run("evidence")` | Bayesian evidence for model comparison (D ≲ 30) |
| Population | `model.fit_population(observations)` | Shared PSD across populations (`PopulationPosterior`) |

## Performance

Forward model timings on Apple M-series CPU:

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|-------------|-------------------|
| Forward model | 140 μs | 356 μs |
| Gradient | 56 μs | 63 μs |

## Dependencies

| Package | Role | Required |
|---------|------|----------|
| [JAX](https://github.com/google/jax) | Autodiff, JIT, GPU | Yes |
| [DSPS](https://github.com/ArgonneCPAC/dsps) | Differentiable SPS | Yes |
| [NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty) | geoVI / MGVI | Yes |
| [NumPy](https://github.com/numpy/numpy) | Array utilities | Yes |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | Plotting | Yes |
| [h5py](https://github.com/h5py/h5py) | SSP template I/O | Yes |
| [BlackJAX](https://github.com/blackjax-devs/blackjax) | NUTS / HMC | Optional |
| [optax](https://github.com/google-deepmind/optax) | MAP optimization | Optional |

## References

- Frank, P. et al. (2021). *Geometric Variational Inference.* [arXiv:2105.10470](https://arxiv.org/abs/2105.10470)
- Hearin, A. P. et al. (2023). *DSPS: Differentiable Stellar Population Synthesis.* [arXiv:2112.08423](https://arxiv.org/abs/2112.08423)
- Edenhofer, G. et al. (2024). *Re-envisioning Numerical Information Field Theory (NIFTy.re).* [arXiv:2402.16683](https://arxiv.org/abs/2402.16683)
- Behroozi, P. (2025). *Ray Tracing Sampler.* [arXiv:2504.20029](https://arxiv.org/abs/2504.20029)
- Yallup, D., Kroupa, S. & Handley, W. (2026). *Nested Slice Sampling.* [arXiv:2601.23252](https://arxiv.org/abs/2601.23252)
- Zhang, L. et al. (2022). *Pathfinder: Parallel quasi-Newton variational inference.* [arXiv:2108.03782](https://arxiv.org/abs/2108.03782)
- Murray, I., Adams, R. P. & MacKay, D. J. C. (2010). *Elliptical Slice Sampling.* [arXiv:1001.0175](https://arxiv.org/abs/1001.0175)
- Ensslin, T. A. (2019). *Information field theory.* [arXiv:1804.03350](https://arxiv.org/abs/1804.03350)

## License

MIT
