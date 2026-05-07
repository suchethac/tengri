# Tengri

**A JAX framework for differentiable galaxy SED fitting.** One modular forward model spans stars, dust, nebular emission, AGN, and IGM — from X-ray to radio. Every inference method (MAP, Laplace, Pathfinder, NUTS, Ray Tracing, Bayesian evidence, hierarchical population) runs on the same model, with gradients available everywhere.

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from the all-encompassing God of Heaven in traditional Turkic, Mongolic, and other Central Asian nomadic religions. A fitting name for a code that models the light of galaxies across cosmic time. This name is chosen with respect for the cultural and spiritual traditions it originates from; no religious claim or appropriation is intended.*

---

> **Status:** v0.1.0, active development. Core pipeline functional with 2000+ tests. Paper I in preparation.

## Why tengri

- **JIT-compiled, fully differentiable**: pure JAX end-to-end. Forward model ~140 μs, gradient ~56 μs on CPU for a smooth 7-D model. JIT + `vmap` + `grad` compose — one forward model powers every inference backend.
- **Modular physics**: stars (DSPS SSPs), SFH (parametric and non-parametric), dust attenuation (15+ laws) and emission, nebular (BakedIn / CloudyGrid / Cue), unified AGN (disc + torus + BLR/NLR), IGM absorption, radio and X-ray components. Each is a swappable pure function.
- **Every inference method, same model**: `fitter.run("map" | "laplace" | "pathfinder" | "mcmc_nuts" | "mcmc_raytrace" | "evidence")`. Add `PopulationFitter` for hierarchical fits across catalogues.
- **GPU/TPU native**: the same code runs on CPU, GPU, and TPU without modification.
- **BYO stellar library**: accepts any SSP in the DSPS HDF5 schema (BC03, BPASS, FSPS, ProGeny). [Pre-formatted templates here](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/).

Paper I covers the framework and mock-recovery validation with smooth / parametric SFHs. Paper II introduces Information-Field-Theory (IFT) correlated-field SFHs with PSD-governed burstiness priors and the geoVI inference pathway (Paper II preview, coming soon).

## Installation

```bash
pip install -e .              # core (JAX, DSPS, NIFTy)
pip install -e ".[all]"       # + BlackJAX (NUTS) + optax (MAP)
pip install -e ".[dev]"       # + pytest, ruff, jupytext
```

Requirements: Python ≥ 3.10, JAX ≥ 0.4.20, DSPS ≥ 0.3, NIFTy.re ≥ 8.5.

## Quick start

```python
from tengri import (
    SEDModel, Parameters, Fitter,
    Uniform, Gaussian,
    Observation, Photometry, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0, 4),
    redshift=0.1,
)

model = SEDModel(spec, ssp, observation=obs)
fitter = Fitter(model, obs_flux, obs_noise)
result = fitter.run("mcmc_nuts")   # swap for "map", "laplace", "pathfinder", etc.
print(result.summary_table())
```

See `notebooks/00_quickstart.py` for the full end-to-end walkthrough.

## Inference methods

| Method | Call | Best for |
|--------|------|----------|
| MAP | `fitter.run("map")` | Point estimates, initialization |
| Laplace | `fitter.run("laplace")` | Gaussian posterior from Hessian at MAP |
| Pathfinder | `fitter.run("pathfinder")` | Fast approximate posterior; good NUTS warm-start |
| NUTS | `fitter.run("mcmc_nuts")` | Gold-standard posterior (D ≲ 30) |
| Ray Tracing | `fitter.run("mcmc_raytrace")` | Exact MCMC, noise-robust, scales past D = 30 |
| Evidence (NSS) | `fitter.run("evidence")` | Bayesian evidence for model comparison |
| Population | `PopulationFitter(...)` | Shared hyperparameters across galaxy samples |
| geoVI / `vi_native` | `fitter.run("vi")` / `"vi_native"` | **Paper II preview.** High-D stochastic SFHs (D ≈ 137+) |

Method choice is introduced in `notebooks/05_fitting_photometry.py`; a deeper walkthrough will land in a future spine notebook.

## Tutorial spine

Tutorials live as Jupytext `.py` files in [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) and are synced to `docs/spine/*.ipynb` via `python scripts/sync_spine_notebooks_for_docs.py`. For one-figure recipes, see the [examples gallery](examples.md) — 70+ thumbnailed scripts, each runnable standalone.

The spine is written for astronomers — physics framing, copy-paste-able code cells, progressive teaching across notebooks. Start with `00` and `01`, then branch based on your use case.

## Performance (Apple M-series CPU)

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|--------------|--------------------|
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
| [h5py](https://github.com/h5py/h5py) | SSP I/O | Yes |
| [BlackJAX](https://github.com/blackjax-devs/blackjax) | NUTS / HMC | Optional |
| [optax](https://github.com/google-deepmind/optax) | MAP optimization | Optional |

## References

- Frank, P. et al. (2021). *Geometric Variational Inference.* [arXiv:2105.10470](https://arxiv.org/abs/2105.10470)
- Hearin, A. P. et al. (2023). *DSPS: Differentiable Stellar Population Synthesis.* [arXiv:2112.08423](https://arxiv.org/abs/2112.08423)
- Edenhofer, G. et al. (2024). *Re-envisioning NIFTy.re.* [arXiv:2402.16683](https://arxiv.org/abs/2402.16683)
- Behroozi, P. (2025). *Ray Tracing Sampler.* [arXiv:2504.20029](https://arxiv.org/abs/2504.20029)
- Yallup, D., Kroupa, S. & Handley, W. (2026). *Nested Slice Sampling.* [arXiv:2601.23252](https://arxiv.org/abs/2601.23252)
- Zhang, L. et al. (2022). *Pathfinder.* [arXiv:2108.03782](https://arxiv.org/abs/2108.03782)
- Murray, I., Adams, R. P. & MacKay, D. J. C. (2010). *Elliptical Slice Sampling.* [arXiv:1001.0175](https://arxiv.org/abs/1001.0175)
- Ensslin, T. A. (2019). *Information Field Theory.* [arXiv:1804.03350](https://arxiv.org/abs/1804.03350)

**License:** MIT

```{eval-rst}
.. toctree::
   :caption: Foundations
   :maxdepth: 1

   spine/00_quickstart
   spine/01_why_jax
   spine/02_sed_anatomy
   spine/03_discovering_the_menu
   spine/04_building_models

.. toctree::
   :caption: Fitting workflows
   :maxdepth: 1

   spine/05_fitting_photometry
   spine/06_fitting_spectroscopy
   spine/07_joint_photo_spec

.. toctree::
   :caption: Physics deep dives
   :maxdepth: 1

   spine/08_emission_lines

.. toctree::
   :caption: Examples gallery
   :maxdepth: 2

   examples
   auto_examples/index
```
