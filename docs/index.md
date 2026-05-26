# Tengri

Tengri is a panchromatic galaxy SED inference library, written in
JAX. The same forward model handles stars, dust, nebular emission,
AGN, and the IGM, from X-rays out to the radio, and every sampler we
ship — MAP, Laplace, Pathfinder, NUTS, ray-tracing MCMC, nested
sampling, geoVI, and hierarchical population fits — runs against it.
Gradients are available everywhere, and they are exact.

This is pre-1.0 research code, developed as a community effort. The
public API is still moving in places, several physics modules are
being independently human-verified, and the Paper I draft is in
preparation. The repository will move to the `tengri-project`
GitHub organisation shortly; collaborative development and issue
tracking will live there going forward.

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from
the all-encompassing God of Heaven in traditional Turkic, Mongolic,
and other Central Asian nomadic religions. A fitting name for a code
that models the light of galaxies across cosmic time. This name is
chosen with respect for the cultural and spiritual traditions it
originates from; no religious claim or appropriation is intended.*

---

## At a glance

The forward model is one object. Stars, dust, nebular gas, AGN, IGM,
radio, and X-ray are all registered components that compose into a
single SED chain. There is no separate fast path for grid fits and
slow path for Bayesian inference; both run against the same code.

Everything is differentiable. Pure-JAX components mean JIT compilation
and `vmap` come for free, and autodiff is clean from the SSP grid
through to the log-likelihood.

Building a model reads like a recipe rather than a configuration
file. `SEDModel.build(sfh={...}, dust={...}, neb={...})` and the
curated entries in `tengri.recipes` cover most of the cases we have
needed in practice. `model.spec.summary()` shows you which parameters
you set, which are free, and which fell back to library defaults.

```python
import jax
from tengri import (
    SEDModel, Parameters, Fitter, ForwardModel,
    Uniform, Gaussian, Observation, Photometry, load_ssp_data,
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
sed     = SEDModel(spec, ssp, observation=obs)
forward = ForwardModel.build(sed=sed, observation=obs)

key  = jax.random.PRNGKey(0)
mock = sed.mock(spec.sample(key), key=key)

fitter = Fitter(forward, mock["flux_obs"], mock["noise"])
result = fitter.run("mcmc_nuts")
print(result.summary_table())
```

The full walkthrough — how the mock is constructed, what the priors
do, how to read the corner plot — is in
[`notebooks/00_quickstart.py`](spine/00_quickstart).

## Where to go next

If you are new, read the [Overview](overview) and then work through
the [quickstart notebook](spine/00_quickstart). If you are coming from
CIGALE, Prospector, or BAGPIPES, the [Reproduction](reproduction/index)
section is where the cross-validation against those codes lives.
Developers and contributors should start with
[`docs/dev/forward-model-architecture.md`](https://github.com/suchethac/tengri/blob/main/docs/dev/forward-model-architecture.md)
and the
[ADR index](https://github.com/suchethac/tengri/tree/main/docs/adr).

## Inference methods

| Method | Call | Best for |
|--------|------|----------|
| MAP | `fitter.run("map")` | Point estimates, initialisation, warm-starts |
| Laplace | `fitter.run("laplace")` | Gaussian posterior from the Hessian at MAP |
| Pathfinder | `fitter.run("pathfinder")` | Fast approximate posterior; good NUTS warm-start |
| NUTS | `fitter.run("mcmc_nuts")` | Reference posterior for D ≲ 30 |
| Ray-tracing | `fitter.run("mcmc_raytrace")` | Exact MCMC; tolerates noisy gradients; scales past D = 30 |
| Nested sampling | `fitter.run("nss")` | Bayesian evidence for model comparison |
| Population | `PopulationSEDModel` + `ForwardModel.build(population=...)` | Shared hyperparameters across galaxy samples |
| geoVI / `vi_native` | `fitter.run("vi")` / `"vi_native"` | **Paper II preview.** High-D stochastic SFHs (D ≈ 137+) |

`tengri.list_inference_methods()` returns the live registry,
including experimental backends (MCLMC, ghMC, ESS, …).

## Citation

Tengri is research software. If it shows up in a publication, please
cite the methods paper and the upstream codes whose physics and
samplers we are building on. The full BibTeX block and the
acknowledgement list are on the [Citing tengri](citation) page. While
Paper I is in preparation, the shortest in-text citation is:

> Cooray et al., *tengri: A Differentiable Framework for
> High-Dimensional Bayesian Inference from Galaxy Spectral Energy
> Distributions*, in prep. (2026).

For automatic, fit-specific BibTeX:

```python
print(tengri.cite_all(result))
```

## License

BSD-3-Clause. See [LICENSE](https://github.com/suchethac/tengri/blob/main/LICENSE)
and [NOTICE](https://github.com/suchethac/tengri/blob/main/NOTICE).

```{eval-rst}
.. toctree::
   :caption: Getting started
   :maxdepth: 1
   :hidden:

   overview
   installation
   spine/00_quickstart
   spine/01_why_jax
   citation

.. toctree::
   :caption: Foundations
   :maxdepth: 1
   :hidden:

   spine/02_sed_anatomy
   spine/03_discovering_the_menu

.. toctree::
   :caption: Fitting workflows
   :maxdepth: 1
   :hidden:

   spine/05_fitting_photometry
   spine/06_fitting_spectroscopy
   spine/07_joint_photo_spec

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   auto_examples/index

.. toctree::
   :caption: Reproduction
   :maxdepth: 1
   :hidden:

   reproduction/index

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   spine/04_building_models
   performance/index
   performance/memory
   units
```
