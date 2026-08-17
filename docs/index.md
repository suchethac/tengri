<div class="tg-hero" markdown="0">
  <img src="_static/tengri-logo.png" alt="" class="tg-hero__logo" />
  <h1 class="tg-hero__title">tengri</h1>
  <p class="tg-hero__tagline">
    A differentiable framework for high-dimensional Bayesian inference
    from galaxy spectral energy distributions.
  </p>
</div>

Tengri is a panchromatic galaxy SED inference library in JAX. One forward model covers stellar populations, dust, nebular emission, AGN, and IGM, from X-rays to radio. Inference backends plug in as registrations: optimizers from `optax`, samplers from `BlackJAX`, variational inference from `NIFTy.re`. Gradients are exact.

Pre-1.0, developed as a community effort. The public API is still moving in places. The repository will move to the `tengri-project` GitHub organization shortly.

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from
the all-encompassing God of Heaven in traditional Turkic, Mongolic,
and other Central Asian nomadic religions. A fitting name for a code
that models the light of galaxies across cosmic time. This name is
chosen with respect for the cultural and spiritual traditions it
originates from; no religious claim or appropriation is intended.*

---

## At a glance

One object manages everything: all components (stellar, dust, nebular, AGN, IGM, radio, X-ray) register into a single SED chain. Pure JAX means JIT, `vmap`, and autodiff work from SSP grid to log-likelihood. One code path serves both grid exploration and Bayesian inference.

Build via recipe: `SEDModel.build(sfh={...}, dust={...}, neb={...})`. The `tengri.recipes` module covers standard cases.

```python
import jax
import tengri
from tengri import (
    SEDModel, Fixed, ForwardModel,
    Observation, Photometry, load_ssp_data, recipes,
)

# Downloads a stellar-population grid on first use, then caches it locally.
ssp = load_ssp_data(tengri.download_ssp("fsps_prsc_miles_chabrier"))
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

# Known-z mock: fixing redshift builds a single-z photometry table in
# seconds (leave the recipe's free z for real catalogs).
config  = recipes.star_forming_photometry()
config["redshift"] = Fixed(0.05)
sed     = SEDModel.build(ssp_data=ssp, observation=obs, **config)
forward = ForwardModel.build(sed=sed, observation=obs)

key  = jax.random.PRNGKey(0)
mock = sed.mock(sed.spec.sample(key), key=key)

result = forward.fit(mock.flux_obs, mock.noise, method="mcmc_nuts")
print(result.summary_table())
```

## Discover what's installed

```python
import tengri

tengri.summary()            # live counts across every registry
tengri.help()               # curated cheat-sheet
tengri.list_sfh_models()    # also list_nebular_backends, list_dust_laws, ...
tengri.describe("skirtor")  # paper, parameters, units, citation
tengri.search("torus")      # full-text across the registry
tengri.doctor()             # install + JAX backend + SSP files
```

Available from the CLI: `python -m tengri summary`. Nothing updates by hand: a model registered through `@register_agn_model` shows up in the registries without a documentation edit.

## Citation

Component-level citation: every model, SSP grid, and backend carries its own cite.

```python
tengri.print_components_bibtex(result)   # BibTeX for components that ran
```

While Paper I is in preparation, cite tengri as:

> Cooray et al., *tengri: A Differentiable Framework for
> High-Dimensional Bayesian Inference from Galaxy Spectral Energy
> Distributions*, in prep. (2026).

## How this was built

Built in close collaboration between a human author and AI agents. Components are verified against established codes; the development trail is kept open in the repository.

## Get involved

Contributors at every level are welcome. If a science case you care about isn't supported yet (a new emission mechanism, an observation mode, a sampler from a paper you read last week), that's exactly the conversation we want to have.

Contribute via the issue tracker at [github.com/suchethac/tengri](https://github.com/suchethac/tengri), or write [astro.tengri@gmail.com](mailto:astro.tengri@gmail.com) for anything that doesn't fit in an issue—collaborations, the org move, joining the project.

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
   spine/04_building_models

.. toctree::
   :caption: Common workflows
   :maxdepth: 1
   :hidden:

   spine/06_fitting_spectroscopy
   spine/07_joint_photo_spec
   spine/10_fastspecfit_joint_fit
   spine/11_catalog_fits
   spine/12_simulation_populations

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :titlesonly:
   :hidden:

   auto_examples/index

.. toctree::
   :caption: Physics reproduction
   :maxdepth: 1
   :hidden:

   reproduction/index

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   forward_model
   components
   api/index
   method_selection
   performance/index
   performance/memory
   performance/compilation
   known_limitations
   units

.. toctree::
   :caption: Experimental
   :maxdepth: 1
   :hidden:

   spine/experimental/index
```
