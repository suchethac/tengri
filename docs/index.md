<div class="tg-hero" markdown="0">
  <img src="_static/tengri-logo.png" alt="" class="tg-hero__logo" />
  <h1 class="tg-hero__title">tengri</h1>
  <p class="tg-hero__tagline">
    A differentiable framework for high-dimensional Bayesian inference
    from galaxy spectral energy distributions.
  </p>
</div>

Tengri is a panchromatic galaxy SED inference library, written in
JAX. The same forward model covers stellar populations, dust,
nebular emission, AGN, and the IGM, from X-rays out to the radio.
Inference is modular too: the `Fitter` interface borrows optimizers
from `optax`, samplers from `BlackJAX`, and variational inference
from `NIFTy.re`. Gradients are available everywhere, and they are
exact.

Tengri is pre-1.0 and developed as a community effort. The public
API is still moving in places. The repository will move to the
`tengri-project` GitHub organization shortly.

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from
the all-encompassing God of Heaven in traditional Turkic, Mongolic,
and other Central Asian nomadic religions. A fitting name for a code
that models the light of galaxies across cosmic time. This name is
chosen with respect for the cultural and spiritual traditions it
originates from; no religious claim or appropriation is intended.*

---

## At a glance

The forward model is one object. Stellar populations, dust, nebular
emission, AGN, IGM, radio, and X-ray are all registered components
that compose into a single SED chain. There is no separate fast path
for grid fits and slow path for Bayesian inference; both run against
the same code.

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

The registries are introspectable from Python. `summary()` prints the
live count of every component, sampler, recipe, and SSP grid the
running install knows about; `help()` is a curated cheat-sheet with
the most common entry points.

```python
import tengri

tengri.summary()            # live counts across every registry
tengri.help()               # curated cheat-sheet
tengri.list_sfh_models()    # also list_nebular_backends, list_dust_laws, ...
tengri.describe("skirtor")  # paper, parameters, units, citation
tengri.search("torus")      # full-text across the registry
tengri.doctor()             # install + JAX backend + SSP files
```

Nothing in the table updates by hand: a model registered through
`@register_agn_model` shows up in `summary()` without a documentation
edit. The same commands are mirrored on the CLI
(`python -m tengri summary`, …).

## Citation

Tengri does component-level citation. Every physics block, SSP grid,
and inference backend carries its own citation, so the BibTeX for a
fit is assembled from what actually ran:

```python
tengri.print_components_bibtex(result)   # BibTeX for every component that ran
```

This keeps the acknowledgement section of a paper honest as you swap
a dust law or a sampler. The full block (methods paper, SSP grids,
nebular and AGN templates, samplers) lives on the
[Citing tengri](citation) page. While Paper I is in preparation, the
shortest in-text citation is:

> Cooray et al., *tengri: A Differentiable Framework for
> High-Dimensional Bayesian Inference from Galaxy Spectral Energy
> Distributions*, in prep. (2026).

## How this was built

The majority of tengri was developed in roughly two months in close
collaboration between a human author and AI agents, across the
physics modules, the inference layer, and the test suite. Human and
AI agents working together is a deliberate part of the design
philosophy going forward. We are now in the trust-building phase,
verifying every component against established codes and keeping the
development trail open (including an `AGENTS.md` in the repo) so
the trust is earned empirically rather than asserted.

## Get involved

Contributors at every level, from anywhere in the world, are
welcome. Open an issue, send a PR, or write. If a science case you
care about isn't supported yet (a new emission mechanism, a
non-standard observation mode, a sampler from a paper you read last
week), that's exactly the conversation we want to have. The
longer-term ambition is for tengri to be a unifying platform for the
kinds of inference that high-dimensional, modular, differentiable
forward models make possible.

For anything that doesn't fit in an issue (collaborations, the
`tengri-project` org move, joining the project), write to
[astro.tengri@gmail.com](mailto:astro.tengri@gmail.com) or
[cooraysuchetha@gmail.com](mailto:cooraysuchetha@gmail.com).

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
