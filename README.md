# Tengri

[![Tests](https://github.com/suchethac/tengri/actions/workflows/tests.yml/badge.svg)](https://github.com/suchethac/tengri/actions/workflows/tests.yml)
[![Docs](https://github.com/suchethac/tengri/actions/workflows/docs.yml/badge.svg)](https://suchethacooray.com/tengri/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Tengri is a panchromatic galaxy SED inference library, written in
JAX. The same forward model covers stellar populations, dust,
nebular emission, AGN, IGM, radio, and X-ray. Inference is modular too: the `Fitter`
interface borrows optimizers from `optax`, samplers from `BlackJAX`,
and variational inference from `NIFTy.re`, so a new backend lands as
a registration rather than a port. Gradients are available
everywhere, and they are exact.

Tengri is pre-1.0 and developed as a community effort. The API is
still moving in places, and the repository will move to the
`tengri-project` GitHub organization shortly, where collaborative
development and issue tracking will live going forward.

**Documentation:** [suchethacooray.com/tengri](https://suchethacooray.com/tengri/) · **Notebooks:** [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks)

> *The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from
> the all-encompassing God of Heaven in traditional Turkic, Mongolic,
> and other Central Asian nomadic religions. A fitting name for a code
> that models the light of galaxies across cosmic time. This name is
> chosen with respect for the cultural and spiritual traditions it
> originates from; no religious claim or appropriation is intended.*

## Why tengri

Modern galaxy SED inference needs some combination of speed,
differentiability, and modularity at once, and most existing codes
give you one or two of those at a time. Tengri is an attempt at all
three.

JIT compilation gets the full physical model down to tens of
microseconds per call on a single CPU core, which is enough for
catalog-scale inference without putting a neural emulator in the
loop. Exact gradients make HMC, variational inference, and Laplace
approximation work in the $D \gtrsim 100$ parameter spaces where
bursty correlated-field SFHs and hierarchical population fits live.
And the codebase is organized so that physics lives in components and
instruments live in observation, which means a new SFH family, dust
law, or AGN template lands as one file without any edits to the
sampling engine.

The full philosophy and an architecture flow chart are on the
[Overview](https://suchethacooray.com/tengri/overview.html) page.

## How this was built

The majority of tengri was developed in roughly two months in close
collaboration between a human author and AI agents, across the
physics modules, the inference layer, and the test suite. Human and
AI agents working together is a deliberate part of the design
philosophy going forward. We are now in the trust-building phase,
verifying every component against established codes and keeping the
development trail open (including [`AGENTS.md`](AGENTS.md) in the
repo) so the trust is earned empirically rather than asserted. The
per-component status table lives at
[docs/dev/verification-protocol.md](docs/dev/verification-protocol.md);
modules marked PENDING there have not been independently
cross-checked and should not be used for publication-grade science
yet.

## Installation

```bash
pip install "astro-tengri[all]"
```

The PyPI distribution name is `astro-tengri`; the import name is `tengri`. (`pip install tengri` is a different, unrelated 2017 package.) The `[all]` extra pulls in the optimizer and sampler backends (`optax`, `blackjax`) that the quick start below uses; a bare `pip install astro-tengri` can build models and predict but not fit.

For development:

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11, JAX ≥ 0.4.20, DSPS 0.4.6–0.4.7 (0.4.8 excluded: its PyPI sdist breaks at install time), NIFTy 8.5+ with the `re` extra.

**JAX backends:**

- **CPU**: default, no extra setup.
- **CUDA**: `pip install -e ".[gpu]"`, then follow [JAX's CUDA notes](https://jax.readthedocs.io/en/latest/installation.html#gpu-support).
- **Apple Silicon**: `jax-metal` is experimental and produces numerical discrepancies on the stochastic SFH path. Set `JAX_PLATFORMS=cpu` for any fit you intend to trust.

### Verify your install

```bash
pytest tests/components/sps/test_alpha_fe.py tests/components/stellar/test_stellar_skeleton.py -q --no-header
```

The same two-file selection runs as the smoke gate on every PR.

## SSP grids

Tengri needs a pre-computed Simple Stellar Population grid in DSPS HDF5 format. Pre-formatted grids (BC03, BPASS, FSPS, ProGeny) live at the [public mirror](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/). Pull the default FSPS grid:

```python
import tengri
tengri.download_ssp()           # -> data/fsps_prsc_miles_chabrier.h5 (or $TENGRI_DATA_DIR)
tengri.list_known_ssps()        # other grids
```

Or via shell:

```bash
bash scripts/setup_ssp.sh
```

## Quick start

```python
import jax
import tengri
from tengri import (
    SEDModel, Fitter, Fixed, ForwardModel,
    Observation, Photometry, load_ssp_data, recipes,
)

ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

# Pick a curated recipe and let it set sensible priors + defaults.
# The mock is at a known redshift, so fix z — the model then builds a
# single-redshift photometry table in seconds instead of tabulating the
# full z in (0.01, 6) grid. Leave z free (the recipe default) for real
# catalogs with unknown redshifts.
config = recipes.star_forming_photometry()
config["redshift"] = Fixed(0.05)
sed = SEDModel.build(ssp_data=ssp, observation=obs, **config)
forward = ForwardModel.build(sed=sed, observation=obs)

key = jax.random.PRNGKey(0)
mock = sed.mock(sed.spec.sample(key), key=key)

fitter = Fitter(forward, mock.flux_obs, mock.noise)
result = fitter.run("mcmc_nuts")
print(result.summary_table())
```

For real data, pass your own `(flux, noise)` to `Fitter`. The full
walkthrough is in [`notebooks/00_quickstart.py`](notebooks/00_quickstart.py).

If you want more control than a recipe gives you, build the model
with the nested-dict grammar (`SEDModel.build(..., sfh={'type': 'dpl',
'*': FREE, 'beta': Uniform(1, 3)}, dust={...}, neb={...})`). See
[`notebooks/04_building_models.py`](notebooks/04_building_models.py)
for the grammar; `tengri.recipes` shows the curated starting points.

## Tutorials

The notebook spine in [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) is the main learning path. The `.py` files are Jupytext source.

| #  | Notebook                       | Topic                                                       |
|----|--------------------------------|-------------------------------------------------------------|
| 00 | `00_quickstart.py`             | mock galaxy → posterior in ~30 s                            |
| 01 | `01_why_jax.py`                | JIT, `vmap`, `grad` in the context of galaxy SED inference  |
| 02 | `02_sed_anatomy.py`            | the panchromatic SED, component by component                |
| 03 | `03_discovering_the_menu.py`   | discovery API (`list_*`, `describe`, `search`)              |
| 04 | `04_building_models.py`        | the nested-dict / recipe builder                            |
| 05 | `05_fitting_photometry.py`     | photometric fit, end to end                                 |
| 05 | `05_adding_a_model.py`         | registering a new physics block                             |
| 06 | `06_fitting_spectroscopy.py`   | spectroscopy with calibration nuisance parameters           |
| 07 | `07_joint_photo_spec.py`       | joint photo + spec to break degeneracies                    |
| 08 | `08_emission_lines.py`         | BPT diagnostics, line ratios, Hα-based SFR                  |

For single-figure recipes, see the [examples gallery](https://suchethacooray.com/tengri/auto_examples/index.html).

## Discover what's installed

```python
import tengri
tengri.summary()                    # live counts from every registry
tengri.help()                       # curated cheatsheet
tengri.list_filters()               # also list_sfh_models, list_nebular_backends, ...
tengri.list_inference_methods()
tengri.describe("skirtor")
tengri.search("torus")
tengri.doctor()                     # install + JAX backend + SSP files
```

The same commands are available from the shell:

```bash
python -m tengri summary
python -m tengri describe skirtor
python -m tengri doctor
```

These read the registries directly, so a model registered via `@register_agn_model` shows up without a documentation edit.

## Inference backends

`Parameters` declares priors. `SEDModel` ties priors, an SSP grid, and an `Observation` into a differentiable SED chain. `ForwardModel` wraps the chain into a single `.predict(params)` interface that every backend consumes. `Fitter` drives the chosen backend and returns a `Posterior`. The samplers themselves are pulled in from `optax`, `BlackJAX`, and `NIFTy.re`, plus a population layer that runs hierarchical fits on top.

`tengri.list_inference_methods()` returns the live registry, including the experimental backends.

## What's modular

Stellar populations come from DSPS SSPs (BC03, BPASS, FSPS, ProGeny).
The SFH layer covers parametric families (15+, registry-driven),
non-parametric reconstructions (Leja+ continuity, Dirichlet), and
stochastic fields (IFT correlated fields with PSD-governed
burstiness). Dust is swappable on both the attenuation and emission
sides. Nebular emission has four backends (`baked_in`, `cue`,
`cloudy_grid`, `cb19`). AGN spans disc, torus, BLR/NLR, and IR
re-emission, unified across optical, IR, and X-ray. IGM, radio, and
X-ray sit alongside as components, not afterthoughts.

Every physics component is a pure JAX function, so `jit`, `vmap`, and
`grad` compose through the whole forward model. `tengri.cite_all()`
returns BibTeX for every SSP, model, and code used in a fit.

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md): bug reports, feature requests, pull requests
- [GOVERNANCE.md](GOVERNANCE.md): decision-making
- [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)
- [.github/SECURITY.md](.github/SECURITY.md)
- [.github/SUPPORT.md](.github/SUPPORT.md)
- [docs/dev/verification-protocol.md](docs/dev/verification-protocol.md): component verification status
- [CHANGELOG.md](CHANGELOG.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)

Contributors at every level, from anywhere in the world, are
welcome. If a science case you care about isn't supported yet (a
new emission mechanism, a non-standard observation mode, a sampler
from a paper you read last week), that's exactly the conversation we
want to have. The longer-term ambition is for tengri to be a
unifying platform for the kinds of inference that high-dimensional,
modular, differentiable forward models make possible.

For anything that doesn't fit in an issue (collaborations, the
`tengri-project` org move, joining the project), write to
[astro.tengri@gmail.com](mailto:astro.tengri@gmail.com) or
[cooraysuchetha@gmail.com](mailto:cooraysuchetha@gmail.com).

## Physics reproduction

Cross-validation against the established panchromatic SED codes is
cataloged in
[docs/reproduction/](https://suchethacooray.com/tengri/reproduction/index.html).
Five comparisons are live — [CIGALE](https://cigale.lam.fr/), BAGPIPES,
Prospector/FSPS, AGNFITTER-RX, and ProSpect; a sixth (Synthesizer) is
complete and currently unpublished. Each is a notebook that puts the
external code's output and tengri's on the same axes, component by
component, and closes with a full-SED head-to-head with residuals.
Next in line — each with a scoped issue on the tracker — are BEAGLE,
MAGPHYS, GRAHSP, and GalaPy.

## Citation

While Paper I is in preparation, the shortest correct in-text citation is:

> Cooray et al., *tengri: A Differentiable Framework for
> High-Dimensional Bayesian Inference from Galaxy Spectral Energy
> Distributions*, in prep. (2026).

See [CITATION.cff](CITATION.cff) for the machine-readable form and the
[Citing tengri](https://suchethacooray.com/tengri/citation.html) page
for the BibTeX + acknowledgement block. For automatic, fit-specific
BibTeX (every SSP grid, model, and sampler that actually ran):

```python
print(tengri.cite_all(result))
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (which lists every upstream the source code ports from or depends on).
