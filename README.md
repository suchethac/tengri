# Tengri

[![Tests](https://github.com/suchethac/tengri/actions/workflows/tests.yml/badge.svg)](https://github.com/suchethac/tengri/actions/workflows/tests.yml)
[![Docs](https://github.com/suchethac/tengri/actions/workflows/docs.yml/badge.svg)](https://suchethacooray.com/tengri/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Tengri is a panchromatic galaxy SED inference library, written in
JAX. The same forward model covers stellar populations, dust,
nebular emission, AGN, IGM, radio, and X-ray. Inference is modular
too: the `Fitter` interface borrows optimizers from `optax`, samplers
from `BlackJAX`, and variational inference from `NIFTy.re`, so new
fitting methods plug in without touching the physics. Gradients are
available everywhere, and they are exact.

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

Modern galaxy SED inference needs speed, differentiability, and
modularity at once; most codes give you one or two. Tengri is an
attempt at all three.

JIT compilation gets the core forward model down to tens of
microseconds per galaxy in batched (`vmap`) evaluation on a single
CPU core, fast enough for catalog-scale inference without putting a
neural emulator in the loop. (A full panchromatic model with dust IR
re-emission and nebular emission is heavier: of order a millisecond
per galaxy.) Exact gradients make HMC, variational inference, and Laplace
approximation work in the 100+ parameter spaces where bursty
star formation histories and hierarchical population fits live. And
the physics and the instrument models are separate, swappable pieces,
so a new SFH family, dust law, or AGN template is one new file, with
no changes to the sampling machinery.

The full philosophy and an architecture flow chart are on the
[Overview](https://suchethacooray.com/tengri/overview.html) page.

## How this was built

Most of tengri was built in about six months by a human author
working closely with AI agents. That is a deliberate part of the
design philosophy, and the development trail is kept open (see
[`docs/dev/agents.md`](docs/dev/agents.md)). Trust has to be earned the usual way:
every piece gets checked against established codes, and the status
of each one is tracked at
[docs/dev/verification-protocol.md](docs/dev/verification-protocol.md).
Modules marked PENDING there have not been cross-checked yet, so
don't use them for publication-grade science.

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

The smoke gate uses `pytest`, which ships in the `[dev]` extra (it is **not**
included in `[all]`), so install that first:

```bash
pip install -e ".[dev]"
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

# download_ssp() fetches the default bare-stellar grid on first run and
# skips on subsequent runs. See "SSP grids" above for other grids or shell setup.
ssp = load_ssp_data(tengri.download_ssp())
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

# The mock is at a known redshift, so fix z. The model then builds a
# single-redshift table in seconds instead of tabulating the
# full z in (0.01, 6) grid.
# Leave z free for real catalogs with unknown redshifts.
config = recipes.star_forming_photometry()
config["redshift"] = Fixed(0.05)
sed = SEDModel.build(ssp_data=ssp, observation=obs, **config)
forward = ForwardModel.build(sed=sed, observation=obs)

key = jax.random.PRNGKey(0)
mock = sed.mock(sed.spec.sample(key), key=key)

# This recipe has 8 free parameters. Past D ~ 6, NUTS spends most of its
# time in warmup, so use fixed-length HMC. See
# docs/method_selection.md for the decision table.
fitter = Fitter(forward, mock.flux_obs, mock.noise)
result = fitter.run("mcmc_hmc")
print(result.summary_table())
```

Building the model takes a few seconds and the fit about 40 s on a laptop
CPU; the first run also pays a one-off JAX compile. Swap in `"map"` for a
point estimate in ~4 s, or `"laplace"` for credible intervals in ~5 s.
`"mcmc_nuts"` is the gold standard below D ≈ 6, but on this 8-parameter
model its warmup pushes the fit past 8 minutes.

For real data, pass your own `(flux, noise)` to `Fitter`. The full
walkthrough is in [`notebooks/00_quickstart.py`](notebooks/00_quickstart.py).

If you want more control than a recipe gives you, build the model with
the nested-dict grammar. Import the sentinels and priors it uses first
(`FREE`/`FIXED` are singletons, distinct from the `Fixed(...)` prior
above):

```python
from tengri import FREE, FIXED, Uniform

sed = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'dpl', 'all_params': FREE, 'beta': Uniform(1, 3)},
    dust={'type': 'two_component', 'all_params': FIXED},
    neb={'type': 'cue', 'all_params': FIXED},
)
```

`all_params` sets every parameter in the group at once; per-parameter keys
(like `beta` above) override it. `'*'` is an accepted synonym.

See [`notebooks/04_building_models.py`](notebooks/04_building_models.py)
for the grammar; `tengri.recipes` shows the curated starting points.

## Tutorials

The notebook spine in [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) is the main learning path. The `.py` files are Jupytext source.

| #  | Notebook                       | Topic                                                       |
|----|--------------------------------|-------------------------------------------------------------|
| 00 | `00_quickstart.py`             | mock galaxy → posterior, end to end                         |
| 01 | `01_why_jax.py`                | JIT, `vmap`, `grad` in the context of galaxy SED inference  |
| 02 | `02_sed_anatomy.py`            | the panchromatic SED, component by component                |
| 03 | `03_discovering_the_menu.py`   | discovery API (`list_*`, `describe`, `search`)              |
| 04 | `04_building_models.py`        | the nested-dict / recipe builder                            |
| 05 | `05_fitting_photometry.py`     | photometric fit, end to end                                 |
| 05 | `05_adding_a_model.py`         | registering a new physics block                             |
| 06 | `06_fitting_spectroscopy.py`   | spectroscopy with calibration nuisance parameters           |
| 07 | `07_joint_photo_spec.py`       | joint photo + spec to break degeneracies                    |
| 08 | `08_emission_lines.py`         | BPT diagnostics, line ratios, Hα-based SFR                  |
| 09 | `09_parameter_sweeps.py`       | one knob at a time: how each parameter moves the SED        |
| 10 | `10_fastspecfit_joint_fit.py`  | joint DESI photometry + emission-line fluxes, timed        |
| 11 | `11_catalog_fits.py`           | a catalog fit in parallel: LSST+Euclid photo-z, timed      |
| 12 | `12_simulation_populations.py` | simulation SFH + Z(t) → photometry and lines, ~8k galaxies/s |
| 19 | `19_joint_spec_phot.py`        | joint spec-phot with a Sersic fiber aperture                |

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

Stellar populations are integrated over SSP grids in the DSPS format
(BC03, BPASS, FSPS, ProGeny) by tengri's own differentiable CIC kernel;
DSPS's histogram kernel is selectable for cross-code parity.
The SFH layer covers parametric families (15+, registry-driven),
non-parametric reconstructions (Leja+ continuity, Dirichlet), and
stochastic fields (IFT correlated fields with PSD-governed
burstiness). Dust is swappable on both the attenuation and emission
sides. Nebular emission has four backends: `ssp` (emission baked into
the SSP grid), `cue` (emulator), `cloudy`, and `cb19`. AGN spans disc,
torus, BLR/NLR, and IR
re-emission, unified across optical, IR, and X-ray. IGM, radio, and
X-ray sit alongside as components, not afterthoughts.

Every physics component is a pure JAX function, so `jit`, `vmap`, and
`grad` compose through the whole forward model.
`tengri.print_components_bibtex(result)` prints BibTeX for every SSP,
model, and code used in a fit.

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md): bug reports, feature requests, pull requests
- [GOVERNANCE.md](GOVERNANCE.md): decision-making
- [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)
- [.github/SECURITY.md](.github/SECURITY.md)
- [.github/SUPPORT.md](.github/SUPPORT.md)
- [docs/dev/verification-protocol.md](docs/dev/verification-protocol.md): component verification status
- [CHANGELOG.md](CHANGELOG.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)

Contributors at every level are welcome. If a science case you care
about isn't supported yet (a new emission mechanism, an unusual
observation mode, a sampler from a paper you read last week), that's
exactly the conversation we want to have. The longer-term goal is for
tengri to be a shared platform for the kinds of inference that
high-dimensional, differentiable forward models make possible.

For anything that doesn't fit in an issue (collaborations, the
`tengri-project` org move, joining the project), write to
[astro.tengri@gmail.com](mailto:astro.tengri@gmail.com) or
[cooraysuchetha@gmail.com](mailto:cooraysuchetha@gmail.com).

## Physics reproduction

tengri reproduces the physics in the established panchromatic SED
codes: the same models, implemented in one framework, so you can
check every assumption and compare models side by side. The
comparisons live in
[docs/reproduction/](https://suchethacooray.com/tengri/reproduction/index.html).
Five are online now ([CIGALE](https://cigale.lam.fr/), BAGPIPES,
Prospector/FSPS, AGNFITTER-RX, ProSpect), each putting the two codes
on the same axes, component by component, ending with a full-SED
head-to-head. A sixth (Synthesizer) is being revised, and BEAGLE,
MAGPHYS, GRAHSP, and GalaPy are next.

## Citation

While Paper I is in preparation, the shortest correct in-text citation is:

> Cooray et al., *tengri: A Differentiable Framework for
> High-Dimensional Bayesian Inference from Galaxy Spectral Energy
> Distributions*, in prep. (2026).

See [CITATION.cff](CITATION.cff) for the machine-readable form and the
[Citing tengri](https://suchethacooray.com/tengri/citation.html) page
for the BibTeX + acknowledgement block. For automatic, fit-specific
citations (every SSP grid, model, and sampler that actually ran),
pass the fit to `cite`, which prints the component table and BibTeX:

```python
tengri.cite(result)
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (which lists every project whose models tengri implements or depends on).
