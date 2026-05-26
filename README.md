# Tengri

[![Tests](https://github.com/suchethac/tengri/actions/workflows/tests.yml/badge.svg)](https://github.com/suchethac/tengri/actions/workflows/tests.yml)
[![Docs](https://github.com/suchethac/tengri/actions/workflows/docs.yml/badge.svg)](https://suchethacooray.com/tengri/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Tengri is a panchromatic galaxy SED inference library, written in
JAX. The same forward model covers stars, dust, nebular gas, AGN, IGM,
radio, and X-ray, and every inference backend we ship (MAP, Laplace,
Pathfinder, NUTS, ray-tracing MCMC, nested sampling, NIFTy geoVI,
hierarchical population fits) talks to it through one `Fitter`
interface. Gradients are available everywhere, and they are exact.

This is pre-1.0 research code, developed as a community effort. The
API is still moving in places, several physics modules are being
independently human-verified, and the Paper I draft is in
preparation. The repository will move to the `tengri-project` GitHub
organisation shortly; collaborative development and issue tracking
will live there going forward.

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
catalogue-scale inference without putting a neural emulator in the
loop. Exact gradients make HMC, variational inference, and Laplace
approximation work in the $D \gtrsim 100$ parameter spaces where
bursty correlated-field SFHs and hierarchical population fits live.
And the codebase is organised so that physics lives in components and
instruments live in observation, which means a new SFH family, dust
law, or AGN template lands as one file without any edits to the
sampling engine.

The full philosophy and an architecture flow chart are on the
[Overview](https://suchethacooray.com/tengri/overview.html) page.

## Status and provenance

This codebase was initially drafted with AI assistance (Claude Code) and
is being human-verified module by module. The per-component status table
lives at [docs/dev/verification-protocol.md](docs/dev/verification-protocol.md);
modules marked PENDING there have not been independently cross-checked and
should not be used for publication-grade science yet.

## Installation

```bash
pip install astro-tengri
```

The PyPI distribution name is `astro-tengri`; the import name is `tengri`. (`pip install tengri` is a different, unrelated 2017 package.)

For development:

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11, JAX ≥ 0.4.20, DSPS 0.4.6 (pinned — 0.4.7 removed `CosmoParams`), NIFTy 8.5+ with the `re` extra.

**JAX backends:**

- **CPU** — default, no extra setup.
- **CUDA** — `pip install -e ".[gpu]"`, then follow [JAX's CUDA notes](https://jax.readthedocs.io/en/latest/installation.html#gpu-support).
- **Apple Silicon** — `jax-metal` is experimental and produces numerical discrepancies on the stochastic SFH path. Set `JAX_PLATFORMS=cpu` for any fit you intend to trust.

### Verify your install

```bash
pytest tests/components/sps/test_alpha_fe.py tests/components/stellar/test_stellar_skeleton.py -q --no-header
```

The same two-file selection runs as the smoke gate on every PR.

## SSP grids

Tengri needs a pre-computed Simple Stellar Population grid in DSPS HDF5 format. Pre-formatted grids (BC03, BPASS, FSPS, ProGeny) live at the [public mirror](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/). Pull the default FSPS grid:

```python
import tengri
tengri.download_ssp()           # -> data/ssp_fsps_v3.2.h5 (or $TENGRI_DATA_DIR)
tengri.list_known_ssps()        # other grids
```

Or via shell:

```bash
bash scripts/setup_ssp.sh
```

## Quick start

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
    sfh_tsnorm_log_total_mass=Uniform(8, 12),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0, 4),
    redshift=0.1,
)
sed = SEDModel(spec, ssp, observation=obs)
forward = ForwardModel.build(sed=sed, observation=obs)

key = jax.random.PRNGKey(0)
mock = sed.mock(spec.sample(key), key=key)

fitter = Fitter(forward, mock["flux_obs"], mock["noise"])
result = fitter.run("mcmc_nuts")
print(result.summary_table())
```

For real data, pass your own `(flux, noise)` to `Fitter`. Full walkthrough in [`notebooks/00_quickstart.py`](notebooks/00_quickstart.py).

The nested-dict / recipe builder (`SEDModel.build(..., sfh={...}, dust={...})`) is the recommended path for new code; the flat-kwarg `Parameters(...)` form shown above still works and is fine for one-off fits. See [`notebooks/04_building_models.py`](notebooks/04_building_models.py) for the grammar.

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

`Parameters` declares priors. `SEDModel` ties priors, an SSP grid, and an `Observation` into a differentiable SED chain. `ForwardModel` wraps the chain into a single `.predict(params)` interface that every backend consumes. `Fitter` drives the chosen backend and returns a `Posterior`.

| Backend       | Command                            | Notes                                                   |
|---------------|------------------------------------|---------------------------------------------------------|
| MAP           | `fitter.run("map")`                | Adam optimization; point estimate / warm-start          |
| Laplace       | `fitter.run("laplace")`            | Gaussian posterior from the Hessian at the MAP          |
| Pathfinder    | `fitter.run("pathfinder")`         | Fast approximate posterior                              |
| NUTS          | `fitter.run("mcmc_nuts")`          | No-U-Turn; primary MCMC for D ≲ 30                      |
| Ray-tracing   | `fitter.run("mcmc_raytrace")`      | Ensemble sampler for higher-D problems                  |
| Nested        | `fitter.run("nss")`                | Nested sampling; produces Bayesian evidence             |
| geoVI         | `fitter.run("vi")`                 | NIFTy geoVI for stochastic-field SFHs (Paper II preview)|
| Population    | `PopulationSEDModel` + `ForwardModel.build(population=...)` | Hierarchical fits with shared parameters across galaxies |

`tengri.list_inference_methods()` shows the full set including experimental backends (MCLMC, ghMC, ESS, etc.).

## What's modular

Stars come from DSPS SSPs (BC03, BPASS, FSPS, ProGeny). The SFH layer
covers parametric families (15+, registry-driven), non-parametric
reconstructions (Leja+ continuity, Dirichlet), and stochastic fields
(IFT correlated fields with PSD-governed burstiness). Dust is swappable
on both the attenuation and emission sides. Nebular emission has four
backends (`baked_in`, `cue`, `cloudy_grid`, `cb19`). AGN spans disc,
torus, BLR/NLR, and IR re-emission, unified across optical, IR, and
X-ray. IGM, radio, and X-ray sit alongside as components, not
afterthoughts.

Every physics component is a pure JAX function, so `jit`, `vmap`, and
`grad` compose through the whole forward model. `tengri.cite_all()`
returns BibTeX for every SSP, model, and code used in a fit.

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md) — bug reports, feature requests, pull requests
- [GOVERNANCE.md](GOVERNANCE.md) — decision-making
- [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)
- [.github/SECURITY.md](.github/SECURITY.md)
- [.github/SUPPORT.md](.github/SUPPORT.md)
- [docs/dev/verification-protocol.md](docs/dev/verification-protocol.md) — component verification status
- [CHANGELOG.md](CHANGELOG.md) · [ROADMAP.md](ROADMAP.md)

## Reproduction

Cross-validation against the established panchromatic SED codes is
catalogued in
[docs/reproduction/](https://suchethacooray.com/tengri/reproduction/index.html).
The first comparison is against [CIGALE](https://cigale.lam.fr/) —
component-for-component agreement on the AGN, dust, nebular, and
X-ray paths — with a side-by-side notebook in the works. Prospector,
BAGPIPES, MAGPHYS, x-cigale, GRAHSP, and Synthesizer follow.

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
