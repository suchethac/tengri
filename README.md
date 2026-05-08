# Tengri

[![Tests](https://github.com/suchethac/tengri/actions/workflows/tests.yml/badge.svg)](https://github.com/suchethac/tengri/actions/workflows/tests.yml)
[![Docs](https://github.com/suchethac/tengri/actions/workflows/docs.yml/badge.svg)](https://suchethacooray.com/tengri/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Tengri is a JAX framework for fitting the spectral energy distributions of galaxies — from the X-ray to the radio — with a single differentiable forward model. Stars, dust, nebular emission, AGN, and IGM components plug into one pipeline, and every inference method (MAP, Laplace, Pathfinder, NUTS, Ray Tracing, Bayesian evidence, hierarchical population, geoVI) runs against the same model with gradients available throughout. Pre-1.0 research code; JIT-compiled and GPU/TPU-native.

**Documentation:** [suchethacooray.com/tengri](https://suchethacooray.com/tengri/) · **Notebooks:** [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) · **Paper:** in preparation

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) is the all-encompassing God of Heaven in traditional Turkic, Mongolic, and Central Asian nomadic religions — the eternal source of order in the natural world. A fitting name for a code that models the light of galaxies across cosmic time. The name is used with respect for the cultural and spiritual traditions it originates from.*

---

## Verification and Provenance

This codebase was initially drafted with AI assistance (Claude Code) and is progressively being human-verified. See [VERIFICATION.md](VERIFICATION.md) for the verification protocol and component status. Physics components marked PENDING there should not be used for publication-grade science without independent cross-validation.

## Installation

```bash
pip install astro-tengri
```

The PyPI distribution name is `astro-tengri`; the import name is `tengri` (`pip install tengri` is a different, unrelated 2017 package).

For development (run tests, build docs, edit code):

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11, JAX ≥ 0.4.20, DSPS ≥ 0.4.6, NIFTy.re ≥ 8.5

JAX backend setup:
- **CPU (default):** works out of the box.
- **CUDA:** follow [JAX CUDA installation](https://jax.readthedocs.io/en/latest/installation.html#gpu-support).
- **Apple Silicon:** Metal backend is enabled by default. For CPU fallback, set `JAX_PLATFORMS=cpu`.

### Verify your install

```bash
.venv/bin/pytest tests/unit/test_alpha_fe.py tests/unit/test_stellar_skeleton.py -q --no-header
```

The same selection runs as a smoke gate on every PR before the full matrix.

## SSP Grids

Tengri needs pre-computed Simple Stellar Population (SSP) grids in DSPS-compatible HDF5 format. A [public mirror of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) (BC03, BPASS, FSPS, ProGeny) is available. Pull the default FSPS grid from Python:

```python
import tengri
tengri.download_ssp()  # FSPS v3.2 → data/ (or $TENGRI_DATA_DIR if set)
```

Or via shell:

```bash
bash scripts/setup_ssp.sh
# or:
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/ssp_fsps_v3.2.h5 -P data/
```

`tengri.list_known_ssps()` shows the other available grids.

## Quick start

```python
import jax
from tengri import (
    SEDModel, Parameters, Fitter,
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
model = SEDModel(spec, ssp, observation=obs)

# Mock recovery. For real data, pass your own (flux, noise) to Fitter.
key = jax.random.PRNGKey(0)
mock = model.mock(spec.sample(key), key=key)

fitter = Fitter(model, mock["flux_obs"], mock["noise"])
result = fitter.run("mcmc_nuts")
print(result.summary_table())
```

Full walkthrough in [`notebooks/00_quickstart.py`](notebooks/00_quickstart.py).

## Tutorials

The tutorial spine in [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) is the primary learning path. The `.py` files use Jupytext format and also render into the [docs site](https://suchethacooray.com/tengri/).

| #  | Notebook                          | What you learn                                                       |
|----|-----------------------------------|----------------------------------------------------------------------|
| 00 | `00_quickstart.py`                | mock galaxy → posterior in ~30 s                                     |
| 01 | `01_why_jax.py`                   | JIT, `vmap`, `grad` — why JAX matters for SED fitting                |
| 02 | `02_sed_anatomy.py`               | wavelength → physics map; what each component does                   |
| 03 | `03_discovering_the_menu.py`      | the registry / discovery API (`list_*`, `describe`, `search`)        |
| 04 | `04_building_models.py`           | compositional model construction with `Parameters`                   |
| 05 | `05_fitting_photometry.py`        | photometric fitting workflow end-to-end                              |
| 06 | `06_fitting_spectroscopy.py`      | optical spectroscopy with calibration nuisance parameters            |
| 07 | `07_joint_photo_spec.py`          | joint photo + spectro fits to break degeneracies                     |
| 08 | `08_emission_lines.py`            | BPT diagnostics, line ratios, Hα-based SFR validation                |

For one-figure recipes, see the [examples gallery](https://suchethacooray.com/tengri/auto_examples/index.html).

## Discover what's there, interactively

After installing, four calls answer "what does tengri do?" and "how do I use it?":

```python
import tengri
tengri.help()           # curated cheatsheet
tengri.summary()        # one-line counts of every menu, live from the registry
tengri.tutorial()       # menu of runnable, copy-pasteable recipes
tengri.<TAB>            # curated entry points (not the kitchen sink)
```

`tengri.tutorial(topic)` prints a recipe; pass `run=True` to execute it where safe:

```python
tengri.tutorial("philosophy")        # layered architecture + IFT framework
tengri.tutorial("first_fit")         # mock galaxy → posterior
tengri.tutorial("swap_inference")    # same model, NUTS → geoVI → MCMC
tengri.tutorial("hierarchical")      # population fit across many galaxies
tengri.tutorial("register_a_model", run=True)   # add a new model — LIVE
```

`tengri.summary()` reads counts directly from the registries — adding a new model via `@register_agn_model` updates them without any doc edit. Each `list_*()` returns a clean table in the REPL and a real HTML table in Jupyter.

`tengri.describe(name)` and `tengri.search(query)` look up models by name or keyword across every menu (components, AGN models, dust laws, SFH models, nebular backends, filters, inference methods).

`tengri.doctor()` confirms install + JAX backend + SSP files.

The same surface is available without entering Python:

```bash
python -m tengri summary
python -m tengri doctor
python -m tengri describe skirtor
python -m tengri search torus
python -m tengri help inference
```

## Inference methods

Parameters declare priors, SSP grids hold pre-computed stellar populations, `SEDModel` ties them into a differentiable forward model, and `Fitter` runs inference and returns a `Posterior`.

| Method        | Command                            | Best for                                                  |
|---------------|------------------------------------|-----------------------------------------------------------|
| MAP           | `fitter.run("map")`                | Point estimates, initialization                           |
| Laplace       | `fitter.run("laplace")`            | Gaussian posterior from Hessian at the MAP                |
| Pathfinder    | `fitter.run("pathfinder")`         | Fast approximate posterior; good NUTS warm-start          |
| NUTS          | `fitter.run("mcmc_nuts")`          | Gold-standard posterior (D ≲ 30)                          |
| Ray Tracing   | `fitter.run("mcmc_raytrace")`      | Exact MCMC, scales beyond D = 30                          |
| Evidence      | `fitter.run("evidence")`           | Bayesian evidence for model comparison                    |
| Population    | `PopulationFitter(...)`            | Shared hyperparameters across galaxy samples              |
| geoVI         | `fitter.run("vi")`                 | High-dim stochastic SFHs (Paper II preview)               |

## What's modular

- **Stars** — DSPS SSPs (BC03, BPASS, FSPS, ProGeny).
- **SFH** — parametric, non-parametric, and stochastic (IFT-correlated fields with PSD-governed burstiness).
- **Dust** — multiple attenuation laws and emission templates, swappable via the registry.
- **Nebular** — BakedIn / CloudyGrid / Cue backends.
- **AGN** — disc + torus + BLR/NLR, unified across optical/IR/X-ray.
- **IGM, radio, X-ray** — first-class components.
- **Filters** — large catalogue of photometric bandpasses (`tengri.list_filters()`).

Each is a swappable pure JAX function. JIT, `vmap`, and `grad` compose throughout.

`tengri.cite_all()` returns BibTeX citations for every upstream SSP grid, paper, and code that contributed to a fit, so reproducibility is not a separate task.

## Status

Pre-1.0; active development; API may change. Paper in preparation.

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md) — bug reports, feature requests, pull requests
- [GOVERNANCE.md](GOVERNANCE.md) — decision-making and core team
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — community standards
- [SECURITY.md](SECURITY.md) — security reporting
- [SUPPORT.md](SUPPORT.md) — getting help
- [VERIFICATION.md](VERIFICATION.md) — component verification status
- [CHANGELOG.md](CHANGELOG.md) — version history
- [ROADMAP.md](ROADMAP.md) — planned features

## Citation

If you use tengri, please see [CITATION.cff](CITATION.cff) and call `tengri.cite_all()` to get citations for every upstream grid, paper, and code that contributed to your fit.

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
