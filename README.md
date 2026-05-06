# Tengri

[![Tests](https://github.com/suchethac/tengri/actions/workflows/tests.yml/badge.svg)](https://github.com/suchethac/tengri/actions/workflows/tests.yml)
[![Docs](https://github.com/suchethac/tengri/actions/workflows/docs.yml/badge.svg)](https://suchethac.github.io/tengri/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Tengri is a JAX framework for differentiable galaxy SED fitting. One modular forward model spans stars, dust, nebular emission, AGN, and IGM — X-ray to radio. Every inference method (MAP, Laplace, Pathfinder, NUTS, Ray Tracing, Bayesian evidence, hierarchical population) runs on the same model, with gradients available everywhere. Pre-1.0 research code; JIT-compiled and GPU/TPU-native. Information-Field-Theory stochastic SFH priors and geoVI are Paper II preview material.

**Documentation:** [https://suchethacooray.github.io/tengri/](https://suchethacooray.github.io/tengri/) · **Notebooks:** [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) · **Paper:** [In preparation]

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from the all-encompassing God of Heaven in traditional Turkic, Mongolic, and other Central Asian nomadic religions. Tengri is the supreme sky deity in Tengrism, the eternal source of order in the natural world. A fitting name for a code that models the light of galaxies across cosmic time. This name is chosen with respect for the cultural and spiritual traditions it originates from; no religious claim or appropriation is intended.*

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

**Requirements:** Python >= 3.11, JAX >= 0.4.20, DSPS >= 0.4.6, NIFTy.re >= 8.5

JAX backend setup:
- **CPU (default):** Works out-of-the-box.
- **CUDA:** Follow [JAX CUDA installation](https://jax.readthedocs.io/en/latest/installation.html#gpu-support).
- **Apple Silicon:** Metal backend enabled by default via `jax_platforms=metal`. For CPU fallback, use `JAX_PLATFORMS=cpu`.

### Verify your install (30 s)

```bash
.venv/bin/pytest tests/unit/test_alpha_fe.py tests/unit/test_stellar_skeleton.py -q --no-header
```

The same selection runs on every PR via GitHub Actions before the full matrix.

## SSP Grids

Tengri requires pre-computed Simple Stellar Population (SSP) grids in DSPS-compatible HDF5 format. A [repository of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) from BC03, BPASS, FSPS, and ProGeny is publicly available.

```bash
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/ssp_fsps_v3.2.h5 -P data/
```

## Quick Start

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
result = fitter.run("mcmc_nuts")   # or "map", "laplace", "pathfinder", "mcmc_raytrace"
print(result.summary_table())
```

Full walkthrough in [`notebooks/00_quickstart.py`](notebooks/00_quickstart.py).

## Discover what's available

After installing, the first three calls to make:

```python
import tengri
tengri.help()           # curated cheatsheet
tengri.summary()        # one-line counts of every menu
tengri.<TAB>            # ~30 curated entry points (not the 175-name kitchen sink)
```

`tengri.summary()` prints something like:

```
tengri — what's available:

     7  physics components           tengri.list_components()
    12  AGN models                   tengri.list_agn_models()
    21  dust attenuation laws        tengri.list_dust_laws()
     7  dust emission templates      tengri.list_dust_emission_models()
    34  SFH models                   tengri.list_sfh_models()
     4  nebular backends             tengri.list_nebular_backends()
     6  primary inference methods    tengri.list_inference_methods(tier='primary')
    19  total inference methods      tengri.list_inference_methods()
```

Each `list_*()` returns a column-aligned table — no `pprint` needed:

```python
>>> tengri.list_inference_methods(tier="primary")
name               tier     short_doc
─────────────────  ───────  ──────────────────────────────────────────────
map                primary  Adam MAP optimization
mcmc_nuts          primary  No-U-Turn Sampler
mcmc_raytrace      primary  Ray-tracing ensemble sampler (high-D)
vi                 primary  NIFTy geoVI variational inference
...
```

Universal lookup across every menu:

```python
tengri.describe("skirtor")        # any AGN model, dust law, SFH variant,
tengri.describe("mcmc_nuts")      # nebular backend, inference method,
tengri.describe("dl07")           # or component
```

In Jupyter the same calls render as real HTML tables.

## Features

- **JIT-compiled, fully differentiable:** pure JAX end-to-end. Forward model ~140 μs, gradient ~56 μs on CPU for a smooth 7-D model. JIT + `vmap` + `grad` compose — one forward model powers every inference backend.
- **Modular physics:** stars (DSPS SSPs), SFH (parametric, non-parametric, stochastic IFT), dust attenuation (15+ laws) and emission, nebular (BakedIn / CloudyGrid / Cue), unified AGN (disc + torus + BLR/NLR), IGM absorption, radio and X-ray. Each is a swappable pure function.
- **Every inference method, same model:** `fitter.run("map" | "laplace" | "pathfinder" | "mcmc_nuts" | "mcmc_raytrace" | "evidence")`; `PopulationFitter` for hierarchical fits across catalogues.
- **Per-component citations:** `tengri.cite_all()` returns citations for every upstream SSP grid, paper, and code contributing to your fit.
- **Survey data readers:** SDSS/DESI/generic FITS readers; specutils bridge for flexible spectroscopy input.
- **CLI utilities:** `tengri doctor` (dependency check), `tengri cite KEY` (targeted citations), preprocessing helpers.
- **GPU/TPU native:** same code runs on CPU, GPU, TPU without modification.

## How It Works

Parameters declare free parameters and priors (SFH shape, dust, metallicity, redshift). SSP grids hold pre-computed stellar population spectra (any DSPS-compatible HDF5). Model combines them into a differentiable forward model that maps physical parameters to predicted photometry or spectra. Fitter runs inference (MAP, VI, MCMC) and returns a Posterior with samples, diagnostics, and provenance.

| Inference Method | Command | Best For |
|------------------|---------|----------|
| MAP | `fitter.run("map")` | Point estimates, initialization |
| Laplace | `fitter.run("laplace")` | Gaussian posterior from Hessian at MAP |
| Pathfinder | `fitter.run("pathfinder")` | Fast approximate posterior; good NUTS warm-start |
| NUTS | `fitter.run("mcmc_nuts")` | Gold-standard posterior (D ≲ 30) |
| Ray Tracing | `fitter.run("mcmc_raytrace")` | Exact MCMC, noise-robust, scales past D = 30 |
| Evidence (NSS) | `fitter.run("evidence")` | Bayesian evidence for model comparison |
| Population | `PopulationFitter(...)` | Shared hyperparameters across galaxy samples |
| geoVI / vi_native | `fitter.run("vi")` / `"vi_native"` | **Paper II preview.** High-D stochastic SFHs (D ≈ 137+) |

## Status

Pre-1.0, active development. Core pipeline fully functional with 2000+ unit and integration tests. Paper in preparation. API may change.

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md) — Bug reports, feature requests, pull requests.
- [GOVERNANCE.md](GOVERNANCE.md) — Decision-making and core team.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Community standards.
- [SECURITY.md](SECURITY.md) — Security reporting.
- [SUPPORT.md](SUPPORT.md) — Getting help.
- [VERIFICATION.md](VERIFICATION.md) — Component verification status.
- [CHANGELOG.md](CHANGELOG.md) — Version history.
- [ROADMAP.md](ROADMAP.md) — Planned features.

## Citation

If you use tengri, please see [CITATION.cff](CITATION.cff) and call `tengri.cite_all()` to get citations for every upstream grid, paper, and code that contributed to your fit.

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
