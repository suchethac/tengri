# Tengri

Tengri is a differentiable JAX framework for SED fitting with Information Field Theory correlated-field star formation history priors. Pre-1.0 research code. Fully JIT-compiled, GPU-native, with gradients enabling modern inference backends (geoVI, Ray Tracing, NUTS, hierarchical population fitting).

**Documentation:** [https://suchethacooray.github.io/tengri/](https://suchethacooray.github.io/tengri/) · **Notebooks:** [`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) · **Paper:** [In preparation]

---

*The name [Tengri](https://en.wikipedia.org/wiki/Tengri) comes from the all-encompassing God of Heaven in traditional Turkic, Mongolic, and other Central Asian nomadic religions. Tengri is the supreme sky deity in Tengrism, the eternal source of order in the natural world. A fitting name for a code that models the light of galaxies across cosmic time. This name is chosen with respect for the cultural and spiritual traditions it originates from; no religious claim or appropriation is intended.*

---

## Verification and Provenance

This codebase was initially drafted with AI assistance (Claude Code) and is progressively being human-verified. See [VERIFICATION.md](VERIFICATION.md) for the verification protocol and component status. Physics components marked PENDING there should not be used for publication-grade science without independent cross-validation.

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, JAX >= 0.4.20, DSPS >= 0.3, NIFTy.re >= 8.5

JAX backend setup:
- **CPU (default):** Works out-of-the-box.
- **CUDA:** Follow [JAX CUDA installation](https://jax.readthedocs.io/en/latest/installation.html#gpu-support).
- **Apple Silicon:** Metal backend enabled by default via `jax_platforms=metal`. For CPU fallback, use `JAX_PLATFORMS=cpu`.

## SSP Grids

Tengri requires pre-computed Simple Stellar Population (SSP) grids in DSPS-compatible HDF5 format. A [repository of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) from BC03, BPASS, FSPS, and ProGeny is publicly available.

```bash
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/ssp_fsps_v3.2.h5 -P data/
```

## Quick Start

```python
import tengri as tg

g = tg.Galaxy.from_arrays(
    filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    flux=[1e-28, 2e-28, 3e-28, 2.5e-28, 2e-28],
    flux_err=[1e-29]*5,
    flux_unit="erg/s/cm2/Hz",
    redshift=0.1,
    ssp_path="data/ssp_fsps_v3.2.h5",
    preset="starforming",
)

g.fit(backend="map")
print(g.summary())
print(g.cite())
```

## Features

- **Galaxy facade:** One-liner setup with sensible presets (`starforming`, `quiescent`, `high_z`, `photoz`, `jwst_spec`, `agn_host`).
- **Per-component citations:** `tengri.cite_all()` returns citations for every upstream SSP grid, paper, and code contributing to your fit.
- **FitResult with provenance:** Posterior samples, summary statistics, convergence diagnostics, and full parameter/forward-model history.
- **Survey data readers:** SDSS/DESI/generic FITS readers; specutils bridge for flexible spectroscopy input.
- **CLI utilities:** `tengri doctor` (dependency check), `tengri cite KEY` (targeted citations), preprocessing helpers.
- **Systematic utilities:** Zero-point registry, systematic floor, upper limit handling.

## How It Works

Parameters declare free parameters and priors (SFH shape, dust, metallicity, redshift). SSP grids hold pre-computed stellar population spectra (any DSPS-compatible HDF5). Model combines them into a differentiable forward model that maps physical parameters to predicted photometry or spectra. Fitter runs inference (MAP, VI, MCMC) and returns a Posterior with samples, diagnostics, and provenance.

| Inference Method | Command | Best For |
|------------------|---------|----------|
| MAP | `fitter.run("map")` | Point estimates |
| geoVI | `fitter.run("vi")` | **Default.** Nonlinear posteriors via NIFTy |
| Ray Tracing | `fitter.run("mcmc_raytrace")` | Exact MCMC, stochastic-gradient robust |
| NUTS | `fitter.run("mcmc_nuts")` | Gold-standard (D ≲ 30) |
| Laplace | `fitter.run("laplace")` | Instant Gaussian posterior from Hessian |
| Pathfinder | `fitter.run("pathfinder")` | Fast approximate posterior |
| Evidence | `fitter.run("evidence")` | Bayesian evidence (D ≲ 30) |
| Population | `model.fit_population(observations)` | Hierarchical fits with shared PSD |

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
