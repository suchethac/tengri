# Documentation

**[Notebook list (00–14) ↓](#spine-notebooks)** — Jupytext sources live in
[`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) on GitHub.

The rest of this page is setup (install, SSP data), then the spine table and a short API example.

## Installation

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e .
```

Core dependencies: JAX, DSPS, NIFTy.re, NumPy, matplotlib, h5py.

**Optional:**

```bash
pip install -e ".[all]"       # + BlackJAX (NUTS), optax (MAP)
pip install -e ".[nuts]"      # BlackJAX only
pip install -e ".[optax]"     # optax only
pip install -e ".[dev]"       # pytest, ruff, jupytext
```

**Requirements:** Python ≥ 3.10, JAX ≥ 0.4.20, DSPS ≥ 0.3, NIFTy.re ≥ 8.5 (see `pyproject.toml` for full ranges).

**Platforms:** On macOS, JAX Metal can be flaky — use `JAX_PLATFORMS=cpu` for reliable runs. For CUDA, install the matching `jax[cuda12]` (or current) wheel. The package enables `jax_enable_x64` on import.

## SSP grids

tengri needs DSPS-compatible SSP templates in HDF5. A [hosted catalog](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/) (BC03, BPASS, FSPS, ProGeny) is available. Example:

```bash
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/ssp_fsps_v3.2.h5 -P data/
```

## Spine notebooks

Sources of truth are **Jupytext** percent-format `.py` files under `notebooks/` at the repo root. To refresh the copies used by this site:

```bash
python scripts/sync_spine_notebooks_for_docs.py
# or: make -C docs spine-ipynb
```

That runs **`jupytext --sync`** on each spine pair, copies the `.ipynb` into `docs/spine/`, and runs **`jupyter nbconvert --to notebook --inplace`** on each copy (format check; Sphinx uses nbconvert again via **nbsphinx** when building HTML). The built docs **do not execute** notebooks (`nbsphinx_execute = "never"` in `conf.py`).

**Suggested order:** 00 → 01 → 02 → 13 → 03–06 → 07–12; use **14** after **08** for joint photometry and spectroscopy.

Tutorial pages in the sidebar are those **`.ipynb`** files rendered as HTML. Optional standalone exports: `make -C docs export-notebooks-html` (writes `_build/html/nbconvert/`).

## Quick start

```python
from tengri import Model, Parameters, Fitter, Uniform, Gaussian
from tengri import Observation, Photometry, load_ssp_data

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0, 4),
    redshift=0.1,
)

model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, obs_flux, obs_noise)
result = fitter.run("vi")
print(result.summary_table())
```

API reference: [API](api/index.md).

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   spine/00_quickstart
   spine/01_why_jax
   spine/02_sed_anatomy
   spine/03_fitting_photometry
   spine/04_fitting_spectra
   spine/05_joint_photometry_spectroscopy
   spine/06_inference_methods
   spine/07_degeneracies
   spine/08_sfh_advanced
   spine/09_dust_emission
   spine/10_agn_advanced
   spine/11_population
   spine/12_diagnostics
   spine/13_extending_tengri
   spine/14_stochastic_sfh
   spine/15_vi_inference
   spine/16_simulation_interface
   spine/17_emission_line_measurements
```
