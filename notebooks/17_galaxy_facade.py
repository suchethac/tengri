# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Galaxy Facade, Presets, and Citations — the one-liner API
#
# This notebook documents the high-level API added in tengri 0.1.0:
#
# - `tengri.Galaxy.from_arrays(...)` — build a fittable galaxy in one call
# - `tengri.presets` — vetted starting configurations for common galaxy types
# - `tengri.cite(...)` / `tengri.cite_all()` — upstream-code credit as a first-class API
# - `tengri.FitResult` + `tengri.Provenance` — reproducible run metadata
# - `tengri.preprocessing` — survey zero-points, systematic floors, upper limits
# - `tengri.io` — SDSS, DESI, generic FITS, and `specutils.Spectrum1D` adapter
# - `tengri.doctor()` — environment health check
#
# This is the "entry point" for users who want a fit without constructing SEDModel + Parameters + Observation + Fitter by hand. Everything here composes with the lower-level API — nothing is hidden; the facade just orchestrates.

# %% [markdown]
# **Spine location:** `notebooks/17_galaxy_facade.py` (not `notebook_code/`).

# %%
import os
import sys
import time
import warnings

# Must be set before JAX initializes its XLA backend (first computation, not import).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
except NameError:
    _nb_dir = os.getcwd()
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))

_src = os.path.join(_repo_root, "src")
if os.path.isdir(os.path.join(_src, "tengri")):
    sys.path.insert(0, _src)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _nb_dir)

import jax
import jax.numpy as jnp
import matplotlib

# Use non-interactive backend when run as a plain script (not in Jupyter).
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

# Locate ``notebooks/_plot_style.py`` and ``data/`` root (nbclient cwd is often wrong).
import importlib.util

_repo_data_root = None
_spec_tengri = importlib.util.find_spec("tengri")
if _spec_tengri is not None and _spec_tengri.origin:
    _walk = os.path.dirname(os.path.abspath(_spec_tengri.origin))
    for _step in range(12):
        _candidate = os.path.join(_walk, "notebooks", "_plot_style.py")
        if os.path.isfile(_candidate):
            sys.path.insert(0, os.path.dirname(_candidate))
            _repo_data_root = os.path.dirname(os.path.dirname(os.path.abspath(_candidate)))
            break
        _parent_walk = os.path.dirname(_walk)
        if _parent_walk == _walk:
            break
        _walk = _parent_walk

if _repo_data_root is None:
    _np_here = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(_np_here, "_plot_style.py")):
            sys.path.insert(0, _np_here)
            _repo_data_root = os.path.dirname(_np_here)
            break
        _ppt = os.path.join(_np_here, "notebooks", "_plot_style.py")
        if os.path.isfile(_ppt):
            _nbsd = os.path.dirname(_ppt)
            sys.path.insert(0, _nbsd)
            _repo_data_root = os.path.dirname(_nbsd)
            break
        _parent_here = os.path.dirname(_np_here)
        if _parent_here == _np_here:
            break
        _np_here = _parent_here

if _repo_data_root is not None and os.path.isdir(os.path.join(_repo_data_root, "data")):
    os.chdir(_repo_data_root)
elif os.path.isdir(os.path.join(_repo_root, "data")):
    os.chdir(_repo_root)
elif os.path.isdir("data"):
    pass
elif os.path.isdir(os.path.join("..", "data")):
    os.chdir("..")

# %% [markdown]
# ## Logo and version

# %%
import tengri as tg

tg.print_logo()
print(f"tengri {tg.__version__}")

# %% [markdown]
# ## Environment health check
#
# Run this first if anything surprises you.

# %%
tg.doctor()

# %% [markdown]
# ## Presets
#
# **Presets** bundle Parameters + ModelConfig for common galaxy types. Call them to see what you get.

# %%
for name in tg.presets.list_presets():
    print(f"--- {name} ---")
    print(tg.presets.describe(name))
    print()

# %% [markdown]
# ## Galaxy.from_arrays — the one-liner entry point
#
# **Galaxy.from_arrays** — the one-liner entry point. Provide filter names, fluxes, errors, redshift, SSP path, and a preset name. Flux units are auto-converted.

# %%
import glob

# Look for any available SSP file; skip gracefully if none is present.
_ssp_candidates = (
    glob.glob("data/ssp_*.h5")
    + glob.glob("../data/ssp_*.h5")
    + ([os.environ["TENGRI_SSP_PATH"]] if os.environ.get("TENGRI_SSP_PATH") else [])
)
_ssp_candidates = [p for p in _ssp_candidates if os.path.exists(p)]

if _ssp_candidates:
    ssp_path = _ssp_candidates[0]
    print(f"Using SSP: {ssp_path}")
    g = tg.Galaxy.from_arrays(
        filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        flux=np.array([1e-28, 2e-28, 3e-28, 2.5e-28, 2e-28]),
        flux_err=np.array([1e-29] * 5),
        flux_unit="erg/s/cm2/Hz",
        redshift=0.1,
        ssp_path=ssp_path,
        preset="starforming",
    )
    print(g.explain())
else:
    print("No SSP file found — set TENGRI_SSP_PATH or drop one under data/.")
    g = None

# %% [markdown]
# ## Citations for this specific galaxy
#
# **cite()** returns citations for the components this specific galaxy uses — stellar synthesis, dust law, nebular model, and inference backend if a fit has been run.

# %%
if g is not None:
    for c in g.cite():
        print(" -", c)
else:
    print("Run the previous cell with SSP data to see configuration-specific citations.")
    # Fallback: show the full registry
    print("\nFull citation registry:")
    for c in tg.cite_all()[:5]:
        print(" -", c)

# %% [markdown]
# ## BibTeX lookup
#
# Single-entry BibTeX lookup — paste straight into your .bib.

# %%
print(tg.cite("calzetti2000").to_bibtex())

# %% [markdown]
# ## Preprocessing utilities
#
# **Preprocessing utilities** — survey zero-point corrections, systematic error floors, upper-limit detection. Offsets are placeholders pending human verification; see VERIFICATION.md.

# %%
from tengri import preprocessing as pp

entries = pp.lookup_zeropoints("JADES", "DR5", ["F150W", "F200W"])
for e in entries:
    print(e)

flux = np.array([1e-28, 2e-28, 3e-28])
err = np.array([1e-29, 1e-29, 1e-29])
flux_c, err_c = pp.apply_zeropoints(flux, err, [entries[0]] * 3)
print("flux corrected:", flux_c)
print("err  corrected:", err_c)

# Upper limits: anything with S/N < 1 treated as non-detection
mask = pp.detect_upper_limits(
    np.array([5, 0.5, 0.3]), np.array([1, 1, 1]), sn_threshold=1.0
)
print("upper-limit mask:", mask)

# %% [markdown]
# ## Filter helpers
#
# **Filter helpers** — discover what the library contains and which filters cover a target redshift.

# %%
print("JWST filters in library:", tg.filters.list_filters(instrument="JWST")[:6])
print(
    "At z=0.5, visible-to-NIR coverage:",
    tg.filters.suggest(redshift=0.5, coverage="visible_to_nir")[:8],
)

# %% [markdown]
# ## FitResult and save/load
#
# **FitResult** is a thin provenance+citations wrapper that serialises to HDF5. Below: a mock result used only to demonstrate save/load. After a real `.fit()` you would call `g.save("my_fit.h5")`.

# %%
from tengri import FitResult, Provenance
import tempfile

fr = FitResult(
    inner={"samples": {"log_mstar": [9.5, 9.7, 9.6]}},
    provenance=Provenance.capture(wall_time_seconds=0.5),
    citation_keys=["tengri", "dsps", "calzetti2000", "jax"],
    backend="map",
    preset="starforming",
)
print(fr.summary())
print()
print(
    f"tengri={fr.provenance.tengri_version}  "
    f"python={fr.provenance.python_version}  "
    f"jax={fr.provenance.jax_version}"
)

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "demo_fit.h5")
    fr.save(path)
    fr2 = FitResult.load(path)
    print("roundtrip ok:", fr2.backend, fr2.preset, fr2.citation_keys)

# %% [markdown]
# ## CLI hint
#
# **CLI** — the same primitives are available outside Python:
#
# ```bash
# $ tengri doctor
# $ tengri cite calzetti2000 --bibtex
# $ tengri cite              # lists everything
# ```

# %% [markdown]
# ## Where to go next
#
# - `notebooks/07_fitting_photometry.py` — full photometry fit with uncertainty
# - `notebooks/08_fitting_spectra.py` — spectroscopic fits
# - `notebooks/14_joint_photometry_spectroscopy.py` — joint phot+spec
# - `docs/recipes/sdss_photometry_oneliner.md` — 80-line recipe
# - `VERIFICATION.md` — physics verification status (every row currently PENDING)
#
# Everything here composes with the low-level API — nothing is hidden. The facade just orchestrates the same SEDModel / Parameters / Observation / Fitter objects used in the other notebooks.
