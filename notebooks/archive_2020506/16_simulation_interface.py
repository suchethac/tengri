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
# # Paper II Preview: Simulation-to-Observation Interface
#
# Forward-model SEDs from simulation time series (SFH, metallicity) through full physics.
#
# ## What you'll learn
#
# - **Tabulated SFH input** — discrete cosmic-time SFR(t) from IllustrisTNG, EAGLE, FIRE, SAMs, UniverseMachine
# - **Metallicity histories** — Z(t) from simulations drives nebular/AGN physics
# - **Full forward pipeline** — stellar → dust attenuation → nebular → AGN → IGM absorption
# - **Batch simulation processing** — evaluate 100+ snapshot SEDs, compare to observations
# - **Bridge to fitting** — reverse process: observe a galaxy, fit simulation-like outputs
#
# ## Prerequisites
#
# [`00_quickstart.py`](00_quickstart.py) (photometry basics) and
# [`02_sed_anatomy.py`](02_sed_anatomy.py) (physics module overview).
#
# **Paper II advanced preview:** Simulation-comparison workflows; optional for first-time users.


# %%
import os
import sys
import warnings

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

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

from tengri import (
    load_ssp_data,
    load_filter_set,
)
from tengri.analysis.simulate import (
    sed_from_sfh,
    photometry_from_sfh,
    spectrum_from_sfh,
)

# Locate plot style
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

try:
    from _plot_style import setup_style, COLORS as _COLORS_DICT

    setup_style()
    # The shared COLORS palette is a band-keyed dict; this notebook indexes
    # by integer for arbitrary curves, so flatten to a list of hex values.
    COLORS = list(_COLORS_DICT.values()) if isinstance(_COLORS_DICT, dict) else list(_COLORS_DICT)
except ImportError:
    COLORS = [
        "#2b6ca3",
        "#d65f27",
        "#3a9a5b",
        "#c03d3e",
        "#8b6bba",
    ]

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name, dpi=200):
    path = os.path.join(FIG_DIR, f"16_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")


# %% [markdown]
# ## 1. Load SSP Data and Filters
#
# Start with the standard SED modeling infrastructure: SSP templates and a filter set.

# %%
SSP_FILE = os.path.join(
    _repo_root, "data", "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
if not os.path.exists(SSP_FILE):
    print(f"WARNING: SSP file not found at {SSP_FILE}")
    print("Using alternate SSP path search...")
    for candidate in [
        os.path.join(_repo_root, "data", "ssp_prsc_miles_chabrier.h5"),
        os.path.join(_repo_root, "data", "ssp_MILES_chabrier.h5"),
    ]:
        if os.path.exists(candidate):
            SSP_FILE = candidate
            print(f"  Found: {SSP_FILE}")
            break

ssp = load_ssp_data(SSP_FILE)

# Load a filter set representing a typical multiwavelength survey
# (galex UV, sdss optical, wise IR)
filters = load_filter_set(
    [
        "galex_fuv",
        "galex_nuv",
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "wise_w1",
        "wise_w3",
    ]
)

print(f"SSP grid: {ssp.ssp_flux.shape[0]} metallicities × {ssp.ssp_flux.shape[1]} ages")
print(f"Wavelength range: {ssp.ssp_wave[0]:.0f} – {ssp.ssp_wave[-1]:.0f} Å")
print(f"Filters: {len(filters[2])} bands loaded (GALEX UV + SDSS optical + WISE IR)")

# %% [markdown]
# ## 2. Three Archetypal SFHs from Simulations
#
# Create tabulated SFHs on a common time grid that mimic outputs from cosmological
# simulations or semi-analytic models. We'll cover:
#
# - **Exponentially declining** — massive early-forming elliptical
# - **Delayed-tau** — typical disk galaxy with peak at intermediate look-back time
# - **Bursty** — recent starburst overlaid on a quiescent history

# %%
# Define time grid (cosmic time in Gyr, from early universe to z=0)
t_gyr = np.linspace(0.1, 13.7, 300)

# Three archetypal SFHs
sfr_exp = 20.0 * np.exp(-t_gyr / 2.0)  # τ = 2 Gyr
sfr_delayed = 20.0 * (t_gyr / 3.0) * np.exp(-t_gyr / 3.0)  # peak at τ = 3 Gyr
sfr_bursty = sfr_delayed.copy()
sfr_bursty[200:220] += 50.0  # Recent starburst (t ~ 11 Gyr, z ~ 0.05)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_gyr, sfr_exp, color=COLORS[0], lw=2.5, label="Exponential decline (τ = 2 Gyr)")
ax.plot(t_gyr, sfr_delayed, color=COLORS[1], lw=2.5, label="Delayed-tau (τ = 3 Gyr)")
ax.plot(
    t_gyr, sfr_bursty, color=COLORS[2], lw=2.5, ls="--", label="Bursty (delayed + recent burst)"
)
ax.set_xlabel("Cosmic time (Gyr)", fontsize=11)
ax.set_ylabel(r"SFR ($M_\odot\,\mathrm{yr}^{-1}$)", fontsize=11)
ax.set_title("Input Star Formation Histories from Simulations", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(0, 14)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, linestyle="--")
savefig(fig, "sfh_archetypes")
plt.show()

print(f"Time grid: {len(t_gyr)} points from {t_gyr[0]:.1f} to {t_gyr[-1]:.1f} Gyr")
print(f"Exponential integral (mass): {np.trapz(sfr_exp, t_gyr):.0f} Msun")
print(f"Delayed-tau integral (mass): {np.trapz(sfr_delayed, t_gyr):.0f} Msun")
print(f"Bursty integral (mass): {np.trapz(sfr_bursty, t_gyr):.0f} Msun")

# %% [markdown]
# ## 3. Forward-Model Rest-Frame SEDs
#
# For each SFH, compute the rest-frame SED using `sed_from_sfh()`. This function:
# - Interpolates the tabulated SFH onto the SSP age grid
# - Integrates the composite stellar population (CSP)
# - Applies dust attenuation
# - Returns wavelength, flux [erg/s/Hz], and stellar mass

# %%
print("\nComputing rest-frame SEDs...")
print("-" * 60)

sfh_dict = {
    "Exponential": sfr_exp,
    "Delayed-tau": sfr_delayed,
    "Bursty": sfr_bursty,
}

seds = {}
for label, sfr in sfh_dict.items():
    result = sed_from_sfh(
        t_gyr,
        sfr,
        ssp,
        log_z=-0.3,  # Solar metallicity
        dust_tau_bc=0.3,  # Birth cloud dust
        dust_tau_diff=0.5,  # Diffuse dust
        dust_slope=-0.7,  # Power-law attenuation
    )
    seds[label] = result
    print(f"{label:15s}  M_* = {result['stellar_mass']:.0f} Msun")

# Plot SEDs
fig, ax = plt.subplots(figsize=(10, 6))
for label, color in zip(sfh_dict.keys(), COLORS):
    result = seds[label]
    ax.loglog(result["wavelength"], result["sed"], lw=2, label=label, color=color)

ax.set_xlabel(r"Rest-frame wavelength (Å)", fontsize=11)
ax.set_ylabel(r"SED $L_\nu$ (erg/s/Hz)", fontsize=11)
ax.set_title("Rest-Frame SEDs from Tabulated SFHs", fontsize=12, fontweight="bold")
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e26, 1e30)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both", linestyle="--")
savefig(fig, "rest_seds")
plt.show()

# %% [markdown]
# ## 4. Full Physics: Add Metallicity, Dust, Nebular, AGN
#
# The real power of tengri's simulation interface is combining **multiple physics modules**.
# Here we add metallicity history, dust attenuation, and show the pipeline extensibility.

# %% [markdown]
# ### 4.1 Metallicity History
#
# Real simulations track Z(t). Let's add chemical enrichment:

# %%
# Metallicity history: starts metal-poor, enriches with time
log_z_history = -2.0 + 2.5 * t_gyr / 13.7  # Simple linear enrichment
log_z_history = np.clip(log_z_history, -2.5, 0.2)  # Clip to realistic range

print("Metallicity history (log Z/Zsun):")
print(f"  t = 0.1 Gyr:  log Z/Z_sun = {log_z_history[0]:.2f}")
print(f"  t = 13.7 Gyr: log Z/Z_sun = {log_z_history[-1]:.2f}")

# Re-compute SED with evolving metallicity
result_z_evolving = sed_from_sfh(
    t_gyr,
    sfr_delayed,
    ssp,
    log_z=log_z_history,  # Pass array instead of scalar
    dust_tau_bc=0.3,
    dust_tau_diff=0.5,
)

print(f"Delayed-tau SED (with Z(t)): M_* = {result_z_evolving['stellar_mass']:.0f} Msun")

# Compare constant vs evolving metallicity
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(
    seds["Delayed-tau"]["wavelength"],
    seds["Delayed-tau"]["sed"],
    lw=2.5,
    label="Constant Z = -0.3",
    color=COLORS[0],
)
ax.loglog(
    result_z_evolving["wavelength"],
    result_z_evolving["sed"],
    lw=2.5,
    label="Evolving Z(t): -2.0 → +0.2",
    color=COLORS[1],
    linestyle="--",
)
ax.set_xlabel(r"Rest-frame wavelength (Å)", fontsize=11)
ax.set_ylabel(r"SED $L_\nu$ (erg/s/Hz)", fontsize=11)
ax.set_title("Effect of Metallicity History", fontsize=12, fontweight="bold")
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e26, 1e30)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both", linestyle="--")
savefig(fig, "metallicity_history")
plt.show()

# %% [markdown]
# ## 5. Mock Photometry: Redshift, IGM, and Filter Convolution
#
# Now generate **observed photometry** at a specified redshift with IGM absorption.

# %%
print("\nGenerating mock photometry...")
print("-" * 60)

# Observer-frame photometry at z = 0.5
redshift = 0.5
dust_tau_bc = 0.3
dust_tau_diff = 0.5

photometry_data = {}
for label, sfr in sfh_dict.items():
    result = photometry_from_sfh(
        t_gyr,
        sfr,
        ssp,
        filters,
        log_z=-0.3,
        redshift=redshift,
        dust_tau_bc=dust_tau_bc,
        dust_tau_diff=dust_tau_diff,
        apply_igm=True,
    )
    photometry_data[label] = result
    flux = np.asarray(result["flux"])
    print(f"\n{label}:")
    print(f"  Stellar mass: {result['stellar_mass']:.0f} Msun")
    print(f"  r-band (SDSS): {flux[4]:.2e} erg/s/cm^2/Hz")
    print(f"  W1 (WISE):     {flux[7]:.2e} erg/s/cm^2/Hz")

# %% [markdown]
# ### 5.1 Photometric Comparison: SED via Filter Curves
#
# Visualize how the different SFHs appear in photometry.

# %%
# Extract filter info
filter_names = ["FUV", "NUV", "u", "g", "r", "i", "z", "W1", "W3"]
filter_curves = filters[2]

# Compute effective wavelengths (flux-weighted)
lambda_eff = []
for fw, ft in zip(filters[0], filters[1]):
    lam_eff = np.average(fw, weights=ft)
    lambda_eff.append(lam_eff)

lambda_eff = np.array(lambda_eff)

# Plot fluxes
fig, ax = plt.subplots(figsize=(10, 6))
for label, color in zip(sfh_dict.keys(), COLORS):
    flux = np.asarray(photometry_data[label]["flux"])
    ax.loglog(lambda_eff, flux, "o-", lw=2, markersize=8, label=label, color=color)

ax.set_xlabel(r"Observed wavelength (Å)", fontsize=11)
ax.set_ylabel(r"Flux (erg/s/cm$^2$/Hz)", fontsize=11)
ax.set_title(f"Mock Photometry at z = {redshift}", fontsize=12, fontweight="bold")
ax.set_xlim(lambda_eff.min() * 0.8, lambda_eff.max() * 1.2)
ax.set_ylim(flux.min() * 0.5, flux.max() * 2.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both", linestyle="--")
savefig(fig, "mock_photometry")
plt.show()

# %% [markdown]
# ## 6. Mock Spectra
#
# Generate observed spectra (continuous, not binned to filters).

# %%
print("\nGenerating mock spectra...")
print("-" * 60)

# Define observed-frame wavelength grid
wave_obs = np.linspace(3500, 9500, 500)  # Å

spec_data = {}
for label, sfr in sfh_dict.items():
    result = spectrum_from_sfh(
        t_gyr,
        sfr,
        ssp,
        wave_obs,
        log_z=-0.3,
        redshift=redshift,
        dust_tau_bc=dust_tau_bc,
        dust_tau_diff=dust_tau_diff,
        apply_igm=True,
        sigma_v=100.0,  # 100 km/s velocity broadening
    )
    spec_data[label] = result
    print(f"{label:15s}  spectrum shape: {np.array(result['flux']).shape}")

# Plot spectra
fig, ax = plt.subplots(figsize=(11, 6))
for label, color in zip(sfh_dict.keys(), COLORS):
    flux = np.asarray(spec_data[label]["flux"])
    ax.plot(spec_data[label]["wave_obs"], flux, lw=1.5, label=label, color=color)

ax.set_xlabel(r"Observed wavelength (Å)", fontsize=11)
ax.set_ylabel(r"Flux (erg/s/cm$^2$/Å)", fontsize=11)
ax.set_title(f"Mock Spectra (z = {redshift}, σ_v = 100 km/s)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
savefig(fig, "mock_spectra")
plt.show()

# %% [markdown]
# ## 7. Batch Processing: Many Galaxies from Simulations
#
# In practice, you'd apply this workflow to **many simulated snapshots**.
# Here's a minimal example showing the parallelizable pattern:

# %%
print("\n" + "=" * 70)
print("Batch Processing Pattern")
print("=" * 70)

# Example: 5 simulated "galaxies" with varying SFR normalization
n_gal = 5
sfr_norms = np.logspace(0, 1.5, n_gal)  # 1 to ~30 Msun/yr

batch_results = []
for i, norm in enumerate(sfr_norms):
    sfr_scaled = norm * (sfr_delayed / np.max(sfr_delayed))
    result = photometry_from_sfh(
        t_gyr,
        sfr_scaled,
        ssp,
        filters,
        log_z=-0.3,
        redshift=0.5,
        dust_tau_bc=0.2,
        dust_tau_diff=0.4,
    )
    batch_results.append(
        {
            "galaxy_id": i,
            "sfr_norm": norm,
            "mass": result["stellar_mass"],
            "flux": np.array(result["flux"]),
        }
    )

print(f"\nProcessed {n_gal} simulated galaxies:")
for res in batch_results:
    print(
        f"  Galaxy {res['galaxy_id']}: M_* = {res['mass']:.0f} Msun, "
        f"r-band = {res['flux'][4]:.2e} erg/s/cm^2/Hz"
    )

# %% [markdown]
# ## 8. Connection: Fitting This SED
#
# You now have **synthetic photometry/spectra from a simulation**. The next step is
# to **fit them** using tengri's inference engine. See:
#
# - **Notebooks 03–07** for fitting examples (photometry, spectroscopy, joint)
# - **Notebook 14** for stochastic SFH fitting
# - **Notebook 15** for VI scaling on high-dimensional models
#
# The workflow is:
# 1. Generate mock via simulation (this notebook)
# 2. Add realistic noise → `Observation`
# 3. Define `Parameters` and `SEDModel`
# 4. Run `Fitter.run("map")`, `Fitter.run("vi")`, or `Fitter.run("nuts")`
# 5. Extract posterior → compare to simulation truth

# %% [markdown]
# ## 9. Summary
#
# tengri's **simulation interface** lets you:
#
# - **Ingest tabulated SFH** from any cosmological code (IllustrisTNG, EAGLE, FIRE, SAMs)
# - **Include metallicity history** Z(t) for chemical enrichment tracking
# - **Layer physics** — dust, nebular, AGN, IGM — with fixed or free parameters
# - **Generate mock observations** — photometry, spectra, with realistic noise
# - **Batch process** many galaxies in parallel (via vmap/jit)
# - **Fit results** to recover physical constraints
#
# This bridges the gap between **simulation predictions** and **observable synthetic catalogs**,
# enabling direct comparison of models against surveys (Paper II context).

# %%
# ## What you learned
#
# - Simulation outputs (SFH + Z(t) time series) map directly to tengri inputs via tabulated interface
# - Full physics pipeline (dust, nebular, AGN, IGM) works identically for simulation-based and parametric models
# - Batch forward-modeling with vmap/jit enables rapid scanning of 100s of snapshots
# - Synthetic observations can be fit with same machinery (notebooks 03–07) to constrain simulation physics
# - Direct quantitative comparison between simulations and surveys (Paper II vision)
#
# **Next:** [`03_fitting_photometry.py`](03_fitting_photometry.py) (fit these mock SEDs to recover parameters) or
# [`14_stochastic_sfh.py`](14_stochastic_sfh.py) (stochastic SFH fits for more complex assembly histories).

# %%
try:
    from tengri import cite_all

    cite_all()
except ImportError:
    print("(citations unavailable)")
