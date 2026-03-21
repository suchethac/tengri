# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Forward-Modeling Simulated SFHs
#
# diffsed can forward-model SEDs from arbitrary tabulated star formation
# histories -- for example, from cosmological simulations like IllustrisTNG,
# EAGLE, or UniverseMachine. The `simulate` module bypasses the parametric
# SFH models and directly uses the DSPS CSP integral.
#
# This notebook demonstrates:
# 1. How to use `sed_from_sfh` and `photometry_from_sfh` for tabulated SFHs
# 2. Example SFH shapes from simulations
# 3. Batch processing of multiple galaxies

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from diffsed import load_filter_set, load_ssp_data
from diffsed.simulate import photometry_from_sfh, sed_from_sfh

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _nb_dir = os.getcwd()
sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
from _plot_style import COLORS, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures", "reference")
os.makedirs(FIGDIR, exist_ok=True)

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## 1. Tabulated SFH Examples
#
# We create synthetic SFHs that mimic typical simulation outputs:
# cosmic time grid in Gyr with SFR in Msun/yr.

# %%
# Define a time grid (cosmic time in Gyr)
t_gyr = np.linspace(0.5, 13.7, 200)

# Create several archetypal SFHs
sfh_library = {
    "Exponential decline": {
        "sfr": 30.0 * np.exp(-t_gyr / 3.0),
        "color": COLORS["rt"],
        "log_z": -0.3,
    },
    "Delayed-tau": {
        "sfr": 15.0 * (t_gyr / 3.0) * np.exp(-t_gyr / 3.0),
        "color": COLORS["geovi"],
        "log_z": -0.5,
    },
    "Late-time burst": {
        "sfr": 5.0 * np.exp(-t_gyr / 5.0) + 50.0 * np.exp(-0.5 * ((t_gyr - 10.0) / 0.5) ** 2),
        "color": COLORS["nuts"],
        "log_z": -0.2,
    },
    "Constant + quench": {
        "sfr": 10.0 * np.where(t_gyr < 8.0, 1.0, np.exp(-(t_gyr - 8.0) / 0.5)),
        "color": COLORS["mgvi"],
        "log_z": -0.1,
    },
    "Rising (high-z analog)": {
        "sfr": 0.5 * np.exp(t_gyr / 5.0),
        "color": "#e377c2",
        "log_z": -1.0,
    },
}

# %%
# --- FIGURE 1: Tabulated SFH examples ---
fig, ax = plt.subplots(figsize=(9, 5))
for name, sfh in sfh_library.items():
    ax.plot(t_gyr, sfh["sfr"], color=sfh["color"], lw=1.5, label=name)
ax.set_xlabel("Cosmic time [Gyr]")
ax.set_ylabel("SFR [$M_\\odot$/yr]")
ax.set_title("Example Tabulated Star Formation Histories")
ax.legend(fontsize=8, frameon=False)
ax.set_xlim(0, 14)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "09_tabulated_sfhs.pdf"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Rest-Frame SED from Tabulated SFH
#
# The `sed_from_sfh` function computes the rest-frame SED for any
# tabulated SFH. It handles the CSP integral, metallicity interpolation,
# and dust attenuation internally.

# %%
# --- FIGURE 2: SEDs from each tabulated SFH ---
fig, ax = plt.subplots(figsize=(10, 5))

for name, sfh in sfh_library.items():
    result = sed_from_sfh(
        jnp.array(t_gyr),
        jnp.array(sfh["sfr"]),
        ssp_data,
        log_z=sfh["log_z"],
        dust_tau_bc=0.3,
        dust_tau_diff=0.5,
    )
    wave = np.array(result["wavelength"])
    sed = np.array(result["sed"])
    ax.loglog(wave, sed, color=sfh["color"], lw=1.0, label=name, alpha=0.8)

ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("Rest-Frame SEDs from Tabulated SFHs")
ax.set_xlim(900, 50000)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "09_tabulated_seds.pdf"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Batch Processing: Photometric Catalog
#
# For simulation post-processing, you often need to compute photometry
# for thousands of galaxies. We demonstrate batch processing with
# `photometry_from_sfh`.

# %%
# Create a mini-catalog of 50 random SFHs
n_galaxies = 50
key = jax.random.PRNGKey(123)

# Random SFH parameters
catalog_sfrs = []
catalog_logz = []
for i in range(n_galaxies):
    subkey = jax.random.fold_in(key, i)
    keys = jax.random.split(subkey, 4)
    # Random tau model with noise
    tau = float(jax.random.uniform(keys[0], minval=1.0, maxval=8.0))
    norm = float(jax.random.uniform(keys[1], minval=1.0, maxval=50.0))
    sfr = norm * np.exp(-t_gyr / tau)
    # Add some stochastic bursts
    n_bursts = int(jax.random.poisson(keys[2], 2))
    for b in range(min(n_bursts, 3)):
        burst_key = jax.random.fold_in(keys[3], b)
        burst_time = float(jax.random.uniform(burst_key, minval=2.0, maxval=12.0))
        sfr += 20.0 * np.exp(-0.5 * ((t_gyr - burst_time) / 0.3) ** 2)
    catalog_sfrs.append(sfr)
    catalog_logz.append(float(jax.random.uniform(keys[3], minval=-1.5, maxval=0.0)))

# Compute photometry for all galaxies
print(f"Computing photometry for {n_galaxies} galaxies...")
import time

t0 = time.time()
catalog_phot = []
for i in range(n_galaxies):
    phot = photometry_from_sfh(
        jnp.array(t_gyr),
        jnp.array(catalog_sfrs[i]),
        ssp_data,
        filters,
        log_z=catalog_logz[i],
        redshift=0.1,
        dust_tau_bc=0.2,
        dust_tau_diff=0.4,
    )
    catalog_phot.append(np.array(phot))
catalog_phot = np.array(catalog_phot)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s ({elapsed / n_galaxies * 1000:.0f} ms/galaxy)")

# %%
# --- FIGURE 3: Color-magnitude diagram from catalog ---
fig, ax = plt.subplots(figsize=(7, 5))

# g-r color vs r-band magnitude
g_flux = catalog_phot[:, 1]
r_flux = catalog_phot[:, 2]
# Convert to magnitudes (AB)
g_mag = -2.5 * np.log10(np.maximum(g_flux, 1e-30)) - 48.6
r_mag = -2.5 * np.log10(np.maximum(r_flux, 1e-30)) - 48.6
gr_color = g_mag - r_mag

# Color by metallicity
sc = ax.scatter(
    r_mag, gr_color, c=catalog_logz, cmap="viridis", s=20, alpha=0.8, edgecolors="k", lw=0.3
)
plt.colorbar(sc, ax=ax, label=r"$\log(Z/Z_\odot)$")
ax.set_xlabel("$r$-band magnitude [AB]")
ax.set_ylabel("$g - r$ color [AB]")
ax.set_title("Color-Magnitude Diagram from Simulated Catalog")
ax.invert_xaxis()
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "09_color_magnitude.pdf"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Function | Input | Output | Use case |
# |----------|-------|--------|----------|
# | `sed_from_sfh` | t_gyr, SFR arrays | Rest-frame SED | Detailed analysis |
# | `photometry_from_sfh` | t_gyr, SFR + filters | Observed photometry | Catalog generation |
#
# **Performance**: On a laptop CPU, each galaxy takes ~10-50 ms depending
# on grid resolution and whether dust is enabled. For large catalogs,
# consider `jax.vmap` for batch vectorization.
#
# **Key features**:
# - Arbitrary SFH shape (no parametric assumption)
# - Scalar or evolving metallicity $Z(t)$
# - Full dust model with any attenuation curve
# - IGM absorption at specified redshift
# - JIT-compilable for speed
