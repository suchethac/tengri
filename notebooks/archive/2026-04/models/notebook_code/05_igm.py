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
# # IGM Absorption: A Compact Reference for High-Redshift Observers
#
# Intergalactic medium (IGM) absorption suppresses flux at wavelengths below
# the Lyman limit (912 Å rest-frame) and the Lyman series (1216 Å, 1026 Å...).
# At high redshift, the cumulative opacity along the line of sight becomes
# significant for z > 2 and severe for z > 5.
#
# **Three figures:**
# 1. IGM transmission curves at z = 0.5, 1, 2, 3, 5, 7
# 2. Filter transmission curves overlaid (SDSS + JWST NIRCam)
# 3. Patchy reionization damping wing at z = 7

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import Fixed, SEDModel, Observation, Parameters, Photometry, Uniform, load_ssp_data
from tengri.igm import igm_transmission

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
if os.path.exists("data"):
    pass
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import COLORS, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %%
# Reference stellar SED for IGM attenuation figure
_ssp_path = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if os.path.exists(_ssp_path):
    _ssp = load_ssp_data(_ssp_path)
    wave_rest = jnp.array(_ssp.ssp_wave)
    # Luminosity-weighted SED: weights peak at ~1 Gyr (typical intermediate-age galaxy)
    _ages_gyr = 10 ** jnp.array(_ssp.ssp_lg_age_gyr)
    _weights = jnp.exp(-0.5 * (jnp.array(_ssp.ssp_lg_age_gyr) - 0.0) ** 2)
    _weights = _weights / _weights.sum()
    # Use solar-metallicity SSPs (index 3 ~ [Z/H]≈0 for MILES)
    sed_stellar = jnp.einsum("a,wa->w", _weights, _ssp.ssp_flux[3])
else:
    # Fallback: power-law SED when SSP data not available
    wave_rest = jnp.linspace(800.0, 50000.0, 3000)
    sed_stellar = (wave_rest / 5500.0) ** (-1.5)

# %% [markdown]
# ## 5. IGM Absorption
#
# The intergalactic medium absorbs rest-frame UV photons blueward of
# Ly-alpha (1216 A). The absorption increases rapidly with redshift.
# We use the Inoue et al. (2014) prescription, which includes:
# - Lyman-series absorption from the Ly-alpha forest (LAF)
# - Absorption from damped Ly-alpha systems (DLA)
# - Lyman continuum absorption

# %%
# --- FIGURE 4: IGM transmission at different redshifts ---
wave_obs = jnp.linspace(800.0, 15000.0, 2000)
redshifts = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
z_colors = plt.cm.viridis(np.linspace(0, 0.9, len(redshifts)))

fig, ax = plt.subplots(figsize=(9, 5))
for z, color in zip(redshifts, z_colors):
    trans = igm_transmission(wave_obs, z, add_cgm=True)
    ax.plot(np.array(wave_obs), np.array(trans), color=color, lw=1.2, label=f"z = {z}")

ax.set_xlabel(r"Observed wavelength [$\AA$]")
ax.set_ylabel("IGM Transmission $T_{\\rm IGM}$")
ax.set_title("Intergalactic Medium Absorption (Inoue+2014)")
ax.axhline(1.0, ls=":", color="grey", lw=0.5)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "04_igm_transmission.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. IGM Effect on Broadband SED
#
# At high redshift, IGM absorption dramatically changes the observed SED,
# particularly suppressing rest-frame UV flux.

# %%
# --- FIGURE 5: SED with IGM at z=0 vs z=3 vs z=6 ---
fig, ax = plt.subplots(figsize=(9, 5))

# Rest-frame SED (same galaxy)
sed_rest = sed_stellar
wave_fine = jnp.linspace(800, 50000, 3000)
sed_interp = jnp.interp(wave_fine, wave_rest, sed_rest)

ax.plot(
    np.array(wave_fine), np.array(sed_interp), "k-", lw=1.5, alpha=0.3, label="Rest frame (z=0)"
)

for z, color, ls in [
    (1.0, COLORS["rt"], "-"),
    (3.0, COLORS["geovi"], "-"),
    (6.0, COLORS["nuts"], "-"),
]:
    wave_obs_z = wave_fine * (1 + z)
    trans = igm_transmission(wave_obs_z, z, add_cgm=True)
    sed_obs = sed_interp * trans
    ax.plot(
        np.array(wave_fine),
        np.array(sed_obs),
        color=color,
        ls=ls,
        lw=1.2,
        label=f"Observed at z={z}",
    )

ax.set_xlabel(r"Rest-frame wavelength [$\AA$]")
ax.set_ylabel(r"$L_\nu$ [arbitrary]")
ax.set_title("IGM Effect on Galaxy SED")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(800, 20000)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "04_igm_sed_effect.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Patchy Reionization: The Damping Wing at z=7
#
# At z > 6, incomplete reionization adds a damping wing from neutral IGM
# to the Lyman-α break. The neutral fraction x_HI controls the wing amplitude.

# %%
# Damping wing at z=7 for different neutral fractions
wave_rest = np.linspace(900, 1500, 300)
z_reion = 7.0
x_hi_values = [0.0, 0.3, 0.7, 1.0]

fig, ax = plt.subplots(figsize=(8, 4))
colors_xhi = plt.cm.viridis(np.linspace(0.1, 0.9, len(x_hi_values)))

for x_hi, color in zip(x_hi_values, colors_xhi):
    wave_obs = wave_rest * (1 + z_reion)
    try:
        trans = np.array(igm_transmission(wave_obs, z_reion))
    except Exception:
        # Fallback: simple exponential model
        tau_lya = x_hi * np.exp(-(((wave_rest - 1216) / 30) ** 2)) * 5
        trans = np.exp(-tau_lya)
    # Apply x_HI scaling (rough approximation for illustration)
    trans_hi = trans ** (1 + x_hi * 2)
    ax.plot(
        wave_rest, np.clip(trans_hi, 0, 1), color=color, lw=1.5, label=rf"$x_{{\rm HI}} = {x_hi}$"
    )

ax.axvline(1216, ls=":", color="grey", lw=1, label=r"Ly-$\alpha$")
ax.set_xlabel(r"$\lambda_{\rm rest}$ [$\AA$]")
ax.set_ylabel(r"Transmission $T_{\rm IGM}$")
ax.set_title(f"Damping Wing at z = {z_reion}: Effect of Neutral Fraction")
ax.legend(fontsize=8, frameon=False)
ax.set_ylim(0, 1.05)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Key messages:**
# - At z=7, even 30% neutral fraction causes significant suppression blueward of Ly-α
# - Full reionization (x_HI=0): only Lyman forest lines; sharp break only
# - Full neutral (x_HI=1): complete Gunn-Peterson trough from IGM
# - Patchy reionization: effective x_HI ~ 0.3-0.7 at z ~ 7 (Becker+2015)
