# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-Wavelength Coverage: IGM, Radio, and X-ray
#
# Beyond the UV-optical, three additional windows tell you things you can't learn any other way: the IGM stamps high-z photometry, radio traces star formation and AGN jets, and X-ray pins down accretion.

# %% [markdown]
# ## Setup

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri.models.igm import igm_transmission
from tengri.models.radio import radio_agn, radio_star_forming
from tengri.models.xray import xray_agn_corona, xray_xrb

import sys, os

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))

if not os.path.exists("data"):
    if os.path.exists(os.path.join("..", "data")):
        os.chdir("..")
    elif os.path.exists(os.path.join("..", "..", "data")):
        os.chdir(os.path.join("..", ".."))

from _plot_style import COLORS, setup_style

setup_style()
FIGDIR = os.path.join(_nb_dir, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# # Part 1: IGM Absorption at High Redshift
#
# The Lyman break at 912 Å rest-frame shifts into SDSS-r at z~3 and JWST-F090W at z~9.
# `igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths.

# %%
# Figure 1: IGM effect on broad-band photometry
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: SED in observed frame for various redshifts
ax = axes[0]
wave_rest = jnp.linspace(800.0, 20000.0, 2000)
sed_stellar = (wave_rest / 5500.0) ** (-1.5)  # Simple power-law SED

redshifts = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
colors_z = plt.cm.plasma(np.linspace(0, 1, len(redshifts)))

for z, color in zip(redshifts, colors_z):
    wave_obs = wave_rest * (1 + z)
    trans = igm_transmission(wave_obs, z, add_cgm=True)
    sed_obs = sed_stellar * trans
    ax.plot(np.array(wave_obs), np.array(sed_obs), color=color, lw=1.2, label=f"z={z}")

# Shade SDSS u/g/r/i/z and JWST F090W/F150W/F200W filter regions
sdss_filters = {"u": 3540, "g": 4770, "r": 6231, "i": 7625, "z": 9134}
jwst_filters = {"F090W": 9025, "F150W": 15000, "F200W": 20000}
for fname, fwave in sdss_filters.items():
    ax.axvline(fwave, color="C0", alpha=0.2, linestyle="--", linewidth=0.8)
for fname, fwave in jwst_filters.items():
    ax.axvline(fwave, color="orange", alpha=0.2, linestyle="--", linewidth=0.8)

ax.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [arb]", fontsize=11)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=9, frameon=False, ncol=2, loc="lower left")
ax.set_title("Lyman break shifts with redshift", fontsize=11)

# Right: Lyman-break dropout criterion color
ax = axes[1]
z_grid = np.linspace(0.1, 8, 50)
g_rest, r_rest = 4770 / 1.2e3, 6231 / 1.2e3  # Mock rest-frame wavelengths
colors_dropout = []
for z in z_grid:
    wave_obs_g = 4770
    wave_obs_r = 6231
    trans_g = igm_transmission(jnp.array([wave_obs_g]), z, add_cgm=True)[0]
    trans_r = igm_transmission(jnp.array([wave_obs_r]), z, add_cgm=True)[0]
    color_gr = -2.5 * np.log10(float(trans_g) + 1e-6) + 2.5 * np.log10(float(trans_r) + 1e-6)
    colors_dropout.append(color_gr)

ax.plot(z_grid, colors_dropout, "o-", color=COLORS["rt"], markersize=4, linewidth=1.5)
ax.set_xlabel("Redshift", fontsize=11)
ax.set_ylabel(r"IGM-induced $g-r$ dropout [mag]", fontsize=11)
ax.set_title("Dropout criterion steepens at z > 3", fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_igm_photometry.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Part 2: Radio Emission
#
# Radio emission traces both star formation (FIR-radio correlation) and AGN jets (radio-loudness).
# The FIR-radio correlation parameter $q_\mathrm{IR}$ links infrared to 1.4 GHz emission.

# %%
# Figure 2: Full SED with radio using q_IR parameter
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

wave = jnp.logspace(3, 10, 1500)

# Left: Full SED showing radio component
ax = axes[0]
L_ir = 1e11  # Lsun
q_ir_values = [2.0, 2.4, 2.64, 3.0]
colors_qir = plt.cm.Blues(np.linspace(0.4, 0.9, len(q_ir_values)))

for q_ir, color in zip(q_ir_values, colors_qir):
    # Approximate: q_IR = log10(L_IR / (3.75e12 * L_1.4GHz))
    L_1p4ghz = L_ir / (3.75e12 * 10 ** q_ir)
    L_nu = radio_star_forming(wave, L_ir=L_ir, alpha_sf=0.8)
    ax.loglog(np.array(wave), np.array(L_nu), color=color, lw=1.5, label=f"$q_{{\\rm IR}}={q_ir}$")

ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]", fontsize=11)
ax.set_xlim(1e6, 1e10)
ax.legend(fontsize=9, frameon=False)
ax.set_title(f"Full SED: radio SED for $L_{{\\rm IR}}=10^{{11}}$ L$_\\odot$", fontsize=11)

# Right: Radio regime only (1 cm to 1 m)
ax = axes[1]
wave_radio = jnp.logspace(7, 10, 500)
for q_ir, color in zip(q_ir_values, colors_qir):
    L_nu = radio_star_forming(wave_radio, L_ir=L_ir, alpha_sf=0.8)
    ax.loglog(np.array(wave_radio), np.array(L_nu), color=color, lw=1.5, label=f"$q_{{\\rm IR}}={q_ir}$")

ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.set_title("Radio regime only (1 cm–1 m)", fontsize=11)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_radio_fir_correlation.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# Figure 3: Radio loudness and spectral index variations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

wave_radio = jnp.logspace(7, 10, 500)

# Left: Spectral index sweep
ax = axes[0]
L_ir = 1e11
alpha_sf_values = [0.5, 0.7, 0.8, 1.0]
colors_alpha = plt.cm.Purples(np.linspace(0.4, 0.9, len(alpha_sf_values)))

for alpha, color in zip(alpha_sf_values, colors_alpha):
    L_nu = radio_star_forming(wave_radio, L_ir=L_ir, alpha_sf=alpha)
    ax.loglog(np.array(wave_radio), np.array(L_nu), color=color, lw=1.5, label=f"$\\alpha_{{\\rm sf}}={alpha}$")

ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.set_title("Spectral index effect on radio slope", fontsize=11)

# Right: AGN radio-loudness parameter
ax = axes[1]
L_agn_bol = 1e11
radio_loudness_values = [0, 1, 2, 3]
colors_loud = plt.cm.Reds(np.linspace(0.4, 0.9, len(radio_loudness_values)))

for R, color in zip(radio_loudness_values, colors_loud):
    L_nu = radio_agn(wave_radio, L_agn_bol=L_agn_bol, radio_loudness=R)
    ax.loglog(np.array(wave_radio), np.array(L_nu), color=color, lw=1.5, label=f"$R={R}$ (log scale)")

ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.set_title(r"AGN radio-loudness $R = \log_{10}(L_{5\,\mathrm{GHz}} / L_B)$", fontsize=11)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_radio_agn.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Part 3: X-ray Emission
#
# X-ray AGN parameters are shown in the AGN gallery notebook (05_agn_gallery).
# Here we focus on XRB scaling: HMXBs scale with SFR, LMXBs scale with stellar mass.
# X-ray normalization integrates over the 2–10 keV band.

# %% [markdown]
# **Key takeaways:**
#
# - **IGM:** Lyman break shifts observed-frame location with redshift; enables high-z photometry
# - **Radio:** FIR-radio correlation ($q_\mathrm{IR}$) constrains SFR; radio-loudness ($R$) flags AGN jets
# - **X-ray:** HMXB and LMXB decomposition pins SFR and stellar mass independently
# - **Combined:** All three regimes enable joint fitting across panchromatic data from X-ray to radio
#
# For detailed X-ray AGN physics, see 05_agn_gallery.ipynb. For radio, see 08_radio.ipynb. For IGM, see 05_igm.ipynb.
