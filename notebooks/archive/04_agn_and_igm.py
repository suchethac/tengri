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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AGN Models and IGM Absorption
#
# tengri includes modular AGN emission models (accretion disc + dust torus)
# and an Inoue et al. (2014) IGM absorption prescription. This notebook
# explores the available AGN configurations and shows how IGM absorption
# modifies the SED at different redshifts.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fixed,
    SEDModel,
    ParamSpec,
    load_ssp_data,
)
from tengri.agn import (
    AGN_MODELS,
    get_agn_model,
    powerlaw_disc,
    simple_torus,
    unified_agn,
)
from tengri.igm import igm_transmission

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
# Change to project root so data/ paths work
# chdir to project root for data/ access
if os.path.exists("data"):
    pass  # already in project root
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import COLORS, SPECTRAL_FEATURES, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures", "reference")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. Available AGN Models
#
# tengri provides several AGN SED configurations registered in a model
# registry.

# %%
print("Registered AGN models:")
for name in AGN_MODELS:
    print(f"  - {name}")

# %% [markdown]
# ## 2. AGN Component Anatomy
#
# A typical AGN SED has two main thermal components:
# - **Accretion disc**: UV/optical power-law emission
# - **Dust torus**: IR thermal re-emission
#
# The disc illuminates the torus, which absorbs and re-radiates. The
# `agn_torus_frac` parameter controls the fraction of bolometric luminosity
# re-emitted by the torus.

# %%
# --- FIGURE 1: AGN component anatomy ---
wavelength = jnp.logspace(np.log10(100), np.log10(1e6), 1000)

# Disc component
disc_lnu = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_slope=-1.5)

# Torus component
torus_lnu = simple_torus(wavelength, agn_log_lbol=44.0, agn_torus_temp=1500.0)

# Combined
combined = disc_lnu + 0.5 * torus_lnu

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(
    np.array(wavelength) / 1e4,
    np.array(disc_lnu),
    "--",
    color=COLORS["rt"],
    lw=1.5,
    label="Accretion disc",
)
ax.loglog(
    np.array(wavelength) / 1e4,
    np.array(torus_lnu),
    "--",
    color=COLORS["geovi"],
    lw=1.5,
    label="Dust torus",
)
ax.loglog(np.array(wavelength) / 1e4, np.array(combined), "k-", lw=2, label="Combined AGN")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
ax.set_title("AGN SED Components (Simple SEDModel)")
ax.set_xlim(0.01, 100)
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "04_agn_components.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. AGN SEDModel Comparison
#
# Compare the registered AGN models at the same bolometric luminosity.

# %%
# --- FIGURE 2: SEDModel comparison ---
fig, ax = plt.subplots(figsize=(9, 5))
model_colors = [
    COLORS["rt"],
    COLORS["geovi"],
    COLORS["nuts"],
    COLORS["mgvi"],
    "#e377c2",
    "#8c564b",
]

for i, name in enumerate(AGN_MODELS):
    try:
        agn_fn = get_agn_model(name)
        lnu = agn_fn(wavelength, agn_log_lbol=44.0)
        color = model_colors[i % len(model_colors)]
        ax.loglog(np.array(wavelength) / 1e4, np.array(lnu), color=color, lw=1.2, label=name)
    except Exception as e:
        print(f"  Skipping {name}: {e}")

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
ax.set_title("AGN SEDModel Comparison (log $L_{\\rm bol}$ = 44)")
ax.set_xlim(0.01, 100)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "04_agn_model_comparison.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Galaxy + AGN SED
#
# A realistic SED combines stellar emission with an AGN contribution.
# We show how adding an AGN modifies the broadband SED.

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# Pure stellar SED
spec_stellar = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.0),
)
model_stellar = SEDModel(spec_stellar, ssp_data)
params_stellar = {
    k: v.value if hasattr(v, "value") else v
    for k, v in {
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 3.0,
        "sfh_tsnorm_width_gyr": 2.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 5.0,
        "met_logzsol": -0.2,
        "dust_tau_bc": 0.3,
        "dust_tau_diff": 0.5,
        "dust_slope": -0.7,
        "redshift": 0.0,
    }.items()
}
sed_stellar = model_stellar.predict_sed(params_stellar)
wave_rest = ssp_data.ssp_wave

# --- FIGURE 3: Galaxy + AGN at different fractions ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(
    np.array(wave_rest), np.array(sed_stellar), "k-", lw=1.5, alpha=0.5, label="Stellar only"
)

agn_fracs = [0.01, 0.1, 0.5]
agn_colors = [COLORS["seq"][2], COLORS["seq"][3], COLORS["seq"][4]]
for frac, color in zip(agn_fracs, agn_colors):
    agn_lnu = unified_agn(wave_rest, agn_log_lbol=44.0, agn_torus_frac=0.5)
    # Scale AGN relative to stellar
    stellar_lbol = float(jnp.sum(sed_stellar))
    agn_scale = frac * stellar_lbol / float(jnp.sum(agn_lnu) + 1e-30)
    combined = sed_stellar + agn_scale * agn_lnu
    ax.loglog(
        np.array(wave_rest), np.array(combined), color=color, lw=1.2, label=f"AGN frac = {frac}"
    )

ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$L_\nu$ [arbitrary]")
ax.set_title("Galaxy + AGN SED")
ax.set_xlim(900, 50000)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "04_galaxy_agn.png"), bbox_inches="tight")
plt.show()

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
# ## Summary
#
# | Component | Module | Key parameters |
# |-----------|--------|---------------|
# | Power-law disc | `disc.powerlaw_disc` | `agn_log_lbol`, `agn_slope` |
# | Multi-color disc | `disc.multicolor_disc` | `agn_log_lbol`, `agn_mass_bh` |
# | Simple torus | `torus.simple_torus` | `agn_log_lbol`, `agn_torus_temp` |
# | Two-temp torus | `torus.two_temperature_torus` | warm + cool components |
# | IGM (Inoue+2014) | `models.igm` | `z_source`, `add_cgm` |
#
# The AGN components are additive: the total SED is stellar + disc + torus.
# At high redshift ($z > 5$), IGM absorption becomes the dominant spectral
# modification, and the optional CGM damping wing term becomes relevant.
#
# **See also:** [Advanced AGN Models](../_notebooks/reference/11_advanced_agn) for
# Kubota & Done multi-color disc, SKIRTOR clumpy torus, and unified NLR/BLR models.
