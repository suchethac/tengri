# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dust Models: Attenuation Curves, Emission, and Energy Balance
#
# Dust is the dominant systematic in SED fitting. It absorbs UV/optical
# photons and re-emits them in the infrared, reshaping the entire SED.
# **tengri** provides a modular, fully differentiable dust framework:
#
# - **Attenuation**: 6 pluggable curves with two-component (birth cloud +
#   diffuse ISM) geometry and clumpy dust (f\_obscuration).
# - **Emission**: 3 IR models (modified blackbody, Dale 2014, Draine & Li
#   2007) plus tabulated DL07 templates, all energy-balanced.
# - **Per-component control**: different laws for birth cloud vs diffuse ISM.
#
# This notebook demonstrates every dust feature in tengri end-to-end,
# from the wavelength-dependent attenuation curves through to a full
# panchromatic SED from UV to FIR.
#
# ### References
#
# - Calzetti et al. (2000) --- starburst attenuation curve
# - Cardelli, Clayton & Mathis (1989) --- MW extinction with R\_V
# - Charlot & Fall (2000) --- two-component dust model
# - Dale et al. (2014) --- 1-parameter IR template family
# - Draine & Li (2007) --- 3-parameter grain model
# - Gordon et al. (2003) --- SMC Bar extinction
# - Kriek & Conroy (2013) --- modified Calzetti + UV bump
# - Lower et al. (2022) --- f\_obscuration (clumpy dust geometry)
# - Salim et al. (2018) --- modified Calzetti (DSPS default)

# %%
import os
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

sys.path.insert(0, ".")
from _plot_style import COLORS, setup_style

setup_style()

# Override with requested settings
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
})

os.makedirs("figures", exist_ok=True)

from tengri import Fixed, SEDModel, ParamSpec, Uniform, load_filter_set, load_ssp_data
from tengri.dust.attenuation import DUST_LAWS, two_component_dust
from tengri.dust.emission import (
    DUST_EMISSION_MODELS,
    compute_absorbed_luminosity,
    create_dl07_from_grid,
)

print(f"Registered attenuation curves: {list(DUST_LAWS.keys())}")
print(f"Registered emission models: {list(DUST_EMISSION_MODELS.keys())}")

# %% [markdown]
# ## 1. Attenuation Curves: k(lambda) vs Wavelength
#
# All 6 attenuation laws are evaluated on the same wavelength grid.
# The curves encode very different physics:
#
# | Curve | UV bump? | Free params | Typical use |
# |-------|----------|-------------|-------------|
# | power\_law | No | n\_slope | Charlot & Fall (2000) original |
# | calzetti | No | --- | Starburst galaxies |
# | kriek\_conroy | Yes | bump, delta | Prospector default |
# | smc | No | --- | High-z, low-metallicity |
# | cardelli | Yes | R\_V | MW sightlines |
# | salim | Yes | bump, delta | DSPS / Zacharegkas+2025 |
#
# The **2175 Angstrom UV bump** is a key diagnostic: it appears in the
# MW (cardelli) and can be modulated in kriek\_conroy / salim via
# `dust_bump_strength`. The SMC and Calzetti curves lack it entirely.

# %%
wave_aa = jnp.linspace(1000.0, 30000.0, 2000)

# Evaluate each curve (all accept wavelength in Angstrom)
curves = {}
for name, fn in DUST_LAWS.items():
    if name in ("kriek_conroy", "salim"):
        curves[name] = fn(wave_aa, dust_bump_strength=1.0, dust_delta=0.0)
    elif name == "power_law":
        curves[name] = fn(wave_aa, n_slope=-0.7)
    elif name == "cardelli":
        curves[name] = fn(wave_aa, dust_Rv=3.1)
    elif name == "li08":
        curves[name] = fn(wave_aa, dust_UV_slope=-1.0, dust_OPT_slope=-1.3,
                           dust_FUV_slope=-1.8, dust_bump_strength=1.0)
    else:
        curves[name] = fn(wave_aa)

# Normalize all curves to k(V)=1 at 5500 A
wave_arr = np.array(wave_aa)
v_idx = np.argmin(np.abs(wave_arr - 5500.0))
curves_norm = {}
for name, k in curves.items():
    k_arr = np.array(k)
    k_at_v = k_arr[v_idx]
    curves_norm[name] = k_arr / k_at_v if k_at_v > 0 else k_arr

# Distinct colors: avoid overlap between Kriek&Conroy and Salim
curve_colors = {
    "power_law": "#1f77b4",      # blue
    "calzetti": "#ff7f0e",       # orange
    "kriek_conroy": "#2ca02c",   # green
    "smc": "#d62728",            # red
    "cardelli": "#9467bd",       # purple
    "salim": "#e377c2",          # pink (was brown, now distinct from K&C)
    "li08": "#8c564b",            # brown
}
curve_labels = {
    "power_law": r"Power law ($n=-0.7$)",
    "calzetti": "Calzetti+2000",
    "kriek_conroy": r"Kriek & Conroy ($E_b=1$)",
    "smc": "SMC (Gordon+2003)",
    "cardelli": r"Cardelli+1989 ($R_V=3.1$)",
    "salim": r"Salim+2018 ($E_b=1$)",
    "li08": r"Li+2008 (3-slope parametric)",
}

fig, ax = plt.subplots(figsize=(9, 5.5))

wave_um = wave_arr / 1e4

for name in DUST_LAWS:
    ax.plot(
        wave_um,
        curves_norm[name],
        color=curve_colors[name],
        lw=1.8,
        label=curve_labels[name],
    )

# V-band vertical line
ax.axvline(0.55, color="0.4", ls="-", lw=0.8, zorder=0, alpha=0.5)
ax.annotate(
    r"$V$-band",
    xy=(0.55, 0.03),
    xycoords=("data", "axes fraction"),
    fontsize=9,
    color="0.4",
    ha="left",
    va="bottom",
    xytext=(5, 0),
    textcoords="offset points",
)

# 2175 A UV bump annotation
ax.axvline(0.2175, color="0.6", ls=":", lw=1.0, zorder=0)
ax.annotate(
    r"2175 $\AA$ bump",
    xy=(0.2175, 0.92),
    xycoords=("data", "axes fraction"),
    fontsize=9,
    color="0.4",
    ha="center",
    rotation=90,
)

# Log wavelength x-axis
ax.set_xscale("log")
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$k(\lambda) / k(V)$  (normalized at 5500 $\AA$)")
ax.set_xlim(0.1, 3.0)
ax.set_ylim(0, None)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])
ax.legend(loc="upper right", fontsize=9, ncol=1)

# INSET: UV bump region (1800-2600 A)
ax_inset = inset_axes(ax, width="40%", height="35%", loc="center right",
                       bbox_to_anchor=(-0.02, 0.08, 1, 1),
                       bbox_transform=ax.transAxes)
bump_mask = (wave_arr >= 1800) & (wave_arr <= 2600)
wave_bump = wave_arr[bump_mask]

for name in DUST_LAWS:
    ax_inset.plot(
        wave_bump,
        curves_norm[name][bump_mask],
        color=curve_colors[name],
        lw=1.5,
    )

ax_inset.axvline(2175, color="0.6", ls=":", lw=0.8)
ax_inset.set_xlim(1800, 2600)
ax_inset.set_xlabel(r"$\lambda$ ($\AA$)", fontsize=8)
ax_inset.set_ylabel(r"$k/k(V)$", fontsize=8)
ax_inset.tick_params(labelsize=7)
ax_inset.set_title("UV bump region", fontsize=8, pad=2)
ax_inset.patch.set_alpha(0.9)

fig.savefig("figures/13_attenuation_curves.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/13_attenuation_curves.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. f\_obscuration: Clumpy Dust Geometry (Lower+2022)
#
# In a **homogeneous dust screen**, all sightlines to stars pass through
# the same column density. But real galaxies have clumpy, porous dust:
# some sightlines escape entirely. The `f_obscuration` parameter
# (Lower et al. 2022) models this:
#
# $$
# T(\lambda) = f_{\rm obs} + (1 - f_{\rm obs})\,e^{-\tau(\lambda)}
# $$
#
# - $f_{\rm obs} = 0$: standard homogeneous screen (all light attenuated)
# - $f_{\rm obs} > 0$: a fraction $f_{\rm obs}$ of the stellar light
#   reaches the observer unattenuated
#
# The effect is strongest in the UV: at high optical depth the screen
# component goes to zero, but the $f_{\rm obs}$ floor lets UV photons
# leak through.

# %%
# Build a simple age grid (young + old populations)
age_grid = jnp.logspace(5, 10.1, 100)  # 0.1 Myr to 13 Gyr

f_obs_values = [0.0, 0.1, 0.2, 0.3]
f_obs_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Precompute a fake intrinsic SED shape (hotter = more UV)
# Use a simple power law as placeholder for illustrative purposes
wave_plot = jnp.linspace(1000.0, 30000.0, 2000)
wave_plot_um = np.array(wave_plot) / 1e4
# Simple UV-bright SED shape for illustration (nu^1 ~ lambda^-3 in Lnu)
fake_sed = (np.array(wave_plot) / 5500.0) ** (-2.0)
fake_sed = fake_sed / np.max(fake_sed)

young_idx = 5   # ~0.3 Myr -- deeply embedded in birth cloud
old_idx = -10   # ~5 Gyr -- only diffuse ISM

fig = plt.figure(figsize=(13, 8))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Top left: SED with and without f_obs (young stars)
ax_sed_y = fig.add_subplot(gs[0, 0])
# Top right: SED with and without f_obs (old stars)
ax_sed_o = fig.add_subplot(gs[0, 1])
# Bottom left: Transmission for young stars
ax_tr_y = fig.add_subplot(gs[1, 0])
# Bottom right: Transmission for old stars
ax_tr_o = fig.add_subplot(gs[1, 1])

for f_obs, color in zip(f_obs_values, f_obs_colors):
    trans = two_component_dust(
        wave_plot,
        age_grid,
        tau_v1=1.5,
        tau_v2=0.5,
        law_bc="calzetti",
        law_diff="calzetti",
        f_obscuration=f_obs,
    )
    label = rf"$f_{{\rm obs}}={f_obs}$"

    trans_young = np.array(trans[young_idx])
    trans_old = np.array(trans[old_idx])

    # SEDs
    ax_sed_y.plot(wave_plot_um, fake_sed * trans_young, color=color, lw=1.5, label=label)
    ax_sed_o.plot(wave_plot_um, fake_sed * trans_old, color=color, lw=1.5, label=label)

    # Transmission
    ax_tr_y.plot(wave_plot_um, trans_young, color=color, lw=1.5, label=label)
    ax_tr_o.plot(wave_plot_um, trans_old, color=color, lw=1.5, label=label)

# Add intrinsic SED to top panels
ax_sed_y.plot(wave_plot_um, fake_sed, color="0.6", lw=1.0, ls="--", label="Intrinsic", zorder=0)
ax_sed_o.plot(wave_plot_um, fake_sed, color="0.6", lw=1.0, ls="--", label="Intrinsic", zorder=0)

# Mark f_obs floor on transmission panels
for f_obs, color in zip(f_obs_values[1:], f_obs_colors[1:]):
    ax_tr_y.axhline(f_obs, color=color, ls="--", lw=0.8, alpha=0.5)
    ax_tr_o.axhline(f_obs, color=color, ls="--", lw=0.8, alpha=0.5)

# Labels and formatting
for ax in [ax_sed_y, ax_sed_o]:
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(r"Relative flux (attenuated SED)")
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper right")

for ax in [ax_tr_y, ax_tr_o]:
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(r"Transmission $T(\lambda)$")
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

ax_sed_y.set_title("Young stars (age < 10 Myr)", fontsize=11)
ax_sed_o.set_title("Old stars (age ~ 5 Gyr)", fontsize=11)
ax_tr_y.set_title("Transmission: young stars", fontsize=11)
ax_tr_o.set_title("Transmission: old stars", fontsize=11)

fig.suptitle(
    r"Effect of $f_{\rm obscuration}$ (Lower+2022): SED and transmission",
    fontsize=13,
    y=1.01,
)

fig.savefig("figures/13_f_obscuration.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/13_f_obscuration.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Per-Component Control: Different Laws for Birth Cloud vs Diffuse ISM
#
# The two-component Charlot & Fall (2000) framework naturally separates:
#
# - **Birth cloud** (age < 10 Myr): dense, turbulent gas surrounding young
#   stars. Steeper attenuation (power-law or SMC-like) with higher optical
#   depth.
# - **Diffuse ISM** (all ages): ambient dust with moderate attenuation
#   (Calzetti-like or MW-like with 2175A bump).
#
# tengri lets you mix any pair of curves. Below we show the transmission
# matrix for an SMC birth cloud paired with a Cardelli (MW) diffuse ISM,
# compared with a uniform Calzetti model.

# %%
# SMC birth cloud + Cardelli diffuse ISM
trans_mixed = two_component_dust(
    wave_aa,
    age_grid,
    tau_v1=1.0,
    tau_v2=0.3,
    law_bc="smc",
    law_diff="cardelli",
    dust_Rv=3.1,
)

# Same laws for both (Calzetti + Calzetti)
trans_uniform = two_component_dust(
    wave_aa,
    age_grid,
    tau_v1=1.0,
    tau_v2=0.3,
    law_bc="calzetti",
    law_diff="calzetti",
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Representative ages with physically meaningful labels
age_configs = [
    (5,  "0.3 Myr (birth cloud)", "-",  "#d62728"),
    (45, "10 Myr (transition)",    "--", "#ff7f0e"),
    (80, "1 Gyr (diffuse only)",   ":",  "#1f77b4"),
]

ax = axes[0]
for idx, label, ls, color in age_configs:
    ax.plot(
        wave_um,
        np.array(trans_mixed[idx]),
        ls=ls,
        lw=1.8,
        color=color,
        label=label,
    )
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Transmission $T(\lambda)$")
ax.set_xlim(0.1, 2.0)
ax.set_title("SMC birth cloud + Cardelli diffuse ISM", fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
for idx, label, ls, color in age_configs:
    ax.plot(
        wave_um,
        np.array(trans_uniform[idx]),
        ls=ls,
        lw=1.8,
        color=color,
        label=label,
    )
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_xlim(0.1, 2.0)
ax.set_title("Calzetti birth cloud + Calzetti diffuse ISM", fontsize=11)
ax.legend(fontsize=9)

fig.suptitle("Per-component dust law control: young vs old star attenuation",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/13_per_component_dust.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/13_per_component_dust.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Dust Emission Models
#
# tengri implements three dust emission models, all normalized by the
# energy-balance constraint $L_{\rm IR} = L_{\rm absorbed}$:
#
# | SEDModel | Parameters | Description |
# |-------|-----------|-------------|
# | `modified_blackbody` | $T_{\rm dust}$, $\beta_{\rm IR}$ | Optically-thin greybody |
# | `dale2014` | $\alpha$ | 1-param template family |
# | `draine_li2007` | $U_{\rm min}$, $\gamma$, $q_{\rm PAH}$ | 3-param grain model (analytic) |
#
# Additionally, tabulated DL07 templates can be loaded from
# `data/dl07_templates.h5` for production work.
#
# Below we show a comprehensive 3-panel comparison:
# - Left: all analytic models at the same $L_{\rm absorbed}$
# - Middle: DL07 tabulated templates varying $U_{\rm min}$
# - Right: DL07 tabulated templates varying $q_{\rm PAH}$

# %%
# Wavelength grid from UV to FIR (0.1 to 1000 um)
wave_full = jnp.logspace(np.log10(1000.0), np.log10(1e7), 3000)  # Angstrom
wave_full_um = np.array(wave_full) / 1e4
L_absorbed = 1e10  # Lsun (typical star-forming galaxy)

# Analytic models
sed_mbb = DUST_EMISSION_MODELS["modified_blackbody"](
    wave_full, L_absorbed, dust_T=35.0, dust_beta_ir=1.8
)
sed_dale = DUST_EMISSION_MODELS["dale2014"](
    wave_full, L_absorbed, dust_alpha_dale=2.0
)
sed_dl07_analytic = DUST_EMISSION_MODELS["draine_li2007"](
    wave_full, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5
)

# Tabulated DL07 templates
dl07_tabulated = create_dl07_from_grid("../data/dl07_templates.h5")
sed_dl07_tab = dl07_tabulated(
    wave_full, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5
)

# Helper: find FIR peak wavelength
def find_peak_wavelength(wave_um_arr, lnu_arr):
    """Return wavelength (um) at peak L_nu."""
    lnu = np.array(lnu_arr)
    valid = lnu > 0
    if not np.any(valid):
        return np.nan
    idx = np.argmax(lnu)
    return wave_um_arr[idx]

# --- 3-panel figure ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# LEFT: All analytic models + tabulated DL07
ax = axes[0]
model_specs = [
    (sed_mbb, "#1f77b4", "-", r"Modified BB ($T=35$ K)"),
    (sed_dale, "#ff7f0e", "-", r"Dale+2014 ($\alpha=2$)"),
    (sed_dl07_analytic, "#2ca02c", "-", "DL07 analytic"),
    (sed_dl07_tab, "#d62728", "--", "DL07 tabulated"),
]

for sed, color, ls, label in model_specs:
    sed_arr = np.array(sed)
    ax.loglog(wave_full_um, sed_arr, lw=1.5, color=color, ls=ls, label=label)
    # Mark FIR peak
    peak_um = find_peak_wavelength(wave_full_um, sed_arr)
    if not np.isnan(peak_um):
        peak_lnu = sed_arr[np.argmax(sed_arr)]
        ax.plot(peak_um, peak_lnu, "v", color=color, ms=6, zorder=5)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ Hz$^{-1}$)")
ax.set_xlim(1, 1000)
ymax = max(np.max(np.array(s)) for s, _, _, _ in model_specs)
ax.set_ylim(bottom=1e-5 * ymax)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("All models comparison", fontsize=11)

# MIDDLE: U_min variation with tabulated DL07
ax = axes[1]
umin_values = [0.1, 1.0, 5.0, 25.0]
umin_cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(umin_values)))

for umin, color in zip(umin_values, umin_cmap):
    sed_u = dl07_tabulated(
        wave_full, L_absorbed, dust_umin=umin, dust_gamma_dl=0.01, dust_qpah=2.5
    )
    sed_u_arr = np.array(sed_u)
    ax.loglog(wave_full_um, sed_u_arr, lw=1.5, color=color,
              label=rf"$U_{{\min}}={umin}$")
    # Mark FIR peak
    peak_um = find_peak_wavelength(wave_full_um, sed_u_arr)
    if not np.isnan(peak_um):
        peak_lnu = sed_u_arr[np.argmax(sed_u_arr)]
        ax.plot(peak_um, peak_lnu, "v", color=color, ms=6, zorder=5)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ Hz$^{-1}$)")
ax.set_xlim(1, 1000)
ax.legend(fontsize=9, loc="upper left")
ax.set_title(r"DL07 tabulated: varying $U_{\min}$", fontsize=11)

# RIGHT: q_PAH variation with tabulated DL07
ax = axes[2]
qpah_values = [0.47, 2.5, 4.58]
qpah_cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(qpah_values)))

for qpah, color in zip(qpah_values, qpah_cmap):
    sed_q = dl07_tabulated(
        wave_full, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=qpah
    )
    sed_q_arr = np.array(sed_q)
    ax.loglog(wave_full_um, sed_q_arr, lw=1.5, color=color,
              label=rf"$q_{{\rm PAH}}={qpah}\%$")
    # Mark FIR peak
    peak_um = find_peak_wavelength(wave_full_um, sed_q_arr)
    if not np.isnan(peak_um):
        peak_lnu = sed_q_arr[np.argmax(sed_q_arr)]
        ax.plot(peak_um, peak_lnu, "v", color=color, ms=6, zorder=5)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ Hz$^{-1}$)")
ax.set_xlim(1, 1000)
ax.legend(fontsize=9, loc="upper left")
ax.set_title(r"DL07 tabulated: varying $q_{\rm PAH}$", fontsize=11)

fig.tight_layout()
fig.savefig("figures/13_dust_emission_models.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/13_dust_emission_models.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Energy Balance: $L_{\rm IR} = L_{\rm absorbed}$
#
# The fundamental constraint linking attenuation and emission is energy
# balance: the total luminosity absorbed by dust in the UV/optical must
# equal the total luminosity re-emitted in the infrared.
#
# We verify this by:
# 1. Computing the intrinsic (dust-free) stellar SED
# 2. Applying dust attenuation
# 3. Computing $L_{\rm absorbed}$ = integral of absorbed flux over frequency
# 4. Computing $L_{\rm IR}$ = integral of dust emission over frequency
# 5. Checking that $L_{\rm IR} / L_{\rm absorbed} = 1$

# %%
# Load SSP data for a realistic stellar SED
ssp_data = load_ssp_data("../data/fsps_prsc_miles_chabrier.h5")
print(f"SSP grid: {len(ssp_data.ssp_lgmet)} metallicities, "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages, "
      f"{len(ssp_data.ssp_wave)} wavelengths")
print(f"Wavelength range: {float(ssp_data.ssp_wave[0]):.0f} -- "
      f"{float(ssp_data.ssp_wave[-1]):.0f} Angstrom")

# %%
# Create a simple star-forming galaxy model
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

spec = ParamSpec(
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(1.5),
    sfh_dpl_tau_gyr=Fixed(8.0),
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.5),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    dust_emission="modified_blackbody",
    dust_law_bc="calzetti",
    dust_law_diff="calzetti",
    mean_sfh_type="dpl",
)
model = SEDModel(spec, ssp_data, filters=filters)

params = {
    "sfh_dpl_alpha": 1.0,
    "sfh_dpl_beta": 1.5,
    "sfh_dpl_tau_gyr": 8.0,
    "sfh_dpl_log_peak_sfr": 1.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 1.5,
    "dust_tau_diff": 0.5,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

# %%
# Compute intrinsic and attenuated SEDs manually for energy balance check
from tengri.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    interpolate_metallicity,
)
from tengri.dust.emission import compute_absorbed_luminosity, modified_blackbody

# Use model internals to get SFR weights
p = model._get_internal_params(params)
sfr = model._compute_sfr(p)
sfr_on_ssp = jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)
weights = compute_csp_weights(sfr_on_ssp, model.ssp_ages_yr)

# Metallicity interpolation
from tengri.forward.sed_model import LOG10_ZSUN

log_z = -0.3 + LOG10_ZSUN
ssp_flux_at_z = interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, log_z)

# Dust attenuation
dust_atten = two_component_dust(
    ssp_data.ssp_wave,
    model.ssp_ages_yr,
    tau_v1=1.5,
    tau_v2=0.5,
    law_bc="calzetti",
    law_diff="calzetti",
)

# Intrinsic and attenuated SEDs (erg/s/Hz)
ones_atten = jnp.ones_like(dust_atten)
sed_intrinsic = compute_csp_sed(weights, ssp_flux_at_z, ones_atten)
sed_attenuated = compute_csp_sed(weights, ssp_flux_at_z, dust_atten)

# Effective (SFH-weighted) transmission
transmission_eff = jnp.where(sed_intrinsic > 0, sed_attenuated / sed_intrinsic, 1.0)

# L_absorbed via energy balance integral
L_abs = compute_absorbed_luminosity(ssp_data.ssp_wave, sed_intrinsic, transmission_eff)
print(f"L_absorbed = {float(L_abs):.4e} Lsun")

# %%
# Now compute L_IR from the emission model
sed_ir = modified_blackbody(ssp_data.ssp_wave, L_abs, dust_T=35.0, dust_beta_ir=1.6)

# Integrate L_IR over frequency
_c_aa = 2.99792458e18  # c in Angstrom/s
nu = _c_aa / ssp_data.ssp_wave
L_ir = -float(jnp.trapezoid(sed_ir, nu))

print(f"L_absorbed = {float(L_abs):.4e} Lsun")
print(f"L_IR       = {L_ir:.4e} Lsun")
print(f"L_IR / L_abs = {L_ir / float(L_abs):.6f}")
print(f"Energy balance deviation: {abs(1.0 - L_ir / float(L_abs)) * 100:.4f}%")

# %%
# Visualize energy balance with shaded absorbed region and arrow
wave_ssp_um = np.array(ssp_data.ssp_wave) / 1e4
sed_intr_arr = np.array(sed_intrinsic)
sed_att_arr = np.array(sed_attenuated)
sed_ir_arr = np.array(sed_ir)

fig, ax = plt.subplots(figsize=(10, 6))

# Intrinsic and attenuated stellar
ax.loglog(wave_ssp_um, sed_intr_arr, color="0.6", lw=1.0,
          label="Intrinsic stellar", alpha=0.7)
ax.loglog(wave_ssp_um, sed_att_arr, color="#1f77b4", lw=1.5,
          label="Attenuated stellar")

# Dust emission
ax.loglog(wave_ssp_um, sed_ir_arr, color="#d62728", lw=1.5,
          label="Dust emission (MBB)")

# Shade absorbed region (UV/optical)
ax.fill_between(
    wave_ssp_um,
    sed_att_arr,
    sed_intr_arr,
    where=sed_intr_arr > sed_att_arr,
    color="#1f77b4",
    alpha=0.15,
    label=rf"$L_{{\rm absorbed}}$ = {float(L_abs):.2e} L$_{{\odot}}$",
)

# Shade emitted IR region
ir_valid = sed_ir_arr > 0
ax.fill_between(
    wave_ssp_um,
    np.zeros_like(sed_ir_arr),
    sed_ir_arr,
    where=ir_valid,
    color="#d62728",
    alpha=0.08,
    label=rf"$L_{{\rm IR}}$ = {L_ir:.2e} L$_{{\odot}}$",
)

# Arrow: absorbed energy -> re-emitted
arrow_y = 0.3 * np.max(sed_intr_arr)
ax.annotate(
    "",
    xy=(80, arrow_y * 0.1),
    xytext=(0.3, arrow_y),
    arrowprops=dict(
        arrowstyle="->,head_width=0.3,head_length=0.2",
        color="0.3",
        lw=2,
        connectionstyle="arc3,rad=-0.3",
    ),
)
ax.text(2.0, arrow_y * 0.5, r"Energy balance" + "\n" + r"$L_{\rm IR} = L_{\rm abs}$",
        fontsize=10, color="0.3", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8))

# Ratio annotation
ax.annotate(
    rf"$L_{{\rm IR}} / L_{{\rm abs}} = {L_ir / float(L_abs):.4f}$",
    xy=(0.98, 0.95),
    xycoords="axes fraction",
    fontsize=11,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ Hz$^{-1}$)")
ax.set_xlim(0.09, 300)
ax.set_ylim(bottom=1e-5 * np.max(sed_intr_arr))
ax.legend(loc="upper left", fontsize=9)

fig.savefig("figures/13_energy_balance.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/13_energy_balance.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Full Panchromatic SED: Stellar + Dust Attenuation + Dust Emission
#
# The complete forward model in tengri:
#
# 1. Compute the intrinsic stellar SED from the SFH + metallicity + SSPs
# 2. Apply two-component dust attenuation (birth cloud + diffuse ISM)
# 3. Compute absorbed luminosity via energy balance
# 4. Add dust IR emission (scaled to $L_{\rm absorbed}$)
#
# Below we show this as $\nu L_\nu$ ($= \lambda F_\lambda$), which
# gives equal visual weight to energy output per logarithmic frequency
# interval. This reveals the stellar peak (~1 um), dust peak (~100 um),
# and PAH features (~6-12 um) on equal footing.

# %%
# Full panchromatic SED in nu*L_nu
fig, ax = plt.subplots(figsize=(11, 6.5))

# Convert to nu*L_nu = L_nu * c / wavelength (using wavelength in cm)
_c_cgs = 2.99792458e10  # cm/s
wave_cm = np.array(ssp_data.ssp_wave) * 1e-8

# Intrinsic stellar (faint)
nu_lnu_intr = sed_intr_arr * _c_cgs / wave_cm
ax.loglog(wave_ssp_um, nu_lnu_intr, color="0.75", lw=0.8,
          label="Intrinsic stellar", alpha=0.6, zorder=1)

# Attenuated stellar
nu_lnu_att = sed_att_arr * _c_cgs / wave_cm
ax.loglog(wave_ssp_um, nu_lnu_att, color="0.3", lw=1.5,
          label="Attenuated stellar", zorder=2)

# Three emission models (total = attenuated + IR)
emission_configs = [
    ("modified_blackbody", dict(dust_T=35.0, dust_beta_ir=1.8),
     "#d62728", r"+ MBB ($T=35$ K)"),
    ("dale2014", dict(dust_alpha_dale=2.0),
     "#ff7f0e", r"+ Dale+2014 ($\alpha=2$)"),
]

for model_name, kw, color, label in emission_configs:
    ir_sed = DUST_EMISSION_MODELS[model_name](ssp_data.ssp_wave, L_abs, **kw)
    total = sed_att_arr + np.array(ir_sed)
    nu_lnu_total = total * _c_cgs / wave_cm
    ax.loglog(wave_ssp_um, nu_lnu_total, lw=1.5, color=color,
              label=f"Total {label}", zorder=3)

# DL07 tabulated total (highlight)
ir_tab = dl07_tabulated(ssp_data.ssp_wave, L_abs,
                        dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
total_tab = sed_att_arr + np.array(ir_tab)
nu_lnu_tab = total_tab * _c_cgs / wave_cm
ax.loglog(wave_ssp_um, nu_lnu_tab, lw=2.0, color="#2ca02c",
          label="Total + DL07 tabulated", zorder=4)

# Label key features
# Stellar peak
stellar_peak_idx = np.argmax(nu_lnu_intr)
ax.annotate(
    "Stellar\npeak",
    xy=(wave_ssp_um[stellar_peak_idx], nu_lnu_intr[stellar_peak_idx]),
    xytext=(0.05, 0.85),
    textcoords="axes fraction",
    fontsize=9,
    color="0.4",
    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
    ha="center",
)

# Dust peak (from DL07 tabulated)
dust_peak_idx = np.argmax(nu_lnu_tab[len(nu_lnu_tab) // 2:]) + len(nu_lnu_tab) // 2
if dust_peak_idx < len(wave_ssp_um):
    ax.annotate(
        "Dust\npeak",
        xy=(wave_ssp_um[dust_peak_idx], nu_lnu_tab[dust_peak_idx]),
        xytext=(0.88, 0.75),
        textcoords="axes fraction",
        fontsize=9,
        color="0.4",
        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
        ha="center",
    )

# PAH region annotation
ax.axvspan(6, 13, alpha=0.04, color="#2ca02c", zorder=0)
ax.text(9.0, 0.03, "PAH", transform=ax.get_xaxis_transform(),
        fontsize=9, color="#2ca02c", ha="center", alpha=0.7)

# Wavelength region shading
ax.axvspan(0.09, 0.3, alpha=0.03, color="blue", zorder=0)
ax.axvspan(30, 1000, alpha=0.03, color="red", zorder=0)
ax.text(0.15, 0.03, "UV", transform=ax.get_xaxis_transform(),
        fontsize=9, color="0.5", ha="center")
ax.text(100, 0.03, "FIR", transform=ax.get_xaxis_transform(),
        fontsize=9, color="0.5", ha="center")

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$\nu L_\nu$ (L$_\odot$)")
ax.set_xlim(0.09, 500)
valid_nuLnu = nu_lnu_intr[nu_lnu_intr > 0]
ax.set_ylim(bottom=1e-4 * np.max(valid_nuLnu))
ax.legend(loc="upper right", fontsize=9, ncol=1)

fig.savefig("figures/13_panchromatic_sed.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/13_panchromatic_sed.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook demonstrated tengri's full dust framework:
#
# 1. **6 attenuation curves** with distinct UV behaviour (bump vs no bump),
#    all normalized to $k(V)=1$, with inset showing the 2175 A region
# 2. **f\_obscuration** for clumpy dust geometry (Lower+2022), showing
#    both the SED modification and the transmission floor
# 3. **Per-component control**: different laws for birth cloud vs diffuse ISM,
#    with physically meaningful age labels
# 4. **3 analytic emission models** + tabulated DL07 templates, compared
#    across $U_{\rm min}$ and $q_{\rm PAH}$ parameter space
# 5. **Energy balance** verification: $L_{\rm IR} = L_{\rm absorbed}$ to
#    machine precision, with clear visualization of absorbed and re-emitted energy
# 6. **Full panchromatic SEDs** in $\nu L_\nu$ from UV to FIR, with
#    stellar peak, dust peak, and PAH features labeled
#
# Every function is pure JAX --- JIT-compilable and fully differentiable
# for gradient-based inference.
