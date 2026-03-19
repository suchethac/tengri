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
# **diffsed** provides a modular, fully differentiable dust framework:
#
# - **Attenuation**: 6 pluggable curves with two-component (birth cloud +
#   diffuse ISM) geometry and clumpy dust (f\_obscuration).
# - **Emission**: 3 IR models (modified blackbody, Dale 2014, Draine & Li
#   2007) plus tabulated DL07 templates, all energy-balanced.
# - **Per-component control**: different laws for birth cloud vs diffuse ISM.
#
# This notebook demonstrates every dust feature in diffsed end-to-end,
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
import numpy as np

sys.path.insert(0, ".")
from _plot_style import COLORS, setup_style

setup_style()
os.makedirs("figures", exist_ok=True)

from diffsed import Fixed, Model, ParamSpec, Uniform, load_filter_set, load_ssp_data
from diffsed.models.dust.attenuation import DUST_LAWS, two_component_dust
from diffsed.models.dust.emission import (
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
        # Show the bump for these laws
        curves[name] = fn(wave_aa, dust_bump_strength=1.0, dust_delta=0.0)
    elif name == "power_law":
        curves[name] = fn(wave_aa, n_slope=-0.7)
    elif name == "cardelli":
        curves[name] = fn(wave_aa, dust_Rv=3.1)
    else:
        curves[name] = fn(wave_aa)

# Plot
curve_colors = {
    "power_law": "#1f77b4",
    "calzetti": "#ff7f0e",
    "kriek_conroy": "#2ca02c",
    "smc": "#d62728",
    "cardelli": "#9467bd",
    "salim": "#8c564b",
}
curve_labels = {
    "power_law": r"Power law ($n=-0.7$)",
    "calzetti": "Calzetti+2000",
    "kriek_conroy": r"Kriek & Conroy ($E_b=1$)",
    "smc": "SMC (Gordon+2003)",
    "cardelli": r"Cardelli+1989 ($R_V=3.1$)",
    "salim": r"Salim+2018 ($E_b=1$)",
}

fig, ax = plt.subplots(figsize=(9, 5))
for name in DUST_LAWS:
    ax.plot(
        np.array(wave_aa) / 1e4,
        np.array(curves[name]),
        color=curve_colors[name],
        lw=1.8,
        label=curve_labels[name],
    )

# Mark the 2175A UV bump
ax.axvline(0.2175, color="0.6", ls=":", lw=1.0, zorder=0)
ax.annotate(
    r"2175 $\AA$ bump",
    xy=(0.2175, 0.55),
    xycoords=("data", "axes fraction"),
    fontsize=9,
    color="0.4",
    ha="center",
    rotation=90,
)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$k(\lambda)$ (relative attenuation)")
ax.set_xlim(0.1, 3.0)
ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Dust attenuation curves in diffsed")
fig.savefig("figures/13_attenuation_curves.pdf", bbox_inches="tight")
fig.savefig("figures/13_attenuation_curves.png", dpi=200, bbox_inches="tight")
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

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: transmission vs wavelength for young stars (age = 1 Myr)
ax = axes[0]
young_idx = 5  # ~0.3 Myr — deeply embedded in birth cloud
for f_obs, color in zip(f_obs_values, f_obs_colors):
    trans = two_component_dust(
        wave_aa,
        age_grid,
        tau_v1=1.5,
        tau_v2=0.5,
        law_bc="calzetti",
        law_diff="calzetti",
        f_obscuration=f_obs,
    )
    # Plot transmission for young stars
    ax.plot(
        np.array(wave_aa) / 1e4,
        np.array(trans[young_idx]),
        color=color,
        lw=1.8,
        label=rf"$f_{{\rm obs}}={f_obs}$",
    )

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Transmission $T(\lambda)$")
ax.set_xlim(0.1, 3.0)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.set_title("Young stars (age < 10 Myr)")

# Right: transmission vs wavelength for old stars (age = 5 Gyr)
ax = axes[1]
old_idx = -10  # ~5 Gyr — only diffuse ISM
for f_obs, color in zip(f_obs_values, f_obs_colors):
    trans = two_component_dust(
        wave_aa,
        age_grid,
        tau_v1=1.5,
        tau_v2=0.5,
        law_bc="calzetti",
        law_diff="calzetti",
        f_obscuration=f_obs,
    )
    ax.plot(
        np.array(wave_aa) / 1e4,
        np.array(trans[old_idx]),
        color=color,
        lw=1.8,
        label=rf"$f_{{\rm obs}}={f_obs}$",
    )

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Transmission $T(\lambda)$")
ax.set_xlim(0.1, 3.0)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.set_title("Old stars (age ~ 5 Gyr)")

fig.suptitle(
    r"Effect of $f_{\rm obscuration}$ (Lower+2022) on dust transmission",
    fontsize=13,
    y=1.02,
)
fig.tight_layout()
fig.savefig("figures/13_f_obscuration.pdf", bbox_inches="tight")
fig.savefig("figures/13_f_obscuration.png", dpi=200, bbox_inches="tight")
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
# diffsed lets you mix any pair of curves. Below we show the transmission
# matrix for an SMC birth cloud paired with a Cardelli (MW) diffuse ISM.

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

# Show 3 representative ages: young (0.3 Myr), transition (10 Myr), old (1 Gyr)
age_labels = [(5, "0.3 Myr (birth cloud)"), (45, "10 Myr (transition)"), (80, "1 Gyr (diffuse only)")]
line_styles = ["-", "--", ":"]

ax = axes[0]
for (idx, label), ls in zip(age_labels, line_styles):
    ax.plot(
        np.array(wave_aa) / 1e4,
        np.array(trans_mixed[idx]),
        ls=ls,
        lw=1.8,
        label=label,
    )
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Transmission $T(\lambda)$")
ax.set_xlim(0.1, 2.0)
ax.set_title("SMC birth cloud + Cardelli diffuse ISM")
ax.legend(fontsize=9)

ax = axes[1]
for (idx, label), ls in zip(age_labels, line_styles):
    ax.plot(
        np.array(wave_aa) / 1e4,
        np.array(trans_uniform[idx]),
        ls=ls,
        lw=1.8,
        label=label,
    )
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_xlim(0.1, 2.0)
ax.set_title("Calzetti birth cloud + Calzetti diffuse ISM")
ax.legend(fontsize=9)

fig.suptitle("Per-component dust law control", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/13_per_component_dust.pdf", bbox_inches="tight")
fig.savefig("figures/13_per_component_dust.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Dust Emission Models
#
# diffsed implements three dust emission models, all normalized by the
# energy-balance constraint $L_{\rm IR} = L_{\rm absorbed}$:
#
# | Model | Parameters | Description |
# |-------|-----------|-------------|
# | `modified_blackbody` | $T_{\rm dust}$, $\beta_{\rm IR}$ | Optically-thin greybody |
# | `dale2014` | $\alpha$ | 1-param template family |
# | `draine_li2007` | $U_{\rm min}$, $\gamma$, $q_{\rm PAH}$ | 3-param grain model (analytic) |
#
# Additionally, tabulated DL07 templates can be loaded from
# `data/dl07_templates.h5` for production work.
#
# Below we compare all three analytic models plus the tabulated DL07.
# We also show how $U_{\rm min}$ shifts the IR peak using the real
# DL07 templates.

# %%
# Wavelength grid from UV to FIR (0.1 to 1000 um)
wave_full = jnp.logspace(np.log10(1000.0), np.log10(1e7), 3000)  # Angstrom
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: compare all 4 models
ax = axes[0]
wave_um = np.array(wave_full) / 1e4

ax.loglog(wave_um, np.array(sed_mbb), lw=1.8, label=r"Modified BB ($T=35$ K)", color="#1f77b4")
ax.loglog(wave_um, np.array(sed_dale), lw=1.8, label=r"Dale+2014 ($\alpha=2$)", color="#ff7f0e")
ax.loglog(wave_um, np.array(sed_dl07_analytic), lw=1.8, label="DL07 analytic", color="#2ca02c")
ax.loglog(wave_um, np.array(sed_dl07_tab), lw=1.8, label="DL07 tabulated", color="#d62728", ls="--")

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ Hz$^{-1}$)")
ax.set_xlim(1, 1000)
ax.set_ylim(bottom=1e-5 * np.max(np.array(sed_mbb)))
ax.legend(fontsize=9, loc="upper left")
ax.set_title("Dust emission model comparison")

# Right: U_min variation with tabulated DL07
ax = axes[1]
umin_values = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0]
umin_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(umin_values)))

for umin, color in zip(umin_values, umin_colors):
    sed_u = dl07_tabulated(
        wave_full, L_absorbed, dust_umin=umin, dust_gamma_dl=0.01, dust_qpah=2.5
    )
    ax.loglog(wave_um, np.array(sed_u), lw=1.5, color=color, label=rf"$U_{{\min}}={umin}$")

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ Hz$^{-1}$)")
ax.set_xlim(1, 1000)
ax.set_ylim(bottom=1e-5 * np.max(np.array(sed_u)))
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_title(r"DL07 tabulated: $U_{\min}$ shifts the IR peak")

fig.tight_layout()
fig.savefig("figures/13_dust_emission_models.pdf", bbox_inches="tight")
fig.savefig("figures/13_dust_emission_models.png", dpi=200, bbox_inches="tight")
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
model = Model(spec, ssp_data, filters=filters)

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
from diffsed.models.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    interpolate_metallicity,
)
from diffsed.models.dust.emission import compute_absorbed_luminosity

# Use model internals to get SFR weights
p = model._get_internal_params(params)
sfr = model._compute_sfr(p)
sfr_on_ssp = jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)
weights = compute_csp_weights(sfr_on_ssp, model.ssp_ages_yr)

# Metallicity interpolation
from diffsed.model import LOG10_ZSUN

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
from diffsed.models.dust.emission import modified_blackbody

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
# Visualize the energy balance
fig, ax = plt.subplots(figsize=(9, 5))

wave_um = np.array(ssp_data.ssp_wave) / 1e4

ax.loglog(wave_um, np.array(sed_intrinsic), color="0.6", lw=1.0, label="Intrinsic stellar", alpha=0.7)
ax.loglog(wave_um, np.array(sed_attenuated), color="#1f77b4", lw=1.5, label="Attenuated stellar")
ax.loglog(wave_um, np.array(sed_ir), color="#d62728", lw=1.5, label="Dust emission (MBB)")

# Shade the absorbed region
ax.fill_between(
    wave_um,
    np.array(sed_attenuated),
    np.array(sed_intrinsic),
    where=np.array(sed_intrinsic) > np.array(sed_attenuated),
    color="#1f77b4",
    alpha=0.15,
    label=r"$L_{\rm absorbed}$",
)

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
ax.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
ax.set_xlim(0.09, 300)
ax.set_ylim(bottom=1e-5 * np.max(np.array(sed_intrinsic)))
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Energy balance verification")
fig.savefig("figures/13_energy_balance.pdf", bbox_inches="tight")
fig.savefig("figures/13_energy_balance.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Full Panchromatic SED: Stellar + Dust Attenuation + Dust Emission
#
# The complete forward model in diffsed:
#
# 1. Compute the intrinsic stellar SED from the SFH + metallicity + SSPs
# 2. Apply two-component dust attenuation (birth cloud + diffuse ISM)
# 3. Compute absorbed luminosity via energy balance
# 4. Add dust IR emission (scaled to $L_{\rm absorbed}$)
#
# Below we show this for three dust emission models, using the same
# stellar population and attenuation parameters.

# %%
# Full panchromatic SED with all three emission models
fig, ax = plt.subplots(figsize=(10, 6))

# Stellar components (same for all)
ax.loglog(wave_um, np.array(sed_intrinsic), color="0.7", lw=0.8,
          label="Intrinsic stellar", alpha=0.5, zorder=1)
ax.loglog(wave_um, np.array(sed_attenuated), color="0.3", lw=1.5,
          label="Attenuated stellar", zorder=2)

# Three emission models
emission_configs = [
    ("modified_blackbody", dict(dust_T=35.0, dust_beta_ir=1.8), "#d62728", r"MBB ($T=35$ K)"),
    ("dale2014", dict(dust_alpha_dale=2.0), "#ff7f0e", r"Dale+2014 ($\alpha=2$)"),
    ("draine_li2007", dict(dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5), "#2ca02c", "DL07 analytic"),
]

for model_name, kw, color, label in emission_configs:
    ir_sed = DUST_EMISSION_MODELS[model_name](ssp_data.ssp_wave, L_abs, **kw)
    total = np.array(sed_attenuated) + np.array(ir_sed)
    ax.loglog(wave_um, total, lw=1.8, color=color, label=f"Total ({label})", zorder=3)

# Also show DL07 tabulated total
ir_tab = dl07_tabulated(ssp_data.ssp_wave, L_abs, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
total_tab = np.array(sed_attenuated) + np.array(ir_tab)
ax.loglog(wave_um, total_tab, lw=1.8, color="#9467bd", ls="--",
          label="Total (DL07 tabulated)", zorder=3)

# Mark key wavelength regions
ax.axvspan(0.09, 0.3, alpha=0.04, color="blue", zorder=0)
ax.axvspan(8, 1000, alpha=0.04, color="red", zorder=0)
ax.text(0.14, 0.02, "UV", transform=ax.get_xaxis_transform(), fontsize=9, color="0.5", ha="center")
ax.text(60, 0.02, "FIR", transform=ax.get_xaxis_transform(), fontsize=9, color="0.5", ha="center")

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
ax.set_xlim(0.09, 300)
ax.set_ylim(bottom=1e-5 * np.max(np.array(sed_intrinsic)))
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_title("Full panchromatic SED: stellar + dust attenuation + dust emission")
fig.savefig("figures/13_panchromatic_sed.pdf", bbox_inches="tight")
fig.savefig("figures/13_panchromatic_sed.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook demonstrated diffsed's full dust framework:
#
# 1. **6 attenuation curves** with distinct UV behaviour (bump vs no bump)
# 2. **f\_obscuration** for clumpy dust geometry (Lower+2022)
# 3. **Per-component control**: different laws for birth cloud vs diffuse ISM
# 4. **3 emission models** + tabulated DL07 templates, all energy-balanced
# 5. **Energy balance** verification: $L_{\rm IR} = L_{\rm absorbed}$ to
#    machine precision
# 6. **Full panchromatic SEDs** from UV to FIR
#
# Every function is pure JAX --- JIT-compilable and fully differentiable
# for gradient-based inference.
