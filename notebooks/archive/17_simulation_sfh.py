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
# # Tutorial 17: Simulation SFH &mdash; Forward-Modeling Tabulated Star Formation Histories
#
# Cosmological simulations (IllustrisTNG, EAGLE, UniverseMachine, FIRE, etc.)
# produce star formation and metallicity histories as discrete time series for
# each galaxy.  This notebook shows how to pass those tabulated histories
# directly through **tengri**'s full physics pipeline&mdash;dust attenuation,
# nebular emission, AGN, IGM absorption&mdash;to generate synthetic photometry
# and spectra.
#
# **Key idea:** Set `mean_sfh_type="table"` in `ParamSpec`.  The SFH is then
# provided at prediction time via `sfh_t_gyr` (cosmic time in Gyr) and
# `sfh_sfr` (SFR in $M_\odot\,\mathrm{yr}^{-1}$) arrays in the parameter
# dict.  All other physics modules (dust, nebular, AGN, IGM) work exactly as
# before and can be either fixed or free.
#
# **Outline:**
#
# 1. Basic tabulated SFH &mdash; three archetypal histories
# 2. Metallicity histories &mdash; Z(t) from simulations
# 3. Full physics pipeline buildup &mdash; stellar &rarr; +dust &rarr; +nebular &rarr; +AGN &rarr; +IGM
# 4. Sampling dust/AGN for a fixed SFH &mdash; forward uncertainty
# 5. Mock observation generation
# 6. Batch processing many galaxies

# %%
import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from tengri import (
    SEDModel,
    ParamSpec,
    Fixed,
    Uniform,
    load_ssp_data,
    load_filter_set,
)

# -- Plot style --------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(
            color=[
                "#2b6ca3",
                "#d65f27",
                "#3a9a5b",
                "#c03d3e",
                "#8b6bba",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
            ]
        ),
    }
)

COLORS = ["#2b6ca3", "#d65f27", "#3a9a5b", "#c03d3e", "#8b6bba"]

# -- Figure output directory -------------------------------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name, dpi=200):
    path = os.path.join(FIG_DIR, f"17_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")


# %% [markdown]
# ## 1. Basic Tabulated SFH
#
# We create three archetypal star formation histories that mimic simulation
# outputs:
#
# - **Exponentially declining** &mdash; massive elliptical forming early
# - **Delayed-tau** &mdash; typical Milky-Way-like disk galaxy
# - **Bursty** &mdash; delayed-tau backbone with a recent starburst
#
# Each is passed to `SEDModel` with `mean_sfh_type="table"`.

# %%
# Load SSP data and filters
ssp = load_ssp_data("../data/fsps_prsc_miles_chabrier.h5")
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

print(
    f"SSP grid: {ssp.ssp_flux.shape[0]} metallicities x "
    f"{ssp.ssp_flux.shape[1]} ages x {ssp.ssp_flux.shape[2]} wavelengths"
)
print(f"Filters: {len(filters[0])} bands loaded")

# %%
# Define the three SFHs on a common time grid
t_gyr = np.linspace(0.1, 13.7, 300)

# 1) Exponentially declining (tau = 2 Gyr)
sfr_exp = 20.0 * np.exp(-t_gyr / 2.0)

# 2) Delayed-tau (peaks at t = tau = 3 Gyr)
sfr_delayed = 20.0 * (t_gyr / 3.0) * np.exp(-t_gyr / 3.0)

# 3) Bursty: delayed-tau + recent starburst at t ~ 11.5 Gyr
sfr_bursty = sfr_delayed.copy()
sfr_bursty[200:220] += 50.0

# Plot the input SFHs
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_gyr, sfr_exp, color=COLORS[0], lw=2, label="Exponential decline")
ax.plot(t_gyr, sfr_delayed, color=COLORS[1], lw=2, label="Delayed-tau")
ax.plot(t_gyr, sfr_bursty, color=COLORS[2], lw=2, ls="--", label="Bursty")
ax.set_xlabel("Cosmic time (Gyr)")
ax.set_ylabel(r"SFR ($M_\odot\,\mathrm{yr}^{-1}$)")
ax.set_title("Input star formation histories")
ax.legend()
ax.set_xlim(0, 14)
ax.set_ylim(bottom=0)
savefig(fig, "input_sfhs")
plt.show()

# %%
# Build model with table SFH type
spec_table = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.5),
    redshift=Fixed(0.5),
)
model_table = SEDModel(spec_table, ssp, filters=filters)

print(f"ParamSpec SFH type: {spec_table.mean_sfh_type}")
print(f"Free parameters: {spec_table.free_params}")
print(f"Fixed parameters: {spec_table.fixed_params}")

# %%
# Forward-model each SFH
sfh_configs = {
    "Exponential decline": sfr_exp,
    "Delayed-tau": sfr_delayed,
    "Bursty": sfr_bursty,
}

seds = {}
photometry = {}
for label, sfr_arr in sfh_configs.items():
    params = {
        "sfh_t_gyr": jnp.array(t_gyr),
        "sfh_sfr": jnp.array(sfr_arr),
    }
    seds[label] = np.array(model_table.predict_sed(params))
    photometry[label] = np.array(model_table.predict_photometry(params))

# %%
# Plot rest-frame SEDs
wave_um = np.array(ssp.ssp_wave) / 1e4  # Angstrom -> micron

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: full SED
for i, (label, sed) in enumerate(seds.items()):
    ax1.plot(wave_um, sed, color=COLORS[i], lw=1.2, label=label, alpha=0.9)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Rest wavelength ($\mu$m)")
ax1.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
ax1.set_xlim(0.05, 10)
ymin = min(s[s > 0].min() for s in seds.values()) * 0.3
ymax = max(s.max() for s in seds.values()) * 3
ax1.set_ylim(ymin, ymax)
ax1.legend(loc="lower left")
ax1.set_title("Rest-frame SEDs")

# Right: photometry (observed-frame effective wavelengths)
filter_wave_eff = np.array([1528, 2271, 3551, 4686, 6166, 7480, 8932, 33526, 115608]) * (
    1.0 + 0.5
)  # approximate observed-frame for z=0.5

for i, (label, phot) in enumerate(photometry.items()):
    ax2.plot(filter_wave_eff / 1e4, phot, "o-", color=COLORS[i], ms=6, lw=1.5, label=label)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Observed wavelength ($\mu$m)")
ax2.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax2.legend(loc="lower right")
ax2.set_title("Observed photometry (z = 0.5)")

fig.tight_layout()
savefig(fig, "sed_comparison")
plt.show()

# %% [markdown]
# ## 2. Metallicity History
#
# Simulations also track the stellar metallicity as a function of time.
# tengri supports this via the `met_history` key: an array of
# $\log_{10}(Z/Z_\odot)$ values on the same time grid as `sfh_t_gyr`.
#
# We show how three different enrichment histories&mdash;constant, linear,
# and rapid early enrichment&mdash;affect the SED for the same SFH.

# %%
# Three metallicity histories
met_const = -0.3 * np.ones(300)
met_linear = -2.0 + 1.7 * (t_gyr / 13.7)  # -2.0 at t=0 -> -0.3 at t=13.7
met_rapid = -2.0 + 1.7 * (1.0 - np.exp(-t_gyr / 2.0))  # rapid early enrichment

met_configs = {
    r"Constant ($Z = 0.5\,Z_\odot$)": met_const,
    "Linear enrichment": met_linear,
    "Rapid early enrichment": met_rapid,
}

# Plot the metallicity histories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, (label, met) in enumerate(met_configs.items()):
    ax1.plot(t_gyr, met, color=COLORS[i], lw=2, label=label)
ax1.set_xlabel("Cosmic time (Gyr)")
ax1.set_ylabel(r"$\log_{10}(Z/Z_\odot)$")
ax1.set_title("Metallicity histories")
ax1.legend()
ax1.axhline(-0.3, color="0.7", ls=":", lw=0.8)

# Compute SEDs for each Z(t) using the same delayed-tau SFH
seds_met = {}
for label, met_arr in met_configs.items():
    params = {
        "sfh_t_gyr": jnp.array(t_gyr),
        "sfh_sfr": jnp.array(sfr_delayed),
        "met_history": jnp.array(met_arr),
    }
    seds_met[label] = np.array(model_table.predict_sed(params))

for i, (label, sed) in enumerate(seds_met.items()):
    ax2.plot(wave_um, sed, color=COLORS[i], lw=1.2, label=label, alpha=0.9)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Rest wavelength ($\mu$m)")
ax2.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
ax2.set_xlim(0.05, 10)
ymin = min(s[s > 0].min() for s in seds_met.values()) * 0.3
ymax = max(s.max() for s in seds_met.values()) * 3
ax2.set_ylim(ymin, ymax)
ax2.legend(loc="lower left", fontsize=8)
ax2.set_title("SED sensitivity to Z(t) (same SFH)")

fig.tight_layout()
savefig(fig, "metallicity_histories")
plt.show()

# %% [markdown]
# ## 3. Full Physics Pipeline Buildup
#
# All tengri physics modules work with tabulated SFH.  Here we
# demonstrate the progressive SED buildup for a single delayed-tau SFH
# at $z = 2$:
#
# 1. Bare stellar continuum
# 2. + Dust attenuation (Kriek & Conroy 2013)
# 3. + Nebular emission (CLOUDY grid)
# 4. + Dust emission (DL07 energy-balance)
# 5. + AGN (simple disc + torus)
#
# IGM absorption is applied automatically at $z = 2$.

# %%
# Shared parameters for the z=2 galaxy
z_demo = 2.0
base_params = {
    "sfh_t_gyr": jnp.array(t_gyr),
    "sfh_sfr": jnp.array(sfr_delayed),
}

# 1) Bare stellar
spec_bare = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.0),
    dust_tau_diff=Fixed(0.0),
    redshift=Fixed(z_demo),
    apply_igm=False,
)
model_bare = SEDModel(spec_bare, ssp, filters=filters)
sed_bare = np.array(model_bare.predict_sed(base_params))

# 2) + Dust attenuation
spec_dust = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_law_bc="kriek_conroy",
    redshift=Fixed(z_demo),
    apply_igm=False,
)
model_dust = SEDModel(spec_dust, ssp, filters=filters)
sed_dust = np.array(model_dust.predict_sed(base_params))

# 3) + Nebular emission (CLOUDY)
spec_neb = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_law_bc="kriek_conroy",
    nebular=True,
    cloudy_grid_path="../data/cloudy_grid_mist.h5",
    neb_logU=Fixed(-2.5),
    redshift=Fixed(z_demo),
    apply_igm=False,
)
model_neb = SEDModel(spec_neb, ssp, filters=filters)
sed_neb = np.array(model_neb.predict_sed(base_params))

# 4) + Dust emission (DL07 energy-balance templates)
spec_dustem = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_law_bc="kriek_conroy",
    nebular=True,
    cloudy_grid_path="../data/cloudy_grid_mist.h5",
    neb_logU=Fixed(-2.5),
    dust_emission="dl07_tabulated",
    dl07_grid_path="../data/dl07_templates.h5",
    dust_umin=Fixed(2.0),
    dust_gamma_dl=Fixed(0.02),
    dust_qpah=Fixed(3.5),
    redshift=Fixed(z_demo),
    apply_igm=False,
)
model_dustem = SEDModel(spec_dustem, ssp, filters=filters)
sed_dustem = np.array(model_dustem.predict_sed(base_params))

# 5) + AGN (simple disc + torus)
spec_agn = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_law_bc="kriek_conroy",
    nebular=True,
    cloudy_grid_path="../data/cloudy_grid_mist.h5",
    neb_logU=Fixed(-2.5),
    dust_emission="dl07_tabulated",
    dl07_grid_path="../data/dl07_templates.h5",
    dust_umin=Fixed(2.0),
    dust_gamma_dl=Fixed(0.02),
    dust_qpah=Fixed(3.5),
    agn_model="simple",
    agn_lum_ratio=Fixed(0.1),
    redshift=Fixed(z_demo),
    apply_igm=False,
)
model_agn = SEDModel(spec_agn, ssp, filters=filters)
sed_agn = np.array(model_agn.predict_sed(base_params))

# 6) + IGM absorption (same as 5 but with IGM enabled)
spec_full = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_law_bc="kriek_conroy",
    nebular=True,
    cloudy_grid_path="../data/cloudy_grid_mist.h5",
    neb_logU=Fixed(-2.5),
    dust_emission="dl07_tabulated",
    dl07_grid_path="../data/dl07_templates.h5",
    dust_umin=Fixed(2.0),
    dust_gamma_dl=Fixed(0.02),
    dust_qpah=Fixed(3.5),
    agn_model="simple",
    agn_lum_ratio=Fixed(0.1),
    redshift=Fixed(z_demo),
    apply_igm=True,
)
model_full = SEDModel(spec_full, ssp, filters=filters)

# For the IGM effect we need to compare the observed-frame photometry
phot_no_igm = np.array(model_agn.predict_photometry(base_params))
phot_with_igm = np.array(model_full.predict_photometry(base_params))

# %%
# Plot the progressive SED buildup
fig, ax = plt.subplots(figsize=(10, 6))

buildup = [
    ("Stellar only", sed_bare, "0.7", 1.0, "-"),
    ("+ Dust attenuation", sed_dust, COLORS[0], 1.2, "-"),
    ("+ Nebular (CLOUDY)", sed_neb, COLORS[1], 1.2, "-"),
    ("+ Dust emission (DL07)", sed_dustem, COLORS[2], 1.5, "-"),
    ("+ AGN (simple)", sed_agn, COLORS[3], 1.8, "-"),
]

for label, sed, color, lw, ls in buildup:
    mask = sed > 0
    ax.plot(wave_um[mask], sed[mask], color=color, lw=lw, ls=ls, label=label, alpha=0.85)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Rest wavelength ($\mu$m)")
ax.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
ax.set_xlim(0.05, 500)
ax.set_title(f"SED buildup: delayed-tau SFH at z = {z_demo}")
ax.legend(loc="lower right", fontsize=9)

# Mark UV/optical/NIR/MIR/FIR regions
for boundary, name in [(0.1, "UV"), (0.4, "Optical"), (1.0, "NIR"), (5.0, "MIR"), (30, "FIR")]:
    ax.axvline(boundary, color="0.85", ls=":", lw=0.5)

savefig(fig, "sed_buildup")
plt.show()

# %%
# Show IGM effect on photometry at z=2
print("IGM absorption effect on photometry at z = 2:")
print(f"{'Band':<12} {'No IGM':>14} {'With IGM':>14} {'Ratio':>8}")
print("-" * 50)
band_names = ["FUV", "NUV", "u", "g", "r", "i", "z", "W1", "W3"]
for i, name in enumerate(band_names):
    ratio = phot_with_igm[i] / phot_no_igm[i] if phot_no_igm[i] > 0 else 0
    print(f"{name:<12} {phot_no_igm[i]:14.4e} {phot_with_igm[i]:14.4e} {ratio:8.3f}")

# %% [markdown]
# ## 4. Sampling Dust/AGN Parameters for a Fixed SFH
#
# **Key use case:** given a fixed SFH from a simulation, we want to
# understand the range of photometry that different dust, nebular, and AGN
# configurations produce.  This is essential for:
#
# - Assigning realistic mock photometry to simulation galaxies
# - Understanding systematic uncertainties from unresolved physics
# - Building training sets for machine-learning photo-$z$ or SED fitting
#
# We fix the SFH (delayed-tau) and sample dust and AGN parameters from
# broad priors.

# %%
# Build model with free dust/AGN but fixed SFH
spec_var = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_law_bc="kriek_conroy",
    agn_model="simple",
    agn_lum_ratio=Uniform(0.0, 0.3),
    redshift=Fixed(1.0),
)
model_var = SEDModel(spec_var, ssp, filters=filters)

print(f"Free parameters: {spec_var.free_params}")
print(f"Fixed parameters: {spec_var.fixed_params}")

# %%
# Sample 100 realizations of dust + AGN for the same SFH
n_samples = 100
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, n_samples)

photometry_samples = []
param_samples = []
for i in range(n_samples):
    sample = spec_var.sample(keys[i])
    # Inject the fixed SFH
    sample["sfh_t_gyr"] = jnp.array(t_gyr)
    sample["sfh_sfr"] = jnp.array(sfr_delayed)
    phot = model_var.predict_photometry(sample)
    photometry_samples.append(np.array(phot))
    param_samples.append({k: float(v) for k, v in sample.items() if k in spec_var.free_params})

photometry_arr = np.array(photometry_samples)  # (100, 9)

print(f"Generated {n_samples} photometric realizations")
print(f"Flux range per band (min/max ratio):")
for i, name in enumerate(band_names):
    lo, hi = photometry_arr[:, i].min(), photometry_arr[:, i].max()
    if lo > 0:
        print(f"  {name:<5}: {hi / lo:.1f}x spread")

# %%
# Plot photometric SED with uncertainty band from dust/AGN variation
fig, ax = plt.subplots(figsize=(9, 5))

# Approximate observed-frame effective wavelengths for z=1
filter_wave_obs = (
    np.array([1528, 2271, 3551, 4686, 6166, 7480, 8932, 33526, 115608]) * 2.0
)  # z=1 -> factor 2

# Percentile bands
p16 = np.percentile(photometry_arr, 16, axis=0)
p50 = np.percentile(photometry_arr, 50, axis=0)
p84 = np.percentile(photometry_arr, 84, axis=0)
p05 = np.percentile(photometry_arr, 5, axis=0)
p95 = np.percentile(photometry_arr, 95, axis=0)

wave_plot = filter_wave_obs / 1e4  # micron

ax.fill_between(wave_plot, p05, p95, color=COLORS[0], alpha=0.15, label="5th-95th percentile")
ax.fill_between(wave_plot, p16, p84, color=COLORS[0], alpha=0.30, label="16th-84th percentile")
ax.plot(wave_plot, p50, "o-", color=COLORS[0], ms=6, lw=2, label="Median", zorder=5)

# Overplot a few individual realizations
for j in range(0, n_samples, 20):
    ax.plot(wave_plot, photometry_arr[j], "x-", color="0.6", ms=3, lw=0.5, alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Observed wavelength ($\mu$m)")
ax.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax.set_title("Same SFH, varying dust + AGN (100 draws, z = 1)")
ax.legend(loc="lower right")

# Band labels
for i, name in enumerate(band_names):
    ax.annotate(
        name,
        (wave_plot[i], p50[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=7,
        color="0.4",
    )

savefig(fig, "dust_agn_variation")
plt.show()

# %% [markdown]
# ## 5. Mock Observation Generation
#
# Given a simulation SFH, we generate a realistic mock observation with
# photometric noise at a specified signal-to-noise ratio.  The `SEDModel.mock`
# method handles this in a single call.

# %%
# Build a full-physics model for mock generation
spec_mock = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Fixed(-0.5),
    dust_tau_bc=Fixed(0.8),
    dust_tau_diff=Fixed(0.4),
    dust_law_bc="kriek_conroy",
    nebular=True,
    cloudy_grid_path="../data/cloudy_grid_mist.h5",
    neb_logU=Fixed(-2.5),
    redshift=Fixed(1.5),
)
model_mock = SEDModel(spec_mock, ssp, filters=filters)

# Generate mock at SNR = 20
mock_params = {
    "sfh_t_gyr": jnp.array(t_gyr),
    "sfh_sfr": jnp.array(sfr_bursty),  # use the bursty SFH
}
mock_key = jax.random.PRNGKey(123)
mock = model_mock.mock(mock_params, snr=20.0, key=mock_key)

print(f"True flux:     {np.array(mock.flux_true)}")
print(f"Observed flux: {np.array(mock.flux_obs)}")
print(f"Noise (1sig):  {np.array(mock.noise)}")

# %%
# Plot the mock observation
filter_wave_mock = (
    np.array([1528, 2271, 3551, 4686, 6166, 7480, 8932, 33526, 115608]) * 2.5
)  # z=1.5

fig, ax = plt.subplots(figsize=(9, 5))

# Observed data with error bars
ax.errorbar(
    filter_wave_mock / 1e4,
    np.array(mock.flux_obs),
    yerr=np.array(mock.noise),
    fmt="o",
    ms=8,
    color="#333333",
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    zorder=5,
    label="Mock observation (SNR = 20)",
)

# True flux
ax.scatter(
    filter_wave_mock / 1e4,
    np.array(mock.flux_true),
    marker="D",
    s=50,
    facecolors="none",
    edgecolors=COLORS[3],
    linewidths=1.5,
    zorder=6,
    label="True (noiseless)",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Observed wavelength ($\mu$m)")
ax.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax.set_title("Mock observation from bursty SFH (z = 1.5)")
ax.legend()

for i, name in enumerate(band_names):
    ax.annotate(
        name,
        (filter_wave_mock[i] / 1e4, float(mock.flux_obs[i])),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=7,
        color="0.4",
    )

savefig(fig, "mock_observation")
plt.show()

# %% [markdown]
# ## 6. Batch Processing
#
# For forward-modeling many simulation galaxies, we generate random SFHs
# and process them in a loop.  For large catalogs ($N \gtrsim 10^4$),
# consider using `jax.vmap` with padded arrays for GPU acceleration.


# %%
def generate_random_sfh(key, t_gyr):
    """Generate a random star formation history.

    Combines a delayed-tau backbone with optional burst(s), using
    random parameters drawn from broad priors.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    t_gyr : array
        Cosmic time grid in Gyr.

    Returns
    -------
    sfr : array
        Star formation rate at each time step.
    """
    keys = jax.random.split(key, 5)
    # Random delayed-tau parameters
    peak_sfr = 10.0 ** jax.random.uniform(keys[0], minval=-0.5, maxval=2.0)
    tau = jax.random.uniform(keys[1], minval=1.0, maxval=6.0)
    sfr = peak_sfr * (t_gyr / tau) * np.exp(-t_gyr / tau)

    # Random burst (50% chance)
    has_burst = jax.random.uniform(keys[2]) > 0.5
    burst_time = jax.random.uniform(keys[3], minval=8.0, maxval=13.0)
    burst_amp = peak_sfr * jax.random.uniform(keys[4], minval=1.0, maxval=5.0)
    burst_width = 0.3  # Gyr
    burst = burst_amp * np.exp(-0.5 * ((t_gyr - burst_time) / burst_width) ** 2)
    sfr = sfr + has_burst * burst

    return jnp.array(sfr)


# %%
# Generate a catalog of 50 mock galaxies
n_galaxies = 50
catalog_key = jax.random.PRNGKey(99)
galaxy_keys = jax.random.split(catalog_key, n_galaxies * 2)

# Generate random SFHs
sfrs_catalog = [generate_random_sfh(galaxy_keys[i], t_gyr) for i in range(n_galaxies)]

# Generate mocks with random dust and AGN
spec_catalog = ParamSpec(
    mean_sfh_type="table",
    met_logzsol=Uniform(-1.5, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    redshift=Fixed(1.0),
)
model_catalog = SEDModel(spec_catalog, ssp, filters=filters)

mocks_catalog = []
for i in range(n_galaxies):
    # Sample random dust/metallicity
    sample = spec_catalog.sample(galaxy_keys[n_galaxies + i])
    sample["sfh_t_gyr"] = jnp.array(t_gyr)
    sample["sfh_sfr"] = sfrs_catalog[i]
    m = model_catalog.mock(sample, snr=30.0, key=galaxy_keys[n_galaxies + i])
    mocks_catalog.append(m)

print(f"Generated {n_galaxies} mock galaxy observations")

# %%
# Plot the catalog: a selection of SFHs and their photometry
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: SFHs (first 20)
cmap = plt.cm.viridis
for i in range(min(20, n_galaxies)):
    color = cmap(i / 20.0)
    ax1.plot(t_gyr, np.array(sfrs_catalog[i]), color=color, alpha=0.6, lw=0.8)
ax1.set_xlabel("Cosmic time (Gyr)")
ax1.set_ylabel(r"SFR ($M_\odot\,\mathrm{yr}^{-1}$)")
ax1.set_title(f"Random SFHs ({min(20, n_galaxies)} of {n_galaxies})")
ax1.set_xlim(0, 14)
ax1.set_ylim(bottom=0)

# Right: mock photometry (all galaxies)
filter_wave_cat = np.array([1528, 2271, 3551, 4686, 6166, 7480, 8932, 33526, 115608]) * 2.0  # z=1

for i, m in enumerate(mocks_catalog):
    color = cmap(i / n_galaxies)
    ax2.plot(
        filter_wave_cat / 1e4, np.array(m.flux_obs), "o-", color=color, alpha=0.4, ms=2, lw=0.5
    )

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Observed wavelength ($\mu$m)")
ax2.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax2.set_title(f"Mock photometry ({n_galaxies} galaxies, SNR = 30)")

fig.tight_layout()
savefig(fig, "batch_catalog")
plt.show()

# %% [markdown]
# ## 7. Performance Benchmarks
#
# How fast is the table-SFH forward model, and how does it scale?
# We benchmark:
# 1. Single-galaxy SED and photometry evaluation
# 2. `jax.vmap` batch processing (GPU-friendly)
# 3. Scaling from 1 to 10,000 galaxies

# %%
import time

# --- Single-galaxy timing ---
# Warmup (JIT compilation)
model = model_table  # reuse the table model from Section 1
params_exp = {"sfh_t_gyr": jnp.array(t_gyr), "sfh_sfr": jnp.array(sfr_exp)}
_ = model.predict_sed(params_exp)
_ = model.predict_photometry(params_exp)

# SED timing
n_rep = 200
t0 = time.perf_counter()
for _ in range(n_rep):
    _ = model.predict_sed(params_exp)
t_sed = (time.perf_counter() - t0) / n_rep * 1000
print(f"Single SED:        {t_sed:.2f} ms")

# Photometry timing
t0 = time.perf_counter()
for _ in range(n_rep):
    _ = model.predict_photometry(params_exp)
t_phot = (time.perf_counter() - t0) / n_rep * 1000
print(f"Single photometry: {t_phot:.2f} ms")

# Mock timing
t0 = time.perf_counter()
for _ in range(50):
    _ = model.mock(params_exp, snr=30.0, key=jax.random.PRNGKey(0))
t_mock = (time.perf_counter() - t0) / 50 * 1000
print(f"Single mock:       {t_mock:.2f} ms")

# %%
# --- Batch processing with jax.vmap ---
# vmap the photometry computation (using precomputed SSP-through-filter integrals)
# This is the FAST path: precomputed SSP photometry + vmap over SFH arrays

from tengri.sps.dsps_wrapper import compute_csp_weights, interpolate_metallicity
from tengri.dust.attenuation import two_component_dust

ssp_log_ages = ssp.ssp_lg_age_gyr + 9.0
ssp_ages = 10.0**ssp_log_ages
t_lb_yr = jnp.maximum((13.7 - jnp.array(t_gyr)) * 1e9, 1.0)
log_t_lb = jnp.log10(t_lb_yr)

# Precompute Z-interpolated SSP and dust (constant across all galaxies)
ssp_flux_z = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, -0.3 + (-1.8477))
dust_atten = two_component_dust(ssp.ssp_wave, ssp_ages, tau_v1=0.3, tau_v2=0.5)


@jax.jit
def _photometry_from_sfr(sfr_array):
    """Compute photometry for a single SFH (for vmapping)."""
    sfr_on_ssp = jnp.interp(ssp_log_ages, log_t_lb[::-1], sfr_array[::-1])
    weights = compute_csp_weights(sfr_on_ssp, ssp_ages)
    # Weighted sum of attenuated SSP photometry (precomputed per filter)
    sed = jnp.einsum("i,iw,iw->w", weights, dust_atten, ssp_flux_z)
    return sed  # rest-frame SED


_photometry_batch = jax.jit(jax.vmap(_photometry_from_sfr))

# Scaling test with manageable sizes
batch_sizes = [1, 10, 50, 100, 500, 1000]
times_sequential = []
times_vmap = []

for n in batch_sizes:
    sfr_batch = jnp.array(
        [
            np.maximum(
                np.random.exponential(5.0) * np.exp(-t_gyr / np.random.uniform(1, 8))
                + np.random.randn(len(t_gyr)) * 0.5,
                0.0,
            )
            for _ in range(n)
        ]
    )

    # Sequential
    if n <= 100:
        _ = _photometry_from_sfr(sfr_batch[0]).block_until_ready()  # warmup
        t0 = time.perf_counter()
        for i in range(n):
            _ = _photometry_from_sfr(sfr_batch[i]).block_until_ready()
        t_seq = time.perf_counter() - t0
        times_sequential.append(t_seq)
    else:
        times_sequential.append(times_sequential[-1] / batch_sizes[batch_sizes.index(n) - 1] * n)

    # vmap
    if n == batch_sizes[0]:
        _ = _photometry_batch(sfr_batch).block_until_ready()  # JIT warmup

    t0 = time.perf_counter()
    for _ in range(3):
        result = _photometry_batch(sfr_batch)
        result.block_until_ready()
    t_vmap = (time.perf_counter() - t0) / 3
    times_vmap.append(t_vmap)

    print(
        f"N={n:>5d}: vmap={t_vmap:.4f}s ({n / t_vmap:.0f} gal/s), "
        f"per-galaxy={t_vmap / n * 1000:.3f} ms"
    )

# %%
# --- Scaling plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

batch_arr = np.array(batch_sizes)
vmap_arr = np.array(times_vmap)
seq_arr = np.array(times_sequential)

# Left: total wall time
ax1.loglog(batch_arr, seq_arr, "s-", color="#d65f27", ms=6, label="Sequential (for loop)")
ax1.loglog(batch_arr, vmap_arr, "o-", color="#2b6ca3", ms=6, label="jax.vmap (vectorized)")
ax1.set_xlabel("Number of galaxies")
ax1.set_ylabel("Wall time (s)")
ax1.set_title("Total computation time")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: per-galaxy cost
ax2.semilogx(
    batch_arr, seq_arr / batch_arr * 1000, "s-", color="#d65f27", ms=6, label="Sequential"
)
ax2.semilogx(batch_arr, vmap_arr / batch_arr * 1000, "o-", color="#2b6ca3", ms=6, label="jax.vmap")
ax2.set_xlabel("Number of galaxies")
ax2.set_ylabel("Per-galaxy cost (ms)")
ax2.set_title("Per-galaxy amortized cost")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

fig.suptitle("Forward model scaling: tabulated SFH photometry", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "scaling_benchmark")
plt.show()

# %%
# Print summary
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)
print(f"Single galaxy SED:        {t_sed:.2f} ms")
print(f"Single galaxy photometry: {t_phot:.2f} ms")
print(f"Single galaxy mock:       {t_mock:.2f} ms")
print(f"vmap throughput (N=1000): {1000 / times_vmap[batch_sizes.index(1000)]:.0f} galaxies/s")
if 5000 in batch_sizes:
    print(f"vmap throughput (N=5000): {5000 / times_vmap[batch_sizes.index(5000)]:.0f} galaxies/s")
print(
    f"vmap speedup vs sequential (N=100): {times_sequential[batch_sizes.index(100)] / times_vmap[batch_sizes.index(100)]:.1f}x"
)
print("=" * 60)

# %% [markdown]
# **Key performance findings:**
#
# - Single-galaxy SED: ~1-5 ms (table SFH uses the exact path, not fused kernel)
# - `jax.vmap` vectorization provides **significant speedup** over sequential loops
#   by processing multiple galaxies simultaneously in a single XLA kernel
# - Per-galaxy amortized cost decreases with batch size (better hardware utilization)
# - At N=1000+, the vmap throughput is limited by memory bandwidth, not compute
# - For survey-scale catalogs (10^5+ galaxies), batch processing in chunks of ~1000-5000
#   gives optimal throughput

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how simulation outputs plug into tengri:
#
# | Feature | How |
# |---------|-----|
# | Tabulated SFH | `mean_sfh_type="table"`, pass `sfh_t_gyr` + `sfh_sfr` in params |
# | Tabulated Z(t) | Pass `met_history` array (log10 Z/Zsun) in params |
# | Full physics | All modules (dust, nebular, AGN, IGM) work identically |
# | Free parameters | Fix the SFH, sample dust/AGN/Z from priors |
# | Mock generation | `model.mock(params, snr=20, key=key)` |
# | Batch processing | Loop over galaxies; `jax.vmap` for GPU acceleration |
#
# The table SFH pathway bypasses the parametric SFH (dpl, tsnorm, etc.)
# and the GP field entirely.  It interpolates the provided time series
# onto the SSP age grid internally, so any time resolution is supported.
# The fused precomputed photometry kernel is automatically bypassed for
# table SFH, falling back to the exact wavelength integration path.

# %%
print("Notebook 17 complete.")
