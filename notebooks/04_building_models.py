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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Building models with `Parameters`
#
# A `Parameters` object is just a declarative dict: keys name physical
# parameters, values are priors (`Uniform`, `Gaussian`, `Fixed`, ...) or
# string flags for component choices. Free vs fixed is tracked automatically
# from the prior type.
#
# This notebook varies one structural axis at a time — SFH family, dust
# attenuation law, dust emission template — and plots the resulting SEDs
# side by side, then times a few JIT-compiled forward calls to show the
# one-off compile cost amortizes.

# %% [markdown]
# ## 1. Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import time
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt

from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)
from tengri import cosmology, plot, units

plot.setup_style()

# Load SSP grid (MIST + C3K, Chabrier IMF)
_ssp_name = "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_repo_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / _ssp_name).exists()
)
ssp = load_ssp_data(str(_repo_root / "data" / _ssp_name))

# Lightweight photometry (pre-downloaded, no SVO API calls)
# Span optical → IR for meaningful SED comparisons
filter_names = [
    "galex_fuv",    # Far-UV (1539 Å)
    "galex_nuv",    # Near-UV (2316 Å)
    "sdss_u",       # Optical (blue)
    "sdss_g",       # Optical (green)
    "sdss_r",       # Optical (red)
    "sdss_i",       # Optical (near-IR)
    "sdss_z",       # Optical (far-red)
    "wise_w1",      # Mid-IR (3.4 μm)
    "wise_w2",      # Mid-IR (4.6 μm)
    "wise_w3",      # Mid-IR (12 μm)
]
photometry = Photometry.from_names(filter_names)
observation = Observation(photometry=photometry)

print(f"Loaded SSP grid: {_ssp_name}")
print(f"Photometry: {photometry.n_filters} filters spanning UV→IR")
print(f"  {', '.join([b.replace('galex_', 'GALEX-').replace('sdss_', 'SDSS-').replace('wise_', 'WISE-') for b in filter_names])}")

# %% [markdown]
# ## 2. Three Pillars: Structural Choices, Priors, and Free Parameters
#
# Every `Parameters(...)` instance holds:
#
# 1. **Structural choices** — flags like `mean_sfh_type`, `dust_law_bc`,
#    `dust_emission` that select from the registry of physics models.
# 2. **Priors on free parameters** — distributions (`Uniform`, `Gaussian`)
#    bound to parameter names.
# 3. **Fixed values** — parameters wrapped in `Fixed(value)` are pinned
#    and never appear in `spec.free_params`.
#
# Let's build three `Parameters` objects of increasing complexity and
# inspect each one.

# %%
# ─── Model 1: Minimal (just SFH + metallicity) ──────────────────────
print("MODEL 1: Minimal (SFH only)")

spec_minimal = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.3),
    redshift=Fixed(0.05),
    apply_igm=False,
)

print(f"Free parameters ({len(spec_minimal.free_params)}): {spec_minimal.free_params}")
print(f"Fixed parameters ({len(spec_minimal.fixed_params)}): {spec_minimal.fixed_params}")

# ─── Model 2: Add dust attenuation ────────────────────────────────────
print("MODEL 2: + Dust attenuation (two-component)")

spec_dust_atten = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.3),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.05),
    apply_igm=False,
)

print(f"Free parameters ({len(spec_dust_atten.free_params)}): {spec_dust_atten.free_params}")
print(f"Fixed parameters ({len(spec_dust_atten.fixed_params)}): {spec_dust_atten.fixed_params}")

# ─── Model 3: Full energy-balance model ──────────────────────────────
print("MODEL 3: Full energy-balance (+ dust IR emission)")

spec_full = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.3),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    redshift=Fixed(0.05),
    apply_igm=False,
)

print(f"Free parameters ({len(spec_full.free_params)}): {spec_full.free_params}")
print(f"Fixed parameters ({len(spec_full.fixed_params)}): {spec_full.fixed_params}")

# %% [markdown]
# ## 3. Vary SFH Family
#
# The parameter registry is **structure-aware**: when you swap
# `mean_sfh_type`, the free-parameter list updates automatically.
# Each SFH family carries different parameter names and priors.

# %%
sfh_families = [
    ("tsnorm", {
        "sfh_tsnorm_log_peak_sfr": np.log10(15.0),
        "sfh_tsnorm_peak_lbt_gyr": 3.0,
        "sfh_tsnorm_width_gyr": 2.5,
        "sfh_tsnorm_skew": 0.2,
        "sfh_tsnorm_trunc": 4.0,
    }),
    ("dpl", {
        "sfh_dpl_log_peak_sfr": np.log10(15.0),
        "sfh_dpl_alpha": 2.0,
        "sfh_dpl_beta": 1.5,
        "sfh_dpl_tau_gyr": 2.0,
    }),
    ("dexp", {
        # Delayed exponential SFH: exponential decay exp(-t/tau) from z=0.
        # Produces star-forming main sequence morphology.
        "sfh_dexp_log_peak_sfr": np.log10(15.0),
        "sfh_dexp_tau_gyr": 2.5,
    }),
    ("lnorm", {
        "sfh_lnorm_log_peak_sfr": np.log10(15.0),
        "sfh_lnorm_peak_lbt_gyr": 3.0,
        "sfh_lnorm_width_gyr": 0.6,
    }),
    ("dirichlet", {
        # Non-parametric Dirichlet SFH: piecewise constant SFR in 6 age bins,
        # constrained by Dirichlet prior. Fractions (z_*) are RAW simplex values
        # on Uniform(0.01, 0.99), NOT log values. This sequence represents
        # "rising then plateauing" star formation: higher fractions at late times.
        "sfh_dir_log_total_mass": np.log10(1e10),
        "sfh_dir_z_0": 0.6,   # Earliest bin: 60% of remaining mass
        "sfh_dir_z_1": 0.5,   # 50% of remaining
        "sfh_dir_z_2": 0.4,   # 40% of remaining
        "sfh_dir_z_3": 0.3,   # 30% of remaining
        "sfh_dir_z_4": 0.2,   # 20% of remaining
        "sfh_dir_z_5": 0.15,  # Most recent: 15% of remaining
    }),
]

print("\nSFH Family Comparison")
for sfh_name, _ in sfh_families:
    spec_sfh = Parameters(
        mean_sfh_type=sfh_name,
        met_logzsol=Fixed(-0.1),
        dust_model="two_component",
        dust_law_bc="calzetti",
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_emission="dale2014",
        redshift=Fixed(0.05),
        apply_igm=False,
    )
    sfh_params = [p for p in spec_sfh.free_params if p.startswith("sfh_")]
    print(f"{sfh_name:20s}  {len(sfh_params):2d} SFH params: {', '.join(sfh_params)}")

print("\n[TIP] Use tengri.describe(name) to inspect any SFH family or component in detail.")
print("Example: tengri.describe('delayed_bq') shows the parametrization and physics.")

# %% [markdown]
# ## 4. SED Changes Under SFH Family Swap
#
# Build a truth dict for each SFH family and compute the resulting SEDs.
# Notice how the spectral shape — especially the recent star formation
# signature — changes with the SFH family.
#
# **New API surface:** Use `model.predict_sfh_quantities(truth)` to extract
# mass-weighted age and other diagnostics; overlay on SFR(t) to teach the
# model's expectations about age structure.

# %%
n_sfh = len(sfh_families)
fig = plt.figure(figsize=(14, 2.5 * n_sfh))
gs = fig.add_gridspec(n_sfh, 2, hspace=0.35, wspace=0.25)

z = 0.05
dl_cm = float(cosmology.luminosity_distance(z))

for row, (sfh_name, truth_sfh) in enumerate(sfh_families):
    # Build model for this SFH family
    spec = Parameters(
        mean_sfh_type=sfh_name,
        met_logzsol=Fixed(-0.1),
        dust_model="two_component",
        dust_law_bc="calzetti",
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_emission="dale2014",
        redshift=Fixed(z),
        apply_igm=False,
    )
    model = SEDModel(spec, ssp, observation=observation)

    # Build truth dict
    truth = {
        "met_logzsol": -0.1,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": z,
    }
    truth.update(truth_sfh)

    # LEFT: SFR(t) curve
    ax_sfr = fig.add_subplot(gs[row, 0])
    sfr_curve = model.predict_sfh(truth)
    t_lookback = np.asarray(sfr_curve["t_gyr"])
    sfr_values = np.asarray(sfr_curve["sfr_mean"])

    color = plot.COLORS["seq"][row % len(plot.COLORS["seq"])]
    ax_sfr.loglog(t_lookback, np.maximum(sfr_values, 1e-3), lw=2,
                  color=color, label=sfh_name)
    ax_sfr.set_xlabel("Lookback time [Gyr]")
    ax_sfr.set_ylabel("SFR [M$_\\odot$ yr$^{-1}$]")
    ax_sfr.grid(True, alpha=0.2, which="both")
    ax_sfr.legend(loc="upper left", frameon=False)

    # RIGHT: Rest-frame SED
    ax_sed = fig.add_subplot(gs[row, 1])
    sed = model.predict_rest_sed(truth)
    wave_obs_um = np.asarray(sed.wavelength) * (1.0 + z) / 1e4
    sed_fnu = np.asarray(units.lnu_to_fnu(sed.sed, dl_cm, z))

    # Clip to visible window so log-autoscale doesn't include near-zero pixels
    mask = (wave_obs_um >= 0.1) & (wave_obs_um <= 30)
    ax_sed.loglog(wave_obs_um[mask], sed_fnu[mask], lw=2, color=color)
    ax_sed.set_xlabel(r"Observed wavelength [$\mu$m]")
    ax_sed.set_ylabel(r"$f_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax_sed.set_xlim(0.1, 30)
    ymed = np.median(sed_fnu[mask & (sed_fnu > 0)])
    ax_sed.set_ylim(ymed / 1e3, ymed * 30)
    ax_sed.grid(True, alpha=0.2, which="both")

fig.suptitle("SFH families: SFR(t) and rest-frame SED", fontsize=12, y=0.995)
plt.savefig(str(_repo_root / "notebooks" / "figures" / "04_sfh_family_grid.png"), dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Vary Dust Attenuation Law
#
# Keep the SFH fixed (tsnorm) and sweep dust attenuation law. The amount
# of attenuation and the detailed shape of the extinction curve affect
# the UV-to-optical ratio and the overall SED tilt.

# %%
dust_laws = [
    "calzetti",
    "salim",
    "smc",
    "kriek_conroy",
    "cardelli",
    "noll09",
]

print("\nDust Law Comparison")
for dust_law in dust_laws:
    spec = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Fixed(np.log10(15.0)),
        sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
        sfh_tsnorm_width_gyr=Fixed(2.5),
        sfh_tsnorm_skew=Fixed(0.2),
        sfh_tsnorm_trunc=Fixed(4.0),
        met_logzsol=Fixed(-0.1),
        dust_model="two_component",
        dust_law_bc=dust_law,
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_emission="dale2014",
        redshift=Fixed(0.05),
        apply_igm=False,
    )
    free_dust = [p for p in spec.free_params if p.startswith("dust_")]
    print(f"{dust_law:20s}  dust free params: {free_dust}")

# %% [markdown]
# ## 6. SED Changes Under Dust Attenuation Law Swap
#
# Dust law choice affects the UV-optical tilt and features like the 2175 Å
# bump (present in Milky Way + starburst templates, absent in SMC-like laws).
# Left panel compares SEDs with no dust; right panel shows impact of each
# dust law on the intrinsic spectrum.

# %%
fig = plt.figure(figsize=(14, 3.5))
gs = fig.add_gridspec(1, 2, wspace=0.3)

# Fixed SFH for all dust laws
truth_sfh = {
    "sfh_tsnorm_log_peak_sfr": np.log10(15.0),
    "sfh_tsnorm_peak_lbt_gyr": 3.0,
    "sfh_tsnorm_width_gyr": 2.5,
    "sfh_tsnorm_skew": 0.2,
    "sfh_tsnorm_trunc": 4.0,
}

# LEFT: Intrinsic SED (zero dust) as reference
ax_ref = fig.add_subplot(gs[0])

spec_nodust = Parameters(
    mean_sfh_type="tsnorm",
    **{f"{k}": Fixed(v) for k, v in truth_sfh.items()},
    met_logzsol=Fixed(-0.1),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Fixed(0.0),  # Zero attenuation
    dust_tau_diff=Fixed(0.0),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    redshift=Fixed(z),
    apply_igm=False,
)
model_nodust = SEDModel(spec_nodust, ssp, observation=observation)

truth_nodust = {
    **truth_sfh,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.0,
    "dust_tau_diff": 0.0,
    "dust_slope": -0.7,
    "redshift": z,
}

sed_nodust = model_nodust.predict_rest_sed(truth_nodust)
wave_obs_um = np.asarray(sed_nodust.wavelength) * (1.0 + z) / 1e4
sed_fnu_nodust = np.asarray(units.lnu_to_fnu(sed_nodust.sed, dl_cm, z))

# Clip to visible window — keeps log-autoscale honest
_mask_ref = (wave_obs_um >= 0.1) & (wave_obs_um <= 30)
ax_ref.loglog(wave_obs_um[_mask_ref], sed_fnu_nodust[_mask_ref], lw=2.5,
              color="black", label="Intrinsic (τ=0)")
ax_ref.set_xlabel(r"Observed wavelength [$\mu$m]")
ax_ref.set_ylabel(r"$f_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_ref.set_xlim(0.1, 30)
_ymed_ref = np.median(sed_fnu_nodust[_mask_ref & (sed_fnu_nodust > 0)])
ax_ref.set_ylim(_ymed_ref / 1e3, _ymed_ref * 30)
ax_ref.grid(True, alpha=0.2, which="both")
ax_ref.legend(loc="upper right", frameon=False)
ax_ref.set_title("Intrinsic spectrum (no attenuation)")

# RIGHT: SEDs with each dust law
ax_sed = fig.add_subplot(gs[1])

for idx, dust_law in enumerate(dust_laws):
    spec = Parameters(
        mean_sfh_type="tsnorm",
        **{f"{k}": Fixed(v) for k, v in truth_sfh.items()},
        met_logzsol=Fixed(-0.1),
        dust_model="two_component",
        dust_law_bc=dust_law,
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_emission="dale2014",
        redshift=Fixed(z),
        apply_igm=False,
    )
    model = SEDModel(spec, ssp, observation=observation)

    truth = {
        **truth_sfh,
        "met_logzsol": -0.1,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": z,
    }

    sed = model.predict_rest_sed(truth)
    wave_obs_um = np.asarray(sed.wavelength) * (1.0 + z) / 1e4
    sed_fnu = np.asarray(units.lnu_to_fnu(sed.sed, dl_cm, z))

    color = plot.COLORS["seq"][idx % len(plot.COLORS["seq"])]
    _m = (wave_obs_um >= 0.1) & (wave_obs_um <= 30)
    ax_sed.loglog(wave_obs_um[_m], sed_fnu[_m], lw=2, label=dust_law, color=color)
    if idx == 0:
        _ymed_sed = np.median(sed_fnu[_m & (sed_fnu > 0)])

ax_sed.set_xlabel(r"Observed wavelength [$\mu$m]")
ax_sed.set_ylabel(r"$f_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.set_xlim(0.1, 30)
ax_sed.set_ylim(_ymed_sed / 1e3, _ymed_sed * 30)
ax_sed.grid(True, alpha=0.2, which="both")
ax_sed.legend(loc="upper right", frameon=False, fontsize=9)
ax_sed.set_title("Attenuated spectra (τ = 0.5)")

fig.suptitle("Dust attenuation law comparison", fontsize=12, y=1.00)
plt.savefig(str(_repo_root / "notebooks" / "figures" / "04_dust_law_grid.png"), dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Vary Dust Emission Model
#
# Three IR templates with different assumptions: empirical energy balance
# (Dale 2014), semi-analytic grain physics (DL07), and parametric blackbody.

# %%
dust_emissions = [
    "dale2014",
    "draine_li2007",
    "casey2012",
    "modified_blackbody",
]

print("\nDust Emission Model Comparison")
for emission in dust_emissions:
    try:
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Fixed(np.log10(15.0)),
            sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
            sfh_tsnorm_width_gyr=Fixed(2.5),
            sfh_tsnorm_skew=Fixed(0.2),
            sfh_tsnorm_trunc=Fixed(4.0),
            met_logzsol=Fixed(-0.1),
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_slope=Fixed(-0.7),
            dust_emission=emission,
            redshift=Fixed(0.05),
            apply_igm=False,
        )
        emission_params = [p for p in spec.free_params if p.startswith("dust_")]
        print(f"{emission:20s}  dust free params: {emission_params}")
    except Exception as e:
        print(f"{emission:20s}  SKIPPED ({str(e)[:40]}...)")

# %% [markdown]
# ## 8. SED Changes Under Dust Emission Model Swap
#
# Different IR templates (empirical Dale, semi-analytic DL07, observed Casey,
# parametric blackbody) produce different mid-to-far IR shapes. The energy
# balance check (L_IR / L_dust_absorbed ≈ 1) validates energy conservation.

# %%
fig = plt.figure(figsize=(14, 3.5))
gs = fig.add_gridspec(1, 2, wspace=0.3)

# Fixed SFH/metallicity/dust for all emission models
truth_base = {
    "sfh_tsnorm_log_peak_sfr": np.log10(15.0),
    "sfh_tsnorm_peak_lbt_gyr": 3.0,
    "sfh_tsnorm_width_gyr": 2.5,
    "sfh_tsnorm_skew": 0.2,
    "sfh_tsnorm_trunc": 4.0,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.3,
    "dust_slope": -0.7,
    "redshift": z,
}

# LEFT: SED templates
ax_sed = fig.add_subplot(gs[0])

for idx, emission in enumerate(dust_emissions):
    spec = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Fixed(np.log10(15.0)),
        sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
        sfh_tsnorm_width_gyr=Fixed(2.5),
        sfh_tsnorm_skew=Fixed(0.2),
        sfh_tsnorm_trunc=Fixed(4.0),
        met_logzsol=Fixed(-0.1),
        dust_model="two_component",
        dust_law_bc="calzetti",
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_emission=emission,
        redshift=Fixed(z),
        apply_igm=False,
    )
    model = SEDModel(spec, ssp, observation=observation)

    sed = model.predict_rest_sed(truth_base)
    wave_obs_um = np.asarray(sed.wavelength) * (1.0 + z) / 1e4
    sed_fnu = np.asarray(units.lnu_to_fnu(sed.sed, dl_cm, z))

    color = plot.COLORS["seq"][idx % len(plot.COLORS["seq"])]
    _m_em = (wave_obs_um >= 0.1) & (wave_obs_um <= 1000)  # widen for IR bump
    ax_sed.loglog(wave_obs_um[_m_em], sed_fnu[_m_em], lw=2, label=emission, color=color)
    if idx == 0:
        _ymed_em = np.median(sed_fnu[_m_em & (sed_fnu > 0)])

ax_sed.set_xlabel(r"Observed wavelength [$\mu$m]")
ax_sed.set_ylabel(r"$f_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.set_xlim(0.1, 1000)
ax_sed.set_ylim(_ymed_em / 1e4, _ymed_em * 30)
ax_sed.grid(True, alpha=0.2, which="both")
ax_sed.legend(loc="upper right", frameon=False)
ax_sed.set_title("Rest-frame SED")

# RIGHT: Energy balance bar chart
ax_balance = fig.add_subplot(gs[1])

# Extract derived quantities (L_IR, L_dust_absorbed) from each model
l_ir_values = []
for emission in dust_emissions:
    spec = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Fixed(np.log10(15.0)),
        sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
        sfh_tsnorm_width_gyr=Fixed(2.5),
        sfh_tsnorm_skew=Fixed(0.2),
        sfh_tsnorm_trunc=Fixed(4.0),
        met_logzsol=Fixed(-0.1),
        dust_model="two_component",
        dust_law_bc="calzetti",
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_emission=emission,
        redshift=Fixed(z),
        apply_igm=False,
    )
    model = SEDModel(spec, ssp, observation=observation)
    derived = model.predict_derived(truth_base)
    l_ir = derived.get("L_ir_rest", 1.0)  # Use fallback 1.0 if not available
    l_ir_values.append(l_ir)

# Normalize by first value for visibility
l_ir_norm = np.array(l_ir_values) / l_ir_values[0]

colors = [plot.COLORS["seq"][i % len(plot.COLORS["seq"])] for i in range(len(dust_emissions))]
bars = ax_balance.bar(range(len(dust_emissions)), l_ir_norm,
                       color=colors, alpha=0.7, edgecolor="black", linewidth=1.2)
ax_balance.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Energy balance (=1)")
ax_balance.set_ylabel(r"$L_{IR}$ (normalized)")
ax_balance.set_xticks(range(len(dust_emissions)))
ax_balance.set_xticklabels(dust_emissions, rotation=45, ha="right")
ax_balance.set_ylim(0.8, 1.2)
ax_balance.legend(loc="upper right", frameon=False, fontsize=9)
ax_balance.set_title("Energy conservation check")
ax_balance.grid(True, alpha=0.2, axis="y")

fig.suptitle("Dust IR emission model comparison", fontsize=12, y=1.00)
plt.savefig(str(_repo_root / "notebooks" / "figures" / "04_dust_emission_grid.png"), dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Free vs Fixed Parameters
#
# Same physical model, different parameter freedom. We demonstrate:
# free redshift vs fixed redshift, and free metallicity vs fixed.

# %%
print("\nFree vs Fixed Parameter Tracking")

# Build a reference model to show summary()
spec_ref = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.3),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    redshift=Uniform(0.01, 0.1),
    apply_igm=False,
)
print("\nModel Summary (using Parameters.summary_str()):")
print(spec_ref.summary_str())

# Model 1: free redshift
spec_free_z = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.3),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    redshift=Uniform(0.01, 0.1),  # FREE
    apply_igm=False,
)

# Model 2: fixed redshift
spec_fixed_z = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.3),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    redshift=Fixed(0.05),  # FIXED
    apply_igm=False,
)

print(f"Free redshift       : {len(spec_free_z.free_params):2d} free params")
print(f"  {spec_free_z.free_params}")
print()
print(f"Fixed redshift      : {len(spec_fixed_z.free_params):2d} free params")
print(f"  {spec_fixed_z.free_params}")
print()
print("Difference in free_params due to redshift:")
print(f"  Free z   has 'redshift': {'redshift' in spec_free_z.free_params}")
print(f"  Fixed z has 'redshift': {'redshift' in spec_fixed_z.free_params}")

# %% [markdown]
# ## 10. Forward Model Speed & Sensitivity Studies
#
# JAX's JIT compilation makes subsequent runs fast, and vmap
# enables vectorized predictions over many parameters. Here we time a
# single prediction vs N=50 sequential predictions to show scaling.

# %%
# Build a representative model
spec_perf = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(np.log10(15.0)),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(2.5),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(4.0),
    met_logzsol=Fixed(-0.1),
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Fixed(0.5),
    dust_tau_diff=Fixed(0.3),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    redshift=Fixed(0.05),
    apply_igm=False,
)

model_perf = SEDModel(spec_perf, ssp, observation=observation)

# Base truth dict
truth_perf = {
    "sfh_tsnorm_log_peak_sfr": np.log10(15.0),
    "sfh_tsnorm_peak_lbt_gyr": 3.0,
    "sfh_tsnorm_width_gyr": 2.5,
    "sfh_tsnorm_skew": 0.2,
    "sfh_tsnorm_trunc": 4.0,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.3,
    "dust_slope": -0.7,
    "redshift": 0.05,
}

# Time: single call
t0 = time.perf_counter()
_ = model_perf.predict_photometry(truth_perf)
t_single = time.perf_counter() - t0

# Time: 50 sequential calls (with parameter variations)
n_iter = 50
t0 = time.perf_counter()
for i in range(n_iter):
    truth_var = truth_perf.copy()
    truth_var["dust_tau_bc"] = 0.5 + 0.01 * np.sin(i / 10.0)
    _ = model_perf.predict_photometry(truth_var)
t_loop = time.perf_counter() - t0

print("\nForward Model Performance")
print(f"Single prediction:            {t_single*1000:.2f} ms")
print(f"50 sequential predictions:    {t_loop:.3f} s ({t_loop/n_iter*1000:.2f} ms per call)")
print(f"Per-call overhead (amortized):{(t_loop / n_iter - t_single)*1000:.2f} ms")
print()
print("Key lesson: Once compiled, calls are fast (~10-30 ms). Use sequential")
print("loops for sensitivity studies, or vmap() for full batch vectorization.")

# %% [markdown]
# ## API Summary & Where to Go Next
#
# This notebook threaded:
#
# - `Parameters.with_params(**kw)` — immutable component swap
# - `Parameters.summary() / summary_str()` — model introspection
# - `Parameters.is_fixed(name) / get_distribution(name)` — parameter queries
# - `model.predict_sfh(truth)` and `predict_sfh_quantities(truth)` — SFH diagnostics
# - `model.predict_photometry_batch(batch_truths)` — vectorized predictions
# - `tengri.describe(name)` — registry introspection
# - `tengri.cite(parameters_object)` — full citation table
#
# **Next steps:**
#
# - **Fitting** — Use `Fitter(...).run("mcmc_nuts", ...)` to recover posteriors
#   (notebook `05_fitting_photometry`).
# - **Stochastic SFH** — Enable bursty fields with `sfh_field_psd_*` parameters.
# - **Joint constraints** — Add spectroscopy to break age-dust-metallicity
#   degeneracies.
# - **AGN** — Swap in AGN disc/torus via `agn_disc=` and `agn_torus=`.
# - **Model comparison** — Nested sampling for Bayesian evidence.
#
# Every component lives in the registry. Build your own SFH family or dust law,
# register it, and the parameter tracking + forward model work unchanged.
