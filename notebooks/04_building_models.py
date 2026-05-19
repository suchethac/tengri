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
# # Building models with nested-dict API
#
# The nested-dict **model builder** provides a structured, Bagpipes-style
# interface for composing galaxy SED models. Instead of flat parameter lists,
# you organize physics into semantic groups (`sfh`, `dust`, `neb`, `agn`, etc.)
# and specify free/fixed status with sentinels.
#
# This notebook demonstrates:
#
# 1. **Three equivalent construction paths** — recipes, from_groups direct,
#    and round-trip edits — showing they all produce identical results.
# 2. **Parameter provenance** — the `summary()` method displays how each
#    parameter got its value: user-specified, wildcard-free, or registry default.
# 3. **Structural variation** — swapping SFH/dust families by editing nested dicts.
# 4. **Physical comparison** — sweeping SFH families, dust laws, and IR templates
#    to visualize their impact on the SED.

# %% [markdown]
# ## Setup

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
    recipes,
)
from tengri import cosmology, plot, units
from tengri.parameters.sentinels import FIXED, FREE

plot.setup_style()

# Load SSP grid: bare-stellar FSPS+MILES (required by Cue nebular backend
# used in the canonical recipes; wNE files cannot be paired with Cue).
_ssp_name = "fsps_prsc_miles_chabrier.h5"
_repo_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / _ssp_name).exists()
)
ssp = load_ssp_data(str(_repo_root / "data" / _ssp_name))

# Lightweight photometry (pre-downloaded, no SVO API calls)
# Span optical → IR for meaningful SED comparisons
filter_names = [
    "galex_fuv",  # Far-UV (1539 Å)
    "galex_nuv",  # Near-UV (2316 Å)
    "sdss_u",  # Optical (blue)
    "sdss_g",  # Optical (green)
    "sdss_r",  # Optical (red)
    "sdss_i",  # Optical (near-IR)
    "sdss_z",  # Optical (far-red)
    "wise_w1",  # Mid-IR (3.4 μm)
    "wise_w2",  # Mid-IR (4.6 μm)
    "wise_w3",  # Mid-IR (12 μm)
]
photometry = Photometry.from_names(filter_names)
observation = Observation(photometry=photometry)

print(f"Loaded SSP grid: {_ssp_name}")
print(f"Photometry: {photometry.n_filters} filters spanning UV→IR")
print(
    f"  {', '.join([b.replace('galex_', 'GALEX-').replace('sdss_', 'SDSS-').replace('wise_', 'WISE-') for b in filter_names])}"
)

# %% [markdown]
# ## Three construction paths (all equivalent)
#
# The nested-dict API offers three ways to build a model:
#
# 1. **Recipe** — curated template for a common scenario
# 2. **from_groups direct** — hand-built nested dict
# 3. **Round-trip** — extract from existing model, tweak, rebuild

# %%
# Path 1: Recipe (curated template)
print("PATH 1: Recipe")
recipe_dict = recipes.star_forming_photometry()
model1 = SEDModel.build(ssp_data=ssp, observation=observation, **recipe_dict)
print(f"  Model: {model1.spec.n_free} free params from recipe")
print(f"  SFH family: {recipe_dict['sfh']['type']}")
print()

# Path 2: From-groups direct (hand-built nested dict)
print("PATH 2: From-groups direct")
groups_dict = {
    "sfh": {"type": "dpl", "*": FREE, "logzsol": Fixed(-0.1)},
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FREE,
        "emission": {"type": "dale2014", "*": FIXED, "logzsol": Fixed(-0.1)},
    },
    "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
    "redshift": Uniform(0.01, 6.0),
    "apply_igm": True,
}
model2 = SEDModel.build(ssp_data=ssp, observation=observation, **groups_dict)
print(f"  Model: {model2.spec.n_free} free params from direct dict")
print(
    f"  Free params match recipe: {set(model1.spec.free_params) == set(model2.spec.free_params)}"
)
print()

# Path 3: Round-trip (extract → edit → rebuild)
print("PATH 3: Round-trip")
groups_from_model = model1.spec.to_groups()
# Tweak: change metallicity to free (lives inside the sfh group as 'logzsol')
groups_from_model.setdefault("sfh", {})["logzsol"] = FREE
model3 = SEDModel.build(ssp_data=ssp, observation=observation, **groups_from_model)
print(f"  Model: {model3.spec.n_free} free params from round-trip + edit")
print(f"  Added metallicity freedom: {'met_logzsol' in model3.spec.free_params}")

# %% [markdown]
# ## Parameter provenance and summary
#
# Use `model.spec.summary_str()` to inspect how each parameter got its value.
# The tags show the source:
#
# - `[user]` — explicitly specified in your nested dict
# - `[* FREE]` — matched by wildcard directive
# - `[* FIXED]` — matched by wildcard directive
# - `[default]` — registry default (usually fixed at median)

# %%
# Build a model with mixed provenance
base_groups = {
    "sfh": {
        "type": "tsnorm",
        "*": FREE,
        "skew": Uniform(-1.0, 1.0),
    },  # skew is [user], others are [* FREE]
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,  # All dust params are [* FIXED]
        "tau_bc": 0.5,  # Override to explicit value (still fixed)
    },
    "neb": {"type": "cue"},  # No wildcard → all use [default]
    "redshift": Fixed(0.05),
    "apply_igm": False,
}
spec = Parameters.from_groups(**base_groups)
print("Parameter Summary with Provenance Tags:")
print(spec.summary_str())

# %% [markdown]
# ## Vary the SFH family
#
# The nested-dict interface makes it simple to swap structural choices.
# Each SFH family carries different parameter names — the parser handles
# this automatically.
#
# Below we show how a single base dict + one-line edits capture the same
# physics as the old six Parameters(...) blocks.

# %%
sfh_families = [
    (
        "tsnorm",
        {
            "sfh_tsnorm_log_peak_sfr": np.log10(15.0),
            "sfh_tsnorm_peak_lbt_gyr": 3.0,
            "sfh_tsnorm_width_gyr": 2.5,
            "sfh_tsnorm_skew": 0.2,
            "sfh_tsnorm_trunc": 4.0,
        },
    ),
    (
        "dpl",
        {
            "sfh_dpl_log_peak_sfr": np.log10(15.0),
            "sfh_dpl_alpha": 2.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 2.0,
        },
    ),
    (
        "dexp",
        {
            "sfh_dexp_log_peak_sfr": np.log10(15.0),
            "sfh_dexp_tau_gyr": 2.5,
        },
    ),
    (
        "lnorm",
        {
            "sfh_lnorm_log_peak_sfr": np.log10(15.0),
            "sfh_lnorm_peak_lbt_gyr": 3.0,
            "sfh_lnorm_width_gyr": 0.6,
        },
    ),
    (
        "dirichlet",
        {
            "sfh_dir_log_total_mass": np.log10(1e10),
            "sfh_dir_z_0": 0.6,
            "sfh_dir_z_1": 0.5,
            "sfh_dir_z_2": 0.4,
            "sfh_dir_z_3": 0.3,
            "sfh_dir_z_4": 0.2,
            "sfh_dir_z_5": 0.15,
        },
    ),
]

print("\nSFH Family Comparison")
print("─" * 70)

# Base groups dict: reused for all SFH families, only type changes
base_groups_sfh = {
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "tau_bc": Fixed(0.5),
        "tau_diff": Fixed(0.3),
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014"},
    },
    "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
    "redshift": Fixed(0.05),
    "apply_igm": False,
}

for sfh_name, _ in sfh_families:
    # Swap SFH family: one-line edit
    groups_variant = base_groups_sfh.copy()
    groups_variant["sfh"] = {"type": sfh_name, "*": FIXED, "logzsol": Fixed(-0.1)}

    spec_sfh = Parameters.from_groups(**groups_variant)
    sfh_params = [p for p in spec_sfh.free_params if p.startswith("sfh_")]
    print(f"{sfh_name:20s}  {len(sfh_params):2d} SFH params (all fixed for this demo)")

print()
print("[TIP] Use tengri.describe(name) to inspect any SFH family or component in detail.")
print("Example: tengri.describe('dpl') shows the parametrization and physics.")

# %% [markdown]
# ## SED under different SFH families
#
# Build a truth dict for each SFH family and compute the resulting SEDs.
# Notice how the spectral shape — especially the recent star formation
# signature — changes with the SFH family.

# %%
n_sfh = len(sfh_families)
fig = plt.figure(figsize=(14, 2.5 * n_sfh))
gs = fig.add_gridspec(n_sfh, 2, hspace=0.35, wspace=0.25)

z = 0.05
dl_cm = float(cosmology.luminosity_distance(z))

for row, (sfh_name, truth_sfh) in enumerate(sfh_families):
    # Build model for this SFH family
    groups_sfh_fig = {
        "sfh": {"type": sfh_name, "*": FIXED, "logzsol": Fixed(-0.1)},
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "slope": Fixed(-0.7),
            "emission": {"type": "dale2014"},
        },
        "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
        "redshift": Fixed(z),
        "apply_igm": False,
    }
    spec = Parameters.from_groups(**groups_sfh_fig)
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
    ax_sfr.loglog(t_lookback, np.maximum(sfr_values, 1e-3), lw=2, color=color, label=sfh_name)
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
plt.savefig(
    str(_repo_root / "notebooks" / "figures" / "04_sfh_family_grid.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Vary the dust attenuation law
#
# Keep the SFH fixed (tsnorm) and sweep dust attenuation law. The amount
# of attenuation and the detailed shape of the extinction curve affect
# the UV-to-optical ratio and the overall SED tilt.
#
# This is where the nested-dict approach shines: swap the `law_bc` value,
# and the parser automatically re-declares the relevant dust parameters.

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
print("─" * 70)

# Base groups dict: dust law is the only thing that changes
base_groups_dust = {
    "sfh": {
        "type": "tsnorm",
        "log_peak_sfr": Fixed(np.log10(15.0)),
        "peak_lbt_gyr": Fixed(3.0),
        "width_gyr": Fixed(2.5),
        "skew": Fixed(0.2),
        "trunc": Fixed(4.0),
    },
    "dust": {
        "type": "two_component",
        "tau_bc": Fixed(0.5),
        "tau_diff": Fixed(0.3),
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014"},
    },
    "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
    "redshift": Fixed(0.05),
    "apply_igm": False,
}

for dust_law in dust_laws:
    # Swap dust law: one-line edit
    groups_dust_var = base_groups_dust.copy()
    groups_dust_var["dust"] = base_groups_dust["dust"].copy()
    groups_dust_var["dust"]["law_bc"] = dust_law

    spec = Parameters.from_groups(**groups_dust_var)
    free_dust = [p for p in spec.free_params if p.startswith("dust_")]
    print(f"{dust_law:20s}  dust free params: {free_dust}")

# %% [markdown]
# ## SED under different attenuation laws
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

# Base groups for no-dust comparison
groups_nodust = {
    "sfh": {
        "type": "tsnorm",
        "log_peak_sfr": Fixed(np.log10(15.0)),
        "peak_lbt_gyr": Fixed(3.0),
        "width_gyr": Fixed(2.5),
        "skew": Fixed(0.2),
        "trunc": Fixed(4.0),
    },
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014"},
    },
    "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
    "redshift": Fixed(z),
    "apply_igm": False,
}
spec_nodust = Parameters.from_groups(**groups_nodust)
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
ax_ref.loglog(
    wave_obs_um[_mask_ref],
    sed_fnu_nodust[_mask_ref],
    lw=2.5,
    color="black",
    label="Intrinsic (τ=0)",
)
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

# Reuse base and swap law_bc
for idx, dust_law in enumerate(dust_laws):
    groups_dustlaw_fig = {
        "sfh": {
            "type": "tsnorm",
            "log_peak_sfr": Fixed(np.log10(15.0)),
            "peak_lbt_gyr": Fixed(3.0),
            "width_gyr": Fixed(2.5),
            "skew": Fixed(0.2),
            "trunc": Fixed(4.0),
        },
        "dust": {
            "type": "two_component",
            "law_bc": dust_law,
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "slope": Fixed(-0.7),
            "emission": {"type": "dale2014"},
        },
        "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
        "redshift": Fixed(z),
        "apply_igm": False,
    }
    spec = Parameters.from_groups(**groups_dustlaw_fig)
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
plt.savefig(
    str(_repo_root / "notebooks" / "figures" / "04_dust_law_grid.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Vary the dust emission model
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
print("─" * 70)

# Base groups dict: emission type is the only thing that changes
base_groups_emission = {
    "sfh": {
        "type": "tsnorm",
        "log_peak_sfr": Fixed(np.log10(15.0)),
        "peak_lbt_gyr": Fixed(3.0),
        "width_gyr": Fixed(2.5),
        "skew": Fixed(0.2),
        "trunc": Fixed(4.0),
    },
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "tau_bc": Fixed(0.5),
        "tau_diff": Fixed(0.3),
        "slope": Fixed(-0.7),
    },
    "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
    "redshift": Fixed(0.05),
    "apply_igm": False,
}

for emission in dust_emissions:
    try:
        # Swap emission type: one-line edit
        groups_emission_var = base_groups_emission.copy()
        groups_emission_var["dust"] = base_groups_emission["dust"].copy()
        groups_emission_var["dust"]["emission"] = {"type": emission}

        spec = Parameters.from_groups(**groups_emission_var)
        emission_params = [p for p in spec.free_params if p.startswith("dust_")]
        print(f"{emission:20s}  dust free params: {emission_params}")
    except Exception as e:
        print(f"{emission:20s}  SKIPPED ({str(e)[:40]}...)")

# %% [markdown]
# ## SED under different IR templates
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
    groups_emission_fig = {
        "sfh": {
            "type": "tsnorm",
            "log_peak_sfr": Fixed(np.log10(15.0)),
            "peak_lbt_gyr": Fixed(3.0),
            "width_gyr": Fixed(2.5),
            "skew": Fixed(0.2),
            "trunc": Fixed(4.0),
        },
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "slope": Fixed(-0.7),
            "emission": {"type": emission},
        },
        "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
        "redshift": Fixed(z),
        "apply_igm": False,
    }
    spec = Parameters.from_groups(**groups_emission_fig)
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
    groups_energy_fig = {
        "sfh": {
            "type": "tsnorm",
            "log_peak_sfr": Fixed(np.log10(15.0)),
            "peak_lbt_gyr": Fixed(3.0),
            "width_gyr": Fixed(2.5),
            "skew": Fixed(0.2),
            "trunc": Fixed(4.0),
        },
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "slope": Fixed(-0.7),
            "emission": {"type": emission},
        },
        "neb": {"type": "cue", "*": FIXED, "logzsol": Fixed(-0.1)},
        "redshift": Fixed(z),
        "apply_igm": False,
    }
    spec = Parameters.from_groups(**groups_energy_fig)
    model = SEDModel(spec, ssp, observation=observation)
    derived = model.predict_derived(truth_base)
    l_ir = derived.get("L_ir_rest", 1.0)  # Use fallback 1.0 if not available
    l_ir_values.append(l_ir)

# Normalize by first value for visibility
l_ir_norm = np.array(l_ir_values) / l_ir_values[0]

colors = [plot.COLORS["seq"][i % len(plot.COLORS["seq"])] for i in range(len(dust_emissions))]
bars = ax_balance.bar(
    range(len(dust_emissions)),
    l_ir_norm,
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1.2,
)
ax_balance.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Energy balance (=1)")
ax_balance.set_ylabel(r"$L_{IR}$ (normalized)")
ax_balance.set_xticks(range(len(dust_emissions)))
ax_balance.set_xticklabels(dust_emissions, rotation=45, ha="right")
ax_balance.set_ylim(0.8, 1.2)
ax_balance.legend(loc="upper right", frameon=False, fontsize=9)
ax_balance.set_title("Energy conservation check")
ax_balance.grid(True, alpha=0.2, axis="y")

fig.suptitle("Dust IR emission model comparison", fontsize=12, y=1.00)
plt.savefig(
    str(_repo_root / "notebooks" / "figures" / "04_dust_emission_grid.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Free vs fixed parameters
#
# Same physical model, different parameter freedom. We demonstrate how the
# nested-dict API tracks free/fixed status via wildcard directives.

# %%
print("\nFree vs Fixed Parameter Tracking")
print("─" * 70)

# Build a reference model to show summary()
groups_ref = {
    "sfh": {"type": "tsnorm", "*": FREE, "logzsol": Fixed(-0.1)},
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FREE,
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014", "*": FIXED, "logzsol": Fixed(-0.1)},
    },
    "redshift": Uniform(0.01, 0.1),
    "apply_igm": False,
}
spec_ref = Parameters.from_groups(**groups_ref)
print("\nModel Summary (using Parameters.summary_str()):")
print(spec_ref.summary_str())

# Model 1: free redshift
groups_free_z = {
    "sfh": {"type": "tsnorm", "*": FREE, "logzsol": Fixed(-0.1)},
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FREE,
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014", "*": FIXED, "logzsol": Fixed(-0.1)},
    },
    "redshift": Uniform(0.01, 0.1),  # FREE
    "apply_igm": False,
}
spec_free_z = Parameters.from_groups(**groups_free_z)

# Model 2: fixed redshift
groups_fixed_z = {
    "sfh": {"type": "tsnorm", "*": FREE, "logzsol": Fixed(-0.1)},
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FREE,
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014", "*": FIXED, "logzsol": Fixed(-0.1)},
    },
    "redshift": Fixed(0.05),  # FIXED
    "apply_igm": False,
}
spec_fixed_z = Parameters.from_groups(**groups_fixed_z)

print(f"\nFree redshift       : {len(spec_free_z.free_params):2d} free params")
print(f"  {spec_free_z.free_params}")
print()
print(f"Fixed redshift      : {len(spec_fixed_z.free_params):2d} free params")
print(f"  {spec_fixed_z.free_params}")
print()
print("Difference in free_params due to redshift:")
print(f"  Free z   has 'redshift': {'redshift' in spec_free_z.free_params}")
print(f"  Fixed z has 'redshift': {'redshift' in spec_fixed_z.free_params}")

# %% [markdown]
# ## Forward-model timing and sensitivity
#
# JAX's JIT compilation makes subsequent runs fast, and vmap
# enables vectorized predictions over many parameters. Here we time a
# single prediction vs N=50 sequential predictions to show scaling.

# %%
# Build a representative model
groups_perf = {
    "sfh": {
        "type": "tsnorm",
        "log_peak_sfr": Fixed(np.log10(15.0)),
        "peak_lbt_gyr": Fixed(3.0),
        "width_gyr": Fixed(2.5),
        "skew": Fixed(0.2),
        "trunc": Fixed(4.0),
    },
    "dust": {
        "type": "two_component",
        "law_bc": "calzetti",
        "tau_bc": Fixed(0.5),
        "tau_diff": Fixed(0.3),
        "slope": Fixed(-0.7),
        "emission": {"type": "dale2014"},
    },
    "redshift": Fixed(0.05),
    "apply_igm": False,
}
spec_perf = Parameters.from_groups(**groups_perf)
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
print("─" * 70)
print(f"Single prediction:            {t_single * 1000:.2f} ms")
print(f"50 sequential predictions:    {t_loop:.3f} s ({t_loop / n_iter * 1000:.2f} ms per call)")
print(f"Per-call overhead (amortized):{(t_loop / n_iter - t_single) * 1000:.2f} ms")
print()
print("Key lesson: Once compiled, calls are fast (~10-30 ms). Use sequential")
print("loops for sensitivity studies, or vmap() for full batch vectorization.")

# %% [markdown]
# ## Where to go next
#
# The nested-dict model builder is the recommended entry point for new models.
# The old flat Parameters(...) constructor remains available as an escape hatch.
#
# Key affordances:
# - `recipes.*()` curated templates for common scenarios
# - `SEDModel.build(..., filters=...)` one-liner to build and evaluate
# - `model.spec.to_groups()` extract structure for inspection/round-trip edits
# - `model.spec.summary_str()` provenance-tagged parameter listing
#
# Natural next steps: [`05_fitting_photometry`](05_fitting_photometry.py)
# runs a real fit and reads its posterior;
# [`06_fitting_spectroscopy`](06_fitting_spectroscopy.py) breaks age, dust,
# and metallicity degeneracies with a spectrum. Stochastic SFHs live
# under `sfh={'type': ['dpl', 'field'], ...}`; AGN under
# `agn={'disc': ..., 'torus': ...}`. Build your own component, register it,
# and the parameter tracking and forward model pick it up unchanged.
