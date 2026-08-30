#!/usr/bin/env python3
"""Figure 2 — Panchromatic rest-frame SED with component decomposition."""

import json
import os
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tengri
from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
)
from tengri.units import lnu_to_llambda

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load pre-downloaded SSP grid
print("Loading SSP grid...")
ssp_path = Path(__file__).parent.parent.parent / "data" / "fsps_prsc_miles_chabrier.h5"
ssp = tengri.load_ssp(ssp_path)

# Panchromatic filter set
filters = [
    "galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "ukidss_j", "ukidss_h", "ukidss_k", "irac_36", "irac_45", "irac_58", "irac_80",
    "mips_24", "mips_70", "mips_160", "herschel_100", "herschel_160",
]
obs = Observation(photometry=Photometry.from_names(filters))

# Build panchromatic model
print("Building model...")
config = {
    "sfh": {
        "type": "dpl", "all_params": FIXED,
        "log_total_mass": 10.48, "alpha": 0.9, "beta": 2.7, "tau_gyr": 13.2,
    },
    "dust_attenuation": {
        "type": "two_component", "law": "calzetti", "all_params": FIXED,
        "tau_bc": 0.8, "tau_diff": 0.3, "slope": -0.4,
    },
    "dust_emission": {"type": "dale2014_cigale", "all_params": FIXED, "alpha_dale": 2.2},
    "neb": {"type": "cue", "all_params": FIXED},
    "agn": {
        "type": "composable", "all_params": FIXED,
        "disc": {"type": "multicolor", "all_params": FIXED, "log_lbol": 10.5},
        "torus": {"type": "skirtor", "all_params": FIXED, "tau_skirtor": 5.0, "torus_frac": 0.5},
        "nlr": {"type": "analytic", "all_params": FIXED},
    },
    "radio": {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}, "all_params": FIXED},
    "xray": {"type": "simple", "all_params": FIXED},
    "redshift": Fixed(0.1),
    "igm": {"type": "none"},
}

model = SEDModel.build(ssp_data=ssp, observation=obs, **config)
print(model.summary())

# Generate prediction
print("Predicting...")
params = model.spec.sample(jax.random.PRNGKey(0))
state = model.predict_state(params)

wave_rest = np.asarray(state.wave)
wave_um = wave_rest / 1e4

# Extract components
def _get(key):
    arr = state.derived.get(key)
    return None if arr is None else np.asarray(arr)

lnu_age = _get("lnu_age")
sed_stellar_intrinsic = lnu_age.sum(axis=0) if lnu_age is not None else None
sed_stellar_attenuated = _get("sed_dust_attenuated")
sed_nebular = _get("sed_nebular")
sed_dust_emission = _get("sed_dust_ir")
sed_agn = _get("sed_agn")
sed_xray = _get("sed_xray")
sed_radio = _get("sed_radio")
sed_total = np.asarray(state.sed_intrinsic)

# Convert to νL_ν
def nu_lnu(lnu):
    if lnu is None:
        return None
    nu_lnu_erg = wave_rest * lnu_to_llambda(lnu, wave_rest)
    L_sun = 3.839e33
    return nu_lnu_erg / L_sun

# Plot
print("Plotting...")
fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"rest wavelength $\lambda$ [$\mu$m]", fontsize=10)
ax.set_ylabel(r"$\nu L_\nu$ [L$_\odot$]", fontsize=10)

components = [
    (sed_stellar_intrinsic, "Stellar (intrinsic)", "--", "0.55", 1.1),
    (sed_stellar_attenuated, "Stellar (attenuated)", "-", "#3a6ea5", 1.4),
    (sed_nebular, "Nebular (Cue)", "-", "#19b3c4", 1.0),
    (sed_dust_emission, "Dust emission (Dale 2014)", "-", "#e8920c", 1.4),
    (sed_agn, "AGN (disc + torus + NLR)", "-", "#d94f4f", 1.4),
    (sed_xray, "X-ray (corona + XRB)", "-.", "#8a5fbf", 1.4),
    (sed_radio, "Radio (SF + AGN)", "-", "#3aa653", 1.4),
]

legend_labels = []
for sed, label, ls, color, lw in components:
    y = nu_lnu(sed)
    if y is not None and np.any(np.isfinite(y)) and np.any(y > 0):
        mask = (y > 0) & np.isfinite(y)
        ax.plot(wave_um[mask], y[mask], label=label, linestyle=ls, color=color, linewidth=lw, zorder=3)
        legend_labels.append(label)

y_total = nu_lnu(sed_total)
if y_total is not None:
    mask_total = (y_total > 0) & np.isfinite(y_total)
    ax.plot(wave_um[mask_total], y_total[mask_total], label="Total", color="k", linewidth=2.5, zorder=4)
    legend_labels.append("Total")

ymax = np.nanmax(y_total[mask_total]) if np.any(mask_total) else 1e-5
# FIX #1: Set x-range to 1e-4 to 1e6 µm, y-range from ~1e4 to 3× peak
ax.set_xlim(1e-4, 1e6)
ax.set_ylim(1e4, ymax * 3)

# Shaded bands for wavelength regimes (keep the three regions)
bands = [
    (1e-4, 1.24e-3, "#ff6b6b"),       # X-ray
    (1.24e-2, 1e3, "#4ecdc4"),        # UV-opt-IR
    (1e3, 1e6, "#95e1d3"),            # Radio
]
for w0, w1, color in bands:
    ax.axvspan(max(w0, 1e-4), min(w1, 1e6), alpha=0.1, color=color, zorder=0)

# Landmarks (staggered to avoid overlap)
landmarks = [(0.0912, "Lyman\nlimit", 0.93), (0.3646, "Balmer\nbreak", 0.86)]
for lam_um, txt, y_frac in landmarks:
    if lam_um < wave_um.min() or lam_um > wave_um.max():
        continue
    ax.axvline(lam_um, color="0.5", linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(lam_um, y_frac, txt, transform=ax.get_xaxis_transform(),
            fontsize=7, ha="center", va="bottom", color="0.5")
# Parameter summary
_sfr = float(state.derived.get("sfr", 0.0))
_mstar = 10.0 ** float(state.derived.get("log_mstar", 10.0))
_log_agn_lbol = float(state.derived.get("agn_log_lbol", 10.5))

summary_text = (
    f"M★ = {_mstar:.2e} M☉\n"
    f"SFR = {_sfr:.1f} M☉ yr⁻¹\n"
    f"log L_bol = {_log_agn_lbol:.1f}"
)
ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=8,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", linewidth=0.5),
        family="monospace")

# FIX #3: Place legend below with proper spacing
fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=True,
           bbox_to_anchor=(0.5, 0.02), handlelength=1.2)
fig.subplots_adjust(bottom=0.28)

fig.savefig(OUTPUT_DIR / "fig02_panchromatic_sed.pdf", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "fig02_panchromatic_sed.png", dpi=150, bbox_inches="tight")
print(f"Saved figures to {OUTPUT_DIR}")

# FIX #2: Fix wavelength unit conversion in probe
def nu_lnu_at_wave(wave_um_target):
    """Interpolate νL_ν to a specific rest wavelength [µm]."""
    if not np.any((y_total > 0) & np.isfinite(y_total)):
        return np.nan
    mask = (y_total > 0) & np.isfinite(y_total)
    return float(np.interp(wave_um_target, wave_um[mask], y_total[mask]))

# Wavelengths for verification (all in µm)
wavelengths_um = {
    "1 keV (12.4 Å)": 12.4 / 1e4,           # Convert Å to µm
    "0.5 µm (5000 Å)": 0.5,
    "100 µm (1e6 Å)": 100.0,
    "1.4 GHz (2.14e8 Å)": 2.14e8 / 1e4,     # Convert Å to µm
}

nu_lnu_values = {}
for label, wave_um_val in wavelengths_um.items():
    val = nu_lnu_at_wave(wave_um_val)
    nu_lnu_values[label] = val
    print(f"{label}: {val:.3e} L☉")

# Save JSON
output_json = {
    "redshift": 0.1,
    "log_mstar": float(state.derived.get("log_mstar", 10.0)),
    "sfr": float(state.derived.get("sfr", 10.0)),
    "agn_log_lbol": _log_agn_lbol,
    "nu_lnu_lsun": nu_lnu_values,
    "legend_labels": legend_labels,
}

with open(RESULTS_DIR / "fig02_panchromatic_sed_data.json", "w") as f:
    json.dump(output_json, f, indent=2)
print(f"\nSaved data to {RESULTS_DIR / 'fig02_panchromatic_sed_data.json'}")
