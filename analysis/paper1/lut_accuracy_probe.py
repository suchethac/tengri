#!/usr/bin/env python
"""
One-command reproduction of Figure 3 Panel (b) accuracy measurement.

Control = pred.photometry() [exact]
Test = model.predict_photometry() [LUT]

Usage:
  PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python analysis/paper1/lut_accuracy_probe.py
"""

import os
import sys

os.chdir("/Users/suchethacooray/Projects/tengri/.claude/worktrees/paper1")
sys.path.insert(0, "src")
os.environ["JAX_PLATFORMS"] = "cpu"

import warnings

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

import tengri
from tengri import Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

# ============================================================================
# Configuration (per analysis/paper1/results/fig03_open_questions.md)
# ============================================================================

print("LUT Accuracy Probe — Reproducible Measurement\n")
print("=" * 130)

# Load SSP
print("Loading SSP: prsc_miles_chabrier_wNE (default via tengri.load_ssp())")
ssp = tengri.load_ssp("prsc_miles_chabrier_wNE", download=False)

# 12 filters
FILTERS = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
]

obs = Observation(photometry=Photometry.from_names(FILTERS))

# Fixed parameters at prior medians
PARAM_MEDIANS = {
    "sfh_tsnorm_log_total_mass": 10.0,
    "sfh_tsnorm_peak_lbt_gyr": 1.0,
    "sfh_tsnorm_width_gyr": 0.5,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 1.0,
    "dust_tau_bc": 1.0,
    "dust_tau_diff": 0.75,
    "dust_slope": -0.7,
}

print("\nBuilding model:")
print("  SFH: tsnorm (all params fixed at prior medians)")
print("  Dust: two-component Calzetti (tau_bc=1.0, tau_diff=0.75)")
print("  Nebular: OFF")
print("  Redshift: FREE, Uniform(0.01, 3.0)")
print("  WavePrecomp: n_z=250, z_min=0.01, z_max=3.0")

# Build model with fixed parameters
sed_model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    approx=WavePrecomp(n_z=250, z_min=0.01, z_max=3.0),
    sfh={
        "type": "tsnorm",
        "log_total_mass": Fixed(PARAM_MEDIANS["sfh_tsnorm_log_total_mass"]),
        "peak_lbt_gyr": Fixed(PARAM_MEDIANS["sfh_tsnorm_peak_lbt_gyr"]),
        "width_gyr": Fixed(PARAM_MEDIANS["sfh_tsnorm_width_gyr"]),
        "skew": Fixed(PARAM_MEDIANS["sfh_tsnorm_skew"]),
        "trunc": Fixed(PARAM_MEDIANS["sfh_tsnorm_trunc"]),
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "tau_bc": Fixed(PARAM_MEDIANS["dust_tau_bc"]),
        "tau_diff": Fixed(PARAM_MEDIANS["dust_tau_diff"]),
        "slope": Fixed(PARAM_MEDIANS["dust_slope"]),
    },
    neb={"type": "none"},
    met={"logzsol": Fixed(0.0)},
    redshift=Uniform(0.01, 3.0),
)

# Build n_subbands=32 model for comparison
sed_model_32 = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    approx=WavePrecomp(n_z=250, z_min=0.01, z_max=3.0, n_subbands=32),
    sfh={
        "type": "tsnorm",
        "log_total_mass": Fixed(PARAM_MEDIANS["sfh_tsnorm_log_total_mass"]),
        "peak_lbt_gyr": Fixed(PARAM_MEDIANS["sfh_tsnorm_peak_lbt_gyr"]),
        "width_gyr": Fixed(PARAM_MEDIANS["sfh_tsnorm_width_gyr"]),
        "skew": Fixed(PARAM_MEDIANS["sfh_tsnorm_skew"]),
        "trunc": Fixed(PARAM_MEDIANS["sfh_tsnorm_trunc"]),
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "tau_bc": Fixed(PARAM_MEDIANS["dust_tau_bc"]),
        "tau_diff": Fixed(PARAM_MEDIANS["dust_tau_diff"]),
        "slope": Fixed(PARAM_MEDIANS["dust_slope"]),
    },
    neb={"type": "none"},
    met={"logzsol": Fixed(0.0)},
    redshift=Uniform(0.01, 3.0),
)

# Sample one parameter draw (redshift will be overridden per z-value)
params_base = sed_model.spec.sample(jax.random.PRNGKey(42))

z_values = [0.5, 1.0, 1.5, 2.0, 3.0]

# Report data for comparison
REPORT_DATA = {
    0.5: {"galex_fuv": 0.085, "sdss_g": 0.020},
    1.0: {"galex_fuv": 1.503, "sdss_g": 0.013},
    1.5: {"galex_fuv": 10.410, "sdss_g": 0.005},
    2.0: {"galex_fuv": 32.960, "sdss_g": 0.024},
    3.0: {"galex_fuv": 83.174, "sdss_g": 1.383},
}

print("\n" + "=" * 130)
print("MEASUREMENT TABLE: My Numbers vs. 2026-08-17 Report")
print("=" * 130)
print("\nControl = pred.photometry() [exact]")
print("Test = model.predict_photometry() [LUT]\n")

print(
    f"{'z':>5} | {'band':>12} | {'My F_ex':>14} | {'Report F_ex':>14} | "
    f"{'My err%':>8} | {'Report%':>8} | {'Ratio':>8} | {'n_sub=32%':>10}"
)
print("-" * 130)

results = {}

for z in z_values:
    params_z = {**params_base, "redshift": float(z)}

    # Exact (control)
    pred = sed_model.predict(params_z)
    F_exact = np.array(pred.photometry())

    # LUT (test)
    F_lut = np.array(sed_model.predict_photometry(params_z))

    # n_subbands=32
    pred_32 = sed_model_32.predict(params_z)
    F_lut_32 = np.array(sed_model_32.predict_photometry(params_z))

    results[z] = {}

    for band in ["galex_fuv", "sdss_g"]:
        idx = FILTERS.index(band)
        f_ex = F_exact[idx]
        f_lu = F_lut[idx]
        f_lu_32 = F_lut_32[idx]

        # Error calculation
        if f_ex != 0:
            err_pct = abs(f_lu - f_ex) / abs(f_ex) * 100
            err_32_pct = abs(f_lu_32 - f_ex) / abs(f_ex) * 100
        else:
            err_pct = float("nan")
            err_32_pct = float("nan")

        report_err = REPORT_DATA[z][band]
        ratio = (
            err_pct / report_err if (not np.isnan(err_pct) and report_err > 0) else float("nan")
        )

        results[z][band] = {
            "f_exact": float(f_ex),
            "err_default": float(err_pct) if not np.isnan(err_pct) else None,
            "err_32": float(err_32_pct) if not np.isnan(err_32_pct) else None,
        }

        err_str = f"{err_pct:7.3f}%" if not np.isnan(err_pct) else "   NaN "
        err_32_str = f"{err_32_pct:9.3f}%" if not np.isnan(err_32_pct) else "     NaN "
        ratio_str = f"{ratio:7.2f}×" if not np.isnan(ratio) else "   NaN "

        print(
            f"{z:5.1f} | {band:>12} | {f_ex:14.4e} | {REPORT_DATA[z][band] * 1e-34 if band == 'galex_fuv' else REPORT_DATA[z][band] * 1e-31:>14.3e} | "
            f"{err_str} | {report_err:7.3f}% | {ratio_str} | {err_32_str}"
        )

print("\n" + "=" * 130)
print("CRITERION CHECK (z=1.5 galex_fuv)")
print("=" * 130)

z15_default = results[1.5]["galex_fuv"]["err_default"]
z15_32 = results[1.5]["galex_fuv"]["err_32"]
report_z15_default = 10.410
report_z15_32 = 1.150

print(f"\nMy measurement:  {z15_default:.3f}% (default)  →  {z15_32:.3f}% (n_sub=32)")
print(f"Report:          {report_z15_default:.3f}%  →  {report_z15_32:.3f}%")

ratio1 = z15_default / report_z15_default
reduction = z15_default / z15_32 if z15_32 > 0 else float("inf")

print(
    f"\nRatio to report (default): {ratio1:.2f}×  {'✓ within ~2×' if 0.5 <= ratio1 <= 2.0 else '✗'}"
)
print(
    f"Error reduction (default→n_sub=32): {reduction:.1f}×  {'✓ ~order of magnitude' if reduction > 5 else '✗'}"
)

if 0.5 <= ratio1 <= 2.0 and reduction > 5:
    print("\n✅ PANEL (B) IS BUILDABLE")
else:
    print("\n⚠️  PANEL (B) NOT YET BUILDABLE")

print("\n" + "=" * 130)
