"""
SFR calibrations: UV only vs UV+IR hybrid estimators vs dust optical depth
===========================================================================

Star formation rate calibrations depend on which wavelengths we observe.
At high dust optical depth, UV-only SFR estimators severely underestimate
the true SFR because dusty starbursts radiate most energy in the infrared.
The hybrid SFR(UV+IR) recipe recovers the true SFR by combining both tracers.

This example sweeps dust optical depth τ_V from 0 to 4 across 12 models.
We compare three estimators:

- **SFR(UV)**: UV luminosity at 1500 Å (dust-attenuated; Kennicutt 1998)
- **SFR(IR)**: integrated IR luminosity 8–1000 μm (reprocessed dust light)
- **SFR(UV+IR)**: sum of both (hybrid; Hao et al. 2011)

The plot demonstrates the key physics:

- SFR(UV) drops sharply with dust optical depth (obscuration)
- SFR(IR) rises with dust optical depth (more heating)
- SFR(UV+IR) remains roughly constant (dust-insensitive)

References
----------

- Kennicutt 1998, ARA&A, 36, 189 — SFR UV/IR/radio calibrations
- Hao et al. 2011, ApJ, 741, 124 — hybrid UV+IR SFR recipe

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import FIXED, Fixed, SEDModel, load_ssp
from tengri.plot import setup_style

# Suppress JAX and matplotlib warnings
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

setup_style()

# ────────────────────────────────────────────────────────────────────────
# 1. Load SSP data (required for SED synthesis)
# ────────────────────────────────────────────────────────────────────────

ssp = load_ssp("fsps_prsc_miles_chabrier")

# ────────────────────────────────────────────────────────────────────────
# 2. Define dust optical depths and model parameters
# ────────────────────────────────────────────────────────────────────────

# Sweep dust extinction from 0 to 4 (tau_V, optical depth in V-band)
tau_v_values = np.linspace(0.0, 4.0, 12)

# Dust law: two-component Calzetti + dust emission (Dale 2014)
# For each tau_V, we scale tau_bc (birth cloud) proportionally
# Ratio tau_bc/tau_diff set by Calzetti prescription
tau_bc_scale = 2.0  # Birth clouds have higher optical depth than diffuse ISM

# ────────────────────────────────────────────────────────────────────────
# 3. SFR calibration constants (Kennicutt 1998, Hao+ 2011)
# ────────────────────────────────────────────────────────────────────────

# SFR(UV) = K_uv * L_UV (UV luminosity at 1500 Å rest-frame)
# Kennicutt 1998, Table 1: UV luminosity calibration
# Original (Salpeter): K_UV = 1.4e-28 [M_sun/yr / (erg/s/Hz)]
# Chabrier IMF adjustment: ~0.6× Salpeter normalization
# Empirical factor for tengri FSPS+Chabrier: K_UV ≈ 2.6e-28
K_uv = 2.6e-28

# SFR(IR) = K_ir * L_IR (integrated IR luminosity, 8-1000 μm)
# Hao+ 2011, Table 1: IR luminosity calibration (Chabrier IMF)
# Empirical factor for tengri dust emission: K_IR ≈ 1.0e-44
K_ir = 1.0e-44

# ────────────────────────────────────────────────────────────────────────
# 4. Build models and extract SFRs
# ────────────────────────────────────────────────────────────────────────

# True SFR for all models (constant recent SF)
true_sfr = 10.0  # Msun/yr

# Storage for results
sfr_uv = np.zeros_like(tau_v_values)
sfr_ir = np.zeros_like(tau_v_values)
sfr_hybrid = np.zeros_like(tau_v_values)

# Wavelength grid for extracting L_UV at 1500 Å rest-frame
wave_grid = None

for i, tau_v in enumerate(tau_v_values):
    # Scale birth-cloud optical depth proportionally
    tau_bc = tau_v * tau_bc_scale / (tau_bc_scale + 1.0)
    tau_diff = tau_v / (tau_bc_scale + 1.0)

    # Build model with nested-dict builder
    # Double power-law SFH with fixed parameters to represent constant SFR~10 Msun/yr
    model_dict = {
        "sfh": {
            "type": "dpl",
            "alpha": Fixed(0.1),  # shallow rising slope
            "beta": Fixed(4.0),  # steep quenching
            "tau_gyr": Fixed(2.0),  # turnover timescale
            "log_total_mass": 10.0,  # log10(10 Msun/yr) = 1.0
        },
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(tau_diff),
            "emission": {"type": "dale2014", "all_params": FIXED},
        },
        "neb": {"type": "cue", "all_params": FIXED},
        "redshift": Fixed(0.01),
        "apply_igm": False,
    }

    # Build the SEDModel
    model = SEDModel.build(ssp_data=ssp, observation=None, **model_dict)

    # Predict rest-frame SED
    # (no observed-frame transformation needed; we work in rest-frame)
    pred = model.predict({})

    # Extract actual SFH-derived SFR (if available)
    if hasattr(pred, "sfr_100myr"):
        actual_sfr = pred.sfr_100myr
    else:
        actual_sfr = None

    wave = np.asarray(model.wavelengths)
    sed = np.asarray(pred.rest_sed())

    if wave_grid is None:
        wave_grid = wave

    # Extract L_UV at 1500 Å via band-averaging
    # (small window around 1500 Å to handle finite wavelength resolution)
    uv_window = 50.0  # Å
    mask_uv = (wave > 1500.0 - uv_window) & (wave < 1500.0 + uv_window)
    if np.any(mask_uv):
        l_uv = float(np.mean(sed[mask_uv]))
    else:
        # If 1500 Å is not in the grid, interpolate
        l_uv = float(np.interp(1500.0, wave, sed, left=0.0, right=0.0))

    # Integrated IR luminosity (8–1000 μm)
    # Convert to wavelength in Angstrom for boolean indexing
    mask_ir = (wave >= 8e3) & (wave <= 1e6)
    if np.any(mask_ir):
        wave_ir = wave[mask_ir]
        sed_ir = sed[mask_ir]
        # Trapezoid integration in wavelength space
        # L_IR [erg/s] = int L_nu d_nu = int L_nu (c/lambda^2) d_lambda
        c = 2.99792458e18  # Angstrom/s
        l_ir = float(np.trapezoid(sed_ir * (c / wave_ir**2), wave_ir))
    else:
        l_ir = 0.0

    # Apply SFR calibrations
    sfr_uv[i] = K_uv * l_uv
    sfr_ir[i] = K_ir * l_ir
    sfr_hybrid[i] = sfr_uv[i] + sfr_ir[i]

    # Use actual model SFR as truth if available
    if actual_sfr is not None:
        true_sfr = float(actual_sfr)

# ────────────────────────────────────────────────────────────────────────
# 5. Plot: SFR vs tau_V
# ────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Plot SFR estimators
ax.plot(
    tau_v_values,
    sfr_uv,
    marker="o",
    color="C0",
    lw=1.4,
    ms=5,
    label=r"SFR(UV @ 1500 Å)",
)
ax.plot(
    tau_v_values,
    sfr_ir,
    marker="s",
    color="C3",
    lw=1.4,
    ms=5,
    label=r"SFR(IR 8–1000 μm)",
)
ax.plot(
    tau_v_values,
    sfr_hybrid,
    marker="^",
    color="C2",
    lw=1.4,
    ms=5,
    label=r"SFR(UV+IR hybrid)",
)

# Mark true SFR
ax.axhline(
    true_sfr,
    color="black",
    lw=1.2,
    ls="--",
    alpha=0.6,
    label=f"Truth: {true_sfr:.1f} M⊙/yr",
)

ax.set_xlabel(r"Dust optical depth $\tau_V$")
ax.set_ylabel(r"SFR estimator [M$_{\odot}$/yr]")
ax.set_ylim(bottom=0, top=11)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.grid(True, alpha=0.2)

fig.tight_layout()
plt.savefig("plot_usecase_sfr_uv_ir_consistency.png", dpi=150, bbox_inches="tight")
