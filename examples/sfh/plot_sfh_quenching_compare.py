"""
SFH Quenching Scenarios: Constant, Exponential, Sharp, and Burst+Decay
=====================================================================

Compare four galaxy quenching scenarios: unquenched (constant SFH),
exponential decline, sharp truncation, and a recent burst on top of quenching.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Parameters, SEDModel, load_ssp_data
from tengri.analysis.plotting import setup_style

setup_style()


def _find_ssp():
    """Find SSP data file in standard locations."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

# Shared baseline
shared = dict(
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

# Scenario 1: Constant star formation (dpl with beta = 0)
spec1 = Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_alpha=Fixed(0.0),  # flat rising
    sfh_dpl_beta=Fixed(0.0),  # flat falling → constant
    sfh_dpl_tau_gyr=Fixed(3.0),
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    **shared,
)
model1 = SEDModel(spec1, ssp)

# Scenario 2: Exponential decline (dpl with steep beta)
spec2 = Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(2.0),  # exponential decline
    sfh_dpl_tau_gyr=Fixed(2.0),
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    **shared,
)
model2 = SEDModel(spec2, ssp)

# Scenario 3: Sharp truncation (dpl with steep alpha and beta)
spec3 = Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_alpha=Fixed(3.0),  # sharp rise
    sfh_dpl_beta=Fixed(3.0),  # sharp fall
    sfh_dpl_tau_gyr=Fixed(1.5),
    sfh_dpl_log_peak_sfr=Fixed(1.2),
    **shared,
)
model3 = SEDModel(spec3, ssp)

# Scenario 4: Recent burst (parametric with recent peak)
spec4 = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.2),  # recent
    sfh_tsnorm_width_gyr=Fixed(0.5),
    sfh_tsnorm_skew=Fixed(0.3),
    sfh_tsnorm_trunc=Fixed(2.0),
    **shared,
)
model4 = SEDModel(spec4, ssp)

# Evaluate SEDs at t=0 lookback
params_eval = {k: float(v.value) for k, v in shared.items()}

sed1 = model1.predict_rest_sed(
    {
        **params_eval,
        "sfh_dpl_alpha": 0.0,
        "sfh_dpl_beta": 0.0,
        "sfh_dpl_tau_gyr": 3.0,
        "sfh_dpl_log_peak_sfr": 1.0,
    }
).sed
sed2 = model2.predict_rest_sed(
    {
        **params_eval,
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 2.0,
        "sfh_dpl_tau_gyr": 2.0,
        "sfh_dpl_log_peak_sfr": 1.0,
    }
).sed
sed3 = model3.predict_rest_sed(
    {
        **params_eval,
        "sfh_dpl_alpha": 3.0,
        "sfh_dpl_beta": 3.0,
        "sfh_dpl_tau_gyr": 1.5,
        "sfh_dpl_log_peak_sfr": 1.2,
    }
).sed
sed4 = model4.predict_rest_sed(
    {
        **params_eval,
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 0.2,
        "sfh_tsnorm_width_gyr": 0.5,
        "sfh_tsnorm_skew": 0.3,
        "sfh_tsnorm_trunc": 2.0,
    }
).sed

wave = ssp.ssp_wave

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(
    wave, np.array(sed1), "k-", lw=2.0, label="Constant (α=0, β=0)"
)
ax.loglog(
    wave, np.array(sed2), "C1--", lw=2.0, label="Exponential decline (β=2)"
)
ax.loglog(
    wave, np.array(sed3), "C2:", lw=2.0, label="Sharp truncation (α=3, β=3)"
)
ax.loglog(
    wave, np.array(sed4), "C3-.", lw=2.0, label="Recent burst"
)

ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=11)
ax.set_title("SFH Quenching Scenarios: Impact on SED", fontsize=12)
ax.set_xlim(1000, 1e6)
ax.set_ylim(1e0, 1e7)
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.grid(True, alpha=0.2, which="both")

fig.tight_layout()
plt.savefig("plot_sfh_quenching_compare.png", dpi=150, bbox_inches="tight")
plt.show()
