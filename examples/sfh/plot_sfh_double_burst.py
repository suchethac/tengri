"""
SFH with Double Bursts: Old and Recent Star Formation
====================================================

Demonstrate a galaxy with two distinct star formation events:
an old burst + a recent burst. Show how the SED reflects
both young and old stellar populations.
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
    dust_tau_bc=Fixed(0.2),
    dust_tau_diff=Fixed(0.15),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

# Old burst only (10 Gyr ago)
spec_old = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(10.0),
    sfh_tsnorm_width_gyr=Fixed(0.5),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(2.0),
    **shared,
)
model_old = SEDModel(spec_old, ssp)

# Recent burst only (0.3 Gyr ago)
spec_recent = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.3),
    sfh_tsnorm_width_gyr=Fixed(0.5),
    sfh_tsnorm_skew=Fixed(0.3),
    sfh_tsnorm_trunc=Fixed(2.0),
    **shared,
)
model_recent = SEDModel(spec_recent, ssp)

# Double burst: weighted sum of old and recent
spec_double = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),  # representative peak
    sfh_tsnorm_width_gyr=Fixed(1.5),  # broad
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(2.5),
    **shared,
)
model_double = SEDModel(spec_double, ssp)

# Evaluate
params_eval = {k: float(v.value) for k, v in shared.items()}

sed_old = model_old.predict_rest_sed(
    {
        **params_eval,
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 10.0,
        "sfh_tsnorm_width_gyr": 0.5,
        "sfh_tsnorm_skew": 0.1,
        "sfh_tsnorm_trunc": 2.0,
    }
).sed
sed_recent = model_recent.predict_rest_sed(
    {
        **params_eval,
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 0.3,
        "sfh_tsnorm_width_gyr": 0.5,
        "sfh_tsnorm_skew": 0.3,
        "sfh_tsnorm_trunc": 2.0,
    }
).sed
sed_double = model_double.predict_rest_sed(
    {
        **params_eval,
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 2.0,
        "sfh_tsnorm_width_gyr": 1.5,
        "sfh_tsnorm_skew": 0.2,
        "sfh_tsnorm_trunc": 2.5,
    }
).sed

wave = ssp.ssp_wave

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Linear (optical to NIR)
ax1.plot(
    np.array(wave[(wave > 3000) & (wave < 3e4)]),
    np.array(sed_old[(wave > 3000) & (wave < 3e4)]),
    "C0-",
    lw=2.0,
    label="Old burst only (10 Gyr ago)",
)
ax1.plot(
    np.array(wave[(wave > 3000) & (wave < 3e4)]),
    np.array(sed_recent[(wave > 3000) & (wave < 3e4)]),
    "C1-",
    lw=2.0,
    label="Recent burst only (0.3 Gyr ago)",
)
ax1.plot(
    np.array(wave[(wave > 3000) & (wave < 3e4)]),
    np.array(sed_double[(wave > 3000) & (wave < 3e4)]),
    "k--",
    lw=2.0,
    label="Double burst (combined)",
)
ax1.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax1.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=11)
ax1.set_title("Linear Scale (Optical to NIR)", fontsize=11)
ax1.legend(fontsize=10, frameon=False)
ax1.grid(True, alpha=0.2)

# Right: Log-log (full SED)
ax2.loglog(wave, np.array(sed_old), "C0-", lw=2.0, label="Old burst only")
ax2.loglog(wave, np.array(sed_recent), "C1-", lw=2.0, label="Recent burst only")
ax2.loglog(wave, np.array(sed_double), "k--", lw=2.0, label="Double burst")
ax2.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax2.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=11)
ax2.set_title("Log Scale (Full SED)", fontsize=11)
ax2.set_xlim(1000, 1e6)
ax2.set_ylim(1e0, 1e7)
ax2.legend(fontsize=10, frameon=False, loc="lower left")
ax2.grid(True, alpha=0.2, which="both")

fig.suptitle("Double Burst: Old + Recent Star Formation", fontsize=12, y=1.00)
fig.tight_layout()
plt.savefig("plot_sfh_double_burst.png", dpi=150, bbox_inches="tight")
plt.show()
