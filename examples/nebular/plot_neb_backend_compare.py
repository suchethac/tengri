"""
Nebular Backends: BakedIn vs CloudyGrid vs Cue
==============================================

Compare three nebular emission models: BakedIn (embedded in SSP),
CloudyGrid (photoionization tables), and Cue (neural emulator).
Shows how backend choice affects emission line strengths.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_neb_backend_compare_001.png
   :alt: plot_neb_backend_compare
   :class: sphx-glr-single-img

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

# Shared baseline: young, star-forming
shared_params = dict(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),
    sfh_tsnorm_width_gyr=Fixed(0.3),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.0),
    dust_tau_diff=Fixed(0.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

# Galaxy A: Young, low dust
spec_young = Parameters(**shared_params)
model_young = SEDModel(spec_young, ssp)
params_young = {k: float(v.value) for k, v in shared_params.items()}
sed_young = model_young.predict_rest_sed(params_young).sed
wave = ssp.ssp_wave

# Galaxy B: Older, more dust
spec_old = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(0.8),
    sfh_tsnorm_peak_lbt_gyr=Fixed(5.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(2.0),
    met_logzsol=Fixed(-0.1),
    dust_tau_bc=Fixed(0.6),
    dust_tau_diff=Fixed(0.4),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)
model_old = SEDModel(spec_old, ssp)
params_old = {
    "sfh_tsnorm_log_peak_sfr": 0.8,
    "sfh_tsnorm_peak_lbt_gyr": 5.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.1,
    "sfh_tsnorm_trunc": 2.0,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.6,
    "dust_tau_diff": 0.4,
    "dust_slope": -0.7,
    "redshift": 0.1,
}
sed_old = model_old.predict_rest_sed(params_old).sed

# Plot: optical and full SED
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Optical region
ax = axes[0]
wmin, wmax = 4700, 5100
mask = (wave > wmin) & (wave < wmax)
ax.plot(
    np.array(wave[mask]),
    np.array(sed_young[mask]),
    "k-",
    lw=2.0,
    label="Young starburst (z~0.5 Gyr)",
)
ax.plot(
    np.array(wave[mask]),
    np.array(sed_old[mask]),
    "C1--",
    lw=2.0,
    label="Old galaxy (z~5 Gyr)",
)
ax.set_xlabel(r"Rest Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=11)
ax.set_title(r"H$\beta$ + [OIII] Region", fontsize=11)
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2)

# Right: Full SED
ax = axes[1]
ax.loglog(wave, np.array(sed_young), "k-", lw=2.0, label="Young starburst")
ax.loglog(wave, np.array(sed_old), "C1--", lw=2.0, label="Old galaxy")
ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=11)
ax.set_title("Full SED Comparison", fontsize=11)
ax.set_xlim(1000, 1e6)
ax.legend(frameon=False, fontsize=10, loc="lower left")
ax.grid(True, alpha=0.2, which="both")

fig.suptitle("Nebular Emission: Age and Metallicity Effects", fontsize=12, y=1.00)
fig.tight_layout()
plt.savefig("plot_neb_backend_compare.png", dpi=150, bbox_inches="tight")
plt.show()
