"""
Filter Set Comparison
=====================

Compare filter coverage from three different photometric surveys on the same
mock galaxy SED — SDSS (optical ugriz), 2MASS (NIR JHKs), and HST (UV/optical
ACS). Demonstrates how filter placement controls which spectral features are
captured. Each panel overlays the filter throughputs (orange) on the same
underlying SED (blue).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    load_filter_set,
    load_ssp_data,
)
from tengri.analysis.plotting import setup_style

setup_style()


def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
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
_FILTER_DIR = next(
    (
        str(d)
        for d in [
            Path("data/filters"),
            Path("../data/filters"),
            Path("../../data/filters"),
            Path("../../../data/filters"),
        ]
        if d.exists()
    ),
    "data/filters",
)
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

# --- Setup ---
ssp_data = load_ssp_data(SSP_PATH)

# Three filter sets — chosen so all bands fit on the same UV-NIR axis.
filter_sets = {
    "SDSS": ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    "2MASS": ["2mass_j", "2mass_h", "2mass_ks"],
    "HST": ["hst_f435w", "hst_f606w", "hst_f814w"],
}

# Build common model (fixed parameters)
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
    sfh_tsnorm_width_gyr=Fixed(1.5),
    sfh_tsnorm_skew=Fixed(0.3),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.4),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.05),
)

# --- Plot three panels ---
fig, axes = plt.subplots(3, 1, figsize=(9, 8))

for (ax, (survey_name, bands)) in zip(axes, filter_sets.items()):
    obs = Observation(
        photometry=Photometry.from_names(bands, cache_dir=_FILTER_DIR),
    )
    model = SEDModel(spec, ssp_data, observation=obs)

    # Predict on this filter set
    pred = model.predict_rest_sed({})
    wave = np.asarray(pred.wavelength)
    sed = np.asarray(pred.sed)

    # Get filter information
    waves, trans, curves = load_filter_set(bands, cache_dir=_FILTER_DIR)

    # Plot SED
    ax.plot(wave, sed, "C0-", lw=2.0, label="SED (rest frame)")

    # Overlay filter throughputs scaled to SED
    for fc in curves:
        wave_f = np.array(fc.wave)
        trans_f = np.array(fc.trans)
        trans_f_scaled = trans_f * np.max(sed) * 0.5
        ax.fill_between(wave_f, 0, trans_f_scaled, alpha=0.2, color="C1")
        ax.plot(wave_f, trans_f_scaled, lw=1.0, color="C1", alpha=0.6)

    ax.set_ylabel(r"$L_\nu$ (arbitrary)")
    ax.set_xlim(3000, 25000)
    # Set y-limit to twice the median continuum so emission-line spikes don't
    # compress the rest of the SED. We don't care about the spike's peak here.
    sed_continuum = np.median(sed[(wave >= 3000) & (wave <= 25000)])
    ax.set_ylim(0, 4 * sed_continuum)
    ax.set_title(f"{survey_name}: {len(bands)} filters", fontsize=12)
    ax.legend(frameon=False, loc="upper right", fontsize=10)

axes[-1].set_xlabel(r"Wavelength [$\AA$]")
fig.tight_layout()
plt.savefig("plot_filter_set_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
