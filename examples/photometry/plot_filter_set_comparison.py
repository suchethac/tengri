"""
Filter placement decides which spectral features a survey can see
==================================================================

The same star-forming galaxy SED is intercepted by three different filter
sets — SDSS *ugriz* (optical), 2MASS *JHKs* (near-infrared), and HST ACS
*F435W/F606W/F814W* (UV-optical). Each panel overlays the survey's
throughputs on the shared SED so the reader sees, at a glance, which
spectral features (the 4000-Å break, Hα + [N II], the 1.6-μm stellar
bump) fall inside each band.

Reference: Conroy 2013, ARA&A, 51, 393.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

import tengri
from tengri import data_path
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

FILTER_SETS = {
    "SDSS *ugriz*": ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    "2MASS *JHKs*": ["2mass_j", "2mass_h", "2mass_ks"],
    "HST ACS": ["hst_f435w", "hst_f606w", "hst_f814w"],
}
PANEL_XLIM = {
    "SDSS *ugriz*": (3000, 11000),
    "2MASS *JHKs*": (8000, 25000),
    "HST ACS": (3000, 11000),
}

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "log_total_mass": 10.0,
        "peak_lbt_gyr": 2.0,
        "width_gyr": 1.5,
        "skew": 0.3,
        "trunc": 3.0,
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.4,
        "tau_diff": 0.2,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

pred = model.predict(baseline)
wave = np.asarray(model.wavelengths)
sed = np.asarray(pred.rest_sed())
# Smooth the emission-line spikes for display only — the photometry path
# still integrates the spiky SED through the filters.
sed_smooth = median_filter(sed, size=51)

fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5))
for ax, (survey, bands) in zip(axes, FILTER_SETS.items()):
    xlo, xhi = PANEL_XLIM[survey]
    panel_mask = (wave >= xlo) & (wave <= xhi)
    continuum = float(np.median(sed_smooth[panel_mask]))
    y_floor = continuum * 0.05
    y_ceil = continuum * 0.8

    ax.semilogy(wave, sed_smooth, color="C0", lw=2.0, label="rest-frame SED")

    _, _, curves = tengri.load_filter_set(bands, cache_dir=str(data_path("filters")))
    for fc in curves:
        wave_f = np.asarray(fc.wave)
        trans_f = np.asarray(fc.trans) / float(np.max(fc.trans))
        scaled = y_floor + trans_f * (y_ceil - y_floor)
        ax.fill_between(wave_f, y_floor, scaled, alpha=0.25, color="C1")
        ax.plot(wave_f, scaled, lw=1.2, color="C1", alpha=0.8)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(continuum * 0.02, continuum * 4)
    ax.set_ylabel(r"$L_\nu$  [arbitrary]")
    ax.text(0.02, 0.95, f"{survey} ({len(bands)} bands)", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

axes[-1].set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
fig.tight_layout()
plt.savefig("plot_filter_set_comparison.png", dpi=150, bbox_inches="tight")
