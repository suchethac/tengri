"""
Photometric SED Fit
====================

Generate a mock galaxy with SDSS ugriz photometry and fit it using
tengri's variational inference. Shows observed vs model photometry
with error bars and residuals.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_photometric_fit_001.png
   :alt: plot_photometric_fit
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    data_path,
    load_ssp,
)
from tengri.plot import setup_style

setup_style()


ssp = load_ssp()

bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(
    photometry=Photometry.from_names(bands, cache_dir=str(data_path("filters"))),
)

sed = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": FREE, "logzsol": Uniform(-2.0, 0.2)},
    dust={"type": "two_component", "*": FIXED, "tau_bc": 0.3, "tau_diff": 0.2, "slope": -0.7},
    redshift=Fixed(0.05),
)

# --- Generate mock data (star-forming galaxy) ---
true_params = sed.spec.sample(jax.random.PRNGKey(42))
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 2.0
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
true_params["sfh_tsnorm_skew"] = 0.3
mock = sed.mock(true_params, snr=20.0, key=jax.random.PRNGKey(43))

# --- Fit with MAP ---
posterior = Fitter(sed, data=mock.flux_obs, noise=mock.noise).run(
    "map", optimizer="adam", n_steps=300, verbose=False
)
best_fit = sed.predict_photometry(posterior.params)

# --- Plot ---
wave_eff = np.array([3551, 4686, 6166, 7480, 8932])  # SDSS effective wavelengths
band_names = ["u", "g", "r", "i", "z"]

fig, (ax, ax_res) = plt.subplots(
    2, 1, figsize=(7, 5), height_ratios=[3, 1], sharex=True, gridspec_kw={"hspace": 0.05}
)

ax.errorbar(
    wave_eff,
    np.array(mock.flux_obs),
    yerr=np.array(mock.noise),
    fmt="o",
    color="0.3",
    ms=6,
    capsize=3,
    label="Observed",
    zorder=5,
)
ax.scatter(
    wave_eff,
    np.array(mock.flux_true),
    marker="s",
    s=50,
    facecolors="none",
    edgecolors="C0",
    lw=1.5,
    label="Truth",
    zorder=4,
)
ax.scatter(wave_eff, np.array(best_fit), marker="D", s=40, color="C3", label="MAP fit", zorder=6)
ax.set_ylabel(r"$f_\nu$ [arbitrary]")
ax.legend(frameon=False)
ax.set_title("SDSS Photometric Fit (MAP)")

residuals = (np.array(mock.flux_obs) - np.array(best_fit)) / np.array(mock.noise)
ax_res.axhline(0, color="0.5", ls="--", lw=0.8)
ax_res.scatter(wave_eff, residuals, c="C3", s=30, zorder=5)
ax_res.set_xlabel(r"Wavelength [$\AA$]")
ax_res.set_ylabel(r"$(f_\mathrm{obs} - f_\mathrm{mod}) / \sigma$")
ax_res.set_ylim(-4, 4)
ax_res.set_xticks(wave_eff)
ax_res.set_xticklabels(band_names)

fig.tight_layout()
plt.savefig("plot_photometric_fit.png", dpi=150, bbox_inches="tight")
plt.show()
