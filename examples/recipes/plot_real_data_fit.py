"""
From a CSV row to a MAP SED fit, end to end
============================================

The astronomer's-eye-view of the tengri ingest path. Starting from a
single CSV row of SDSS *ugriz* fluxes and per-band errors (the same
shape pandas would hand you from a survey catalog), we parse the
row, build the photometric ``Observation`` from the column names,
fit with MAP, and overlay the recovered SED on the observed bands
with normalized residuals.

Companion to ``plot_recipe_load_real_csv.py`` (bulk-fit three mock
galaxies); this script focuses on the *parsing* and *single-row*
workflow that catalog work begins with.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Inline CSV — same shape as a pandas row from a real catalog.
# Columns: name, redshift, then (band, band_err) pairs in erg s-1 cm-2 Hz-1.
CSV = (
    "name,redshift,sdss_u,sdss_u_err,sdss_g,sdss_g_err,"
    "sdss_r,sdss_r_err,sdss_i,sdss_i_err,sdss_z,sdss_z_err\n"
    "mock_galaxy,0.05,3.0e-27,5e-28,2.3e-26,1e-27,3.5e-26,1e-27,"
    "4.4e-26,1e-27,5.1e-26,2e-27\n"
)

lines = [ln for ln in CSV.strip().splitlines() if ln]
header = lines[0].split(",")
row = dict(zip(header, lines[1].split(",")))

bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
fluxes = jnp.array([float(row[b]) for b in bands])
errors = jnp.array([float(row[b + "_err"]) for b in bands])
z = float(row["redshift"])

obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))
model = tengri.SEDModel.build(
    tengri.load_ssp(),
    observation=obs,
    sfh={"type": "dpl", "*": tengri.FREE},
    redshift=tengri.Fixed(z),
)
forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(fluxes, errors, method="map", optimizer="adam", n_steps=300, verbose=False)

flux_fit = np.asarray(model.predict_photometry(posterior.params))
flux_obs = np.asarray(fluxes)
noise = np.asarray(errors)
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

fig, (ax_sed, ax_res) = plt.subplots(
    2,
    1,
    figsize=(7.0, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)
ax_sed.errorbar(
    wave_eff,
    flux_obs,
    yerr=noise,
    fmt="o",
    color="k",
    ms=5,
    capsize=2,
    label=f"CSV row ({row['name']}, z={z})",
)
ax_sed.plot(wave_eff, flux_fit, "^", color="C3", ms=7, mfc="none", mew=1.4, label="MAP")
ax_sed.set_yscale("log")
ax_sed.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=8, loc="lower right")

residual = (flux_fit - flux_obs) / noise
ax_res.axhline(0.0, color="0.5", lw=0.6)
ax_res.axhspan(-1.0, 1.0, color="0.85", alpha=0.6, lw=0)
ax_res.plot(wave_eff, residual, "o", color="C3", ms=5)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_ylabel(r"$(F_{\rm fit} - F_{\rm obs})/\sigma$")
ax_res.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")

plt.savefig("plot_real_data_fit.png", dpi=150, bbox_inches="tight")
