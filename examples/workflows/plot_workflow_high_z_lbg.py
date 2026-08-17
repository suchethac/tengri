"""
High-redshift Lyman-break galaxy: Lyman dropout signatures in JWST/HST
======================================================================

The Lyman-break signature (sharp UV dropout at observed ≈ 4 μm) constrains
age and metallicity even with just 4 bands.

References: Steidel+1996; Conroy+2013.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

# z=4 Lyman-break galaxy: very young, minimal dust
z_true = 4.0

# Observed filters: HST-F814W (rest-frame ~1300 A, near Lyman break at 912 A),
# then JWST NIR bands
bands = ["hst_f814w", "jwst_f150w", "jwst_f200w", "jwst_f277w"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_total_mass": 10.0,
        "peak_lbt_gyr": tengri.Uniform(0.1, 2.0),
        "width_gyr": tengri.Uniform(0.05, 1.0),
        "skew": tengri.Fixed(0.5),
        "trunc": tengri.Fixed(3.0),
        "logzsol": tengri.Uniform(-2.0, 0.0),
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": tengri.Uniform(0.0, 0.5),
        "tau_diff": 0.0,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(z_true),
)

# Generate mock data
key = jax.random.PRNGKey(42)
truth_params = {
    "sfh_tsnorm_log_total_mass": 1.2,
    "sfh_tsnorm_peak_lbt_gyr": 0.3,
    "sfh_tsnorm_width_gyr": 0.2,
    "sfh_tsnorm_skew": 0.5,
    "sfh_tsnorm_trunc": 3.0,
    "met_logzsol": -0.5,
    "dust_tau_bc": 0.1,
    "dust_tau_diff": 0.0,
    "dust_slope": -0.7,
    "redshift": z_true,
}
mock = model.mock(truth_params, snr=15.0, key=key)

# Fit with MAP
forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)

# Plot: SED with Lyman-break signature
fig, (ax_sed, ax_res) = plt.subplots(
    2,
    1,
    figsize=(7.5, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

wave_eff = np.array([8140, 15000, 20000, 27700])
band_labels = ["F814W", "F150W", "F200W", "F277W"]

ax_sed.errorbar(
    wave_eff,
    np.array(mock.flux_obs),
    yerr=np.array(mock.noise),
    fmt="o",
    color="k",
    ms=6,
    capsize=2,
    label="Observed",
)
ax_sed.plot(
    wave_eff,
    np.array(mock.flux_true),
    "s",
    color="0.2",
    ms=6,
    mfc="none",
    mew=1.2,
    label="Truth",
)
ax_sed.plot(
    wave_eff,
    np.array(model.predict_photometry(posterior.params)),
    "^",
    color="C3",
    ms=6,
    mfc="none",
    mew=1.2,
    label="MAP",
)

# Lyman-break location (rest-frame 912 Å → obs-frame 4350 Å)
ax_sed.axvline(912 * (1 + z_true), color="red", ls=":", lw=1, alpha=0.5)
ax_sed.text(912 * (1 + z_true), ax_sed.get_ylim()[1] * 0.9, "Lyman break", fontsize=9, color="red")

ax_sed.set_ylabel(r"$f_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=8)

residual = (
    np.array(mock.flux_obs) - np.array(model.predict_photometry(posterior.params))
) / np.array(mock.noise)
ax_res.axhline(0.0, color="0.5", lw=0.6)
ax_res.axhspan(-1.0, 1.0, color="0.85", alpha=0.6, lw=0)
ax_res.plot(wave_eff, residual, "o", color="C3", ms=5)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_ylabel(r"$(F_{\rm fit} - F_{\rm obs})/\sigma$")
ax_res.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_res.set_xticks(wave_eff)
ax_res.set_xticklabels(band_labels)

plt.savefig("plot_workflow_high_z_lbg.png", dpi=150, bbox_inches="tight")
