"""
First Photometric Fit
=====================

The simplest possible tengri workflow: define a parametric SFH model,
generate mock SDSS photometry, fit it with MAP optimization, and plot
the resulting SED fit.
"""

import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fitter,
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

# --- Load SSP data ---
SSP_PATH = "data/ssp_fsps_v3.2.h5"
try:
    ssp = load_ssp_data(SSP_PATH)
except FileNotFoundError:
    print(f"SSP data not found at {SSP_PATH}. Skipping.")
    sys.exit(0)

# --- Define model ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.05),
    mean_sfh_type="tsnorm",
)

obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))
model = Model(spec, ssp, observation=obs)

# --- Generate mock photometry ---
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit with MAP ---
fitter = Fitter(model, data=mock.flux_obs, noise=mock.noise)
posterior = fitter.run("map", optimizer="adam", n_steps=300, verbose=False)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 4))
filter_names = obs.photometry.names
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

ax.errorbar(wave_eff, np.array(mock.flux_obs), yerr=np.array(mock.noise),
            fmt="o", color="k", ms=5, label="Observed (SNR=20)")
ax.plot(wave_eff, np.array(mock.flux_true), "s", color="C0",
        ms=7, mfc="none", label="Truth")
ax.plot(wave_eff, np.array(model.predict_photometry(posterior.params)),
        "^", color="C3", ms=7, mfc="none", label="MAP fit")

ax.set_xlabel("Wavelength [A]")
ax.set_ylabel("Flux density [erg/s/cm$^2$/Hz]")
ax.set_title("First Photometric Fit with tengri")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
plt.savefig("plot_first_fit.png", dpi=150, bbox_inches="tight")
plt.show()
