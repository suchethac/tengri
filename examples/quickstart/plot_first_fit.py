"""
First Photometric Fit
=====================

Fit a parametric star-formation history and dust attenuation to mock SDSS
photometry using MAP optimization. Demonstrates the recommended workflow:
build an SED model, wrap it with ForwardModel, generate mock data, and
optimize with Fitter.run("map").

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_first_fit_001.png
   :alt: plot_first_fit
   :class: sphx-glr-single-img

"""

import jax
import jax.numpy as jnp
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
    load_ssp,
)
from tengri.plot import setup_style

setup_style()

# --- Build observation and SED model ---
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(bands))

sed = SEDModel.build(
    ssp_data=load_ssp(),
    observation=obs,
    sfh={"type": "tsnorm", "*": FREE},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,
        "tau_diff": Uniform(0.0, 1.5),
        "slope": -0.7,
    },
    redshift=Fixed(0.05),
)

# --- Wrap with ForwardModel for fitting ---
# --- Mock a star-forming galaxy at z=0.05 (SNR=20) ---
key = jax.random.PRNGKey(42)
truth = sed.spec.sample(key)
truth.update(
    sfh_tsnorm_peak_lbt_gyr=3.0,
    sfh_tsnorm_width_gyr=2.0,
    sfh_tsnorm_log_peak_sfr=1.0,
    sfh_tsnorm_skew=0.3,
)
mock = sed.mock(truth, snr=20.0, key=key)

# --- Fit with MAP (Adam) ---
posterior = Fitter(sed, data=mock.flux_obs, noise=mock.noise).run(
    "map", optimizer="adam", n_steps=300, verbose=False
)

# --- Plot: data, truth, MAP fit ---
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    wave_eff, mock.flux_obs, yerr=mock.noise, fmt="o", color="k", ms=5, label="Observed (SNR=20)"
)
ax.plot(wave_eff, mock.flux_true, "s", color="C0", ms=7, mfc="none", label="Truth")
ax.plot(
    wave_eff,
    sed.predict_photometry(posterior.params),
    "^",
    color="C3",
    ms=7,
    mfc="none",
    label="MAP fit",
)

ax.set(
    xlabel="Wavelength [Å]",
    ylabel=r"Flux density [erg/s/cm$^2$/Hz]",
    title="First Photometric Fit with tengri",
)
ax.legend(fontsize=10, frameon=False)
fig.tight_layout()
plt.savefig("plot_first_fit.png", dpi=150, bbox_inches="tight")
plt.show()
