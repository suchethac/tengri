"""
Corner Plot with Truth Overlay
==============================

Fits mock photometry and displays a corner plot with injected truth
values marked. Uses tengri's safe_corner utility.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_corner_001.png
   :alt: plot_corner
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt

from tengri import (
    Fitter,
    Fixed,
    ForwardModel,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp,
    safe_corner,
    setup_style,
)

setup_style()


# --- Data ---

ssp = load_ssp()
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

# --- SEDModel + mock ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = SEDModel(spec, ssp, observation=obs)
key = jax.random.PRNGKey(99)
true_params = spec.sample(key)
# Override to ensure star-forming galaxy
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 2.0
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
mock = model.mock(true_params, snr=25.0, key=key)

# --- Fit ---
forward = ForwardModel.build(sed=model, observation=obs)
fitter = Fitter(forward, mock.flux_obs, mock.noise, data_type="photometry")
fitter.run("map", n_steps=300, verbose=False)
fitter.compile(verbose=False)
posterior = fitter.run(
    "vi",
    n_iterations=10,
    n_samples=4,
    n_posterior_samples=3000,
    verbose=False,
)

# --- Corner plot ---
fig = safe_corner(posterior, truths=true_params)
if fig is not None:
    fig.suptitle("Posterior corner plot (truth = blue lines)", y=1.02)

plt.savefig("plot_corner.png", dpi=150, bbox_inches="tight")
plt.show()
