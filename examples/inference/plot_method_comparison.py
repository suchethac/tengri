"""
MAP vs geoVI Posterior Comparison
=================================

Compares point-estimate (MAP) and variational (vi/geoVI) inference
on mock 5-band photometry. Overlays posteriors as a corner plot.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_method_comparison_001.png
   :alt: plot_method_comparison
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt
import numpy as np

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

# --- SEDModel ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = SEDModel(spec, ssp, observation=obs)

# --- Mock photometry (star-forming galaxy) ---
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 2.0
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
true_params["sfh_tsnorm_skew"] = 0.3  # Positive skew = recent star formation
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit: MAP ---
forward = ForwardModel.build(sed=model, observation=obs)
fitter = Fitter(forward, mock.flux_obs, mock.noise, data_type="photometry")
result_map = fitter.run("map", n_steps=300, verbose=False)

# --- Fit: vi/geoVI (quick settings) ---
fitter.compile(verbose=False)
result_geovi = fitter.run(
    "vi",
    n_iterations=10,
    n_samples=4,
    n_posterior_samples=2000,
    verbose=False,
)

# --- Figure: corner comparison ---
fig = safe_corner(result_geovi, truths=true_params)
if fig is not None:
    # Mark MAP point
    map_vals = [float(result_map.params[p]) for p in spec.free_params]
    n = len(spec.free_params)
    # Reshape axes to a square grid; safe_corner creates n_params x n_params axes
    n_axes = int(np.sqrt(len(fig.axes)))
    axes = np.array(fig.axes).reshape(n_axes, n_axes) if n_axes > 0 else np.array(fig.axes)
    for i in range(n):
        if i < n_axes:
            axes[i, i].axvline(map_vals[i], color="C3", ls="--", lw=1.2, label="MAP")
            for j in range(i):
                if j < n_axes:
                    axes[i, j].axhline(map_vals[i], color="C3", ls="--", lw=0.8)
                    axes[i, j].axvline(map_vals[j], color="C3", ls="--", lw=0.8)
    if n_axes > 0:
        axes[0, 0].legend(fontsize=10)
    fig.suptitle("MAP (dashed red) vs geoVI posteriors", y=1.02)

plt.savefig(
    "plot_method_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

# --- SFH: truth vs MAP vs geoVI ---
sfh_true = model.predict_sfh(true_params)
sfh_map = model.predict_sfh(result_map.params)
sfh_geovi = model.predict_sfh(result_geovi.params)
t_gyr = np.array(sfh_true["t_gyr"])
mask = t_gyr < 5.0

fig_sfh, ax_sfh = plt.subplots(figsize=(6, 3.5))
ax_sfh.plot(t_gyr[mask], np.array(sfh_true["sfr_mean"])[mask], "k-", lw=2, label="Truth")
ax_sfh.plot(
    t_gyr[mask], np.array(sfh_map["sfr_mean"])[mask], "--", color="C3", lw=1.5, label="MAP"
)
ax_sfh.plot(
    t_gyr[mask], np.array(sfh_geovi["sfr_mean"])[mask], "--", color="C0", lw=1.5, label="geoVI"
)
ax_sfh.set_xlabel("Lookback time [Gyr]")
ax_sfh.set_ylabel("SFR [Msun/yr]")
ax_sfh.set_title("SFH recovery: MAP vs geoVI")
ax_sfh.legend(fontsize=10, frameon=False)
fig_sfh.tight_layout()
plt.show()
