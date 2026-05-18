"""
Convergence Diagnostics
=======================

Runs a quick fit and displays convergence diagnostics: ESS per parameter,
summary table, and trace plots for a subset of parameters.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_convergence_001.png
   :alt: plot_convergence
   :class: sphx-glr-single-img

"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp,
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
key = jax.random.PRNGKey(7)
true_params = spec.sample(key)
# Override to ensure star-forming galaxy
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 2.0
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
true_params["sfh_tsnorm_skew"] = 0.3  # Positive skew = recent star formation
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit with vi (geoVI) ---
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")
fitter.run("map", n_steps=300, verbose=False)
fitter.compile(verbose=False)
posterior = fitter.run(
    "vi",
    n_iterations=10,
    n_samples=4,
    n_posterior_samples=3000,
    verbose=False,
)

# --- Diagnostics ---
print(posterior.summary_table())
ess = posterior.effective_sample_size()

# --- Figure: ESS bar chart + trace plots + SFH inset ---
names = spec.free_params
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Left: ESS per parameter
ess_vals = [ess.get(n, 0.0) for n in names]
colors = ["#2ecc71" if e > 100 else "#e74c3c" for e in ess_vals]
short_names = [n.replace("sfh_tsnorm_", "").replace("_", " ") for n in names]
axes[0].barh(short_names, ess_vals, color=colors)
axes[0].axvline(100, color="k", ls="--", lw=0.8, label="ESS = 100 threshold")
axes[0].set_xlabel("Effective sample size")
axes[0].set_title("ESS per parameter")
axes[0].legend(fontsize=10)

# Right: trace plot for first 3 params
for i, name in enumerate(names[:3]):
    samples = np.array(posterior.samples[name])
    axes[1].plot(samples, alpha=0.6, lw=0.4, label=short_names[i])
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Parameter value")
axes[1].set_title("Trace plots (first 3 parameters)")
axes[1].legend(fontsize=10)

# Right: SFH truth vs inferred
sfh_true = model.predict_sfh(true_params)
sfh_fit = model.predict_sfh(posterior.params)
t_gyr = np.array(sfh_true["t_gyr"])
mask = t_gyr < 5.0
axes[2].plot(t_gyr[mask], np.array(sfh_true["sfr_mean"])[mask], "k-", lw=1.5, label="Truth")
axes[2].plot(t_gyr[mask], np.array(sfh_fit["sfr_mean"])[mask], "r--", lw=1.2, label="geoVI")
axes[2].set_xlabel("Lookback [Gyr]")
axes[2].set_ylabel("SFR [Msun/yr]")
axes[2].set_title("SFH recovery")
axes[2].legend(fontsize=10)

fig.tight_layout()
outdir = Path(__file__).resolve().parent.parent / "figures" if "__file__" in dir() else Path(".")
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig("plot_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
