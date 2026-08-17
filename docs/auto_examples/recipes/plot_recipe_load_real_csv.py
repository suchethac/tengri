"""
Load and fit photometry from CSV
================================

Mock 3 galaxies, fit each independently with MAP. The workflow is: sample
true parameters → generate mock fluxes + noise → fit with free SFH/dust and
fixed redshift. Demonstrates vectorizing catalog-scale fits when redshift is
already known (e.g., spectroscopy).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# Generate mock photometry for 3 galaxies
key = jax.random.PRNGKey(42)
galaxy_data = []
for gal_id in range(3):
    model_template = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "tsnorm",
            "all_params": tengri.FIXED,
            "log_total_mass": 10.0,
            "peak_lbt_gyr": 3.0,
            "width_gyr": 2.0,
            "skew": 0.0,
            "trunc": 5.0,
        },
        dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.3, "slope": -0.7},
        redshift=tengri.Fixed(0.1),
    )
    params = dict(model_template.spec.sample(jax.random.fold_in(key, gal_id)))
    mock = model_template.mock(params, snr=20.0, key=jax.random.fold_in(key, 100 + gal_id))
    galaxy_data.append(
        {
            "name": f"galaxy_{gal_id}",
            "redshift": 0.1,
            "flux": np.array(mock.flux_obs),
            "error": np.array(mock.noise),
        }
    )

# Fit each galaxy with MAP
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), sharey=True)

for gal_idx, gal in enumerate(galaxy_data):
    model = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "tsnorm", "all_params": tengri.FREE},
        dust={"type": "two_component", "all_params": tengri.FREE},
        redshift=tengri.Fixed(gal["redshift"]),
    )

    forward = tengri.ForwardModel.build(sed=model, observation=obs)
    posterior = forward.fit(
        gal["flux"], gal["error"], method="map", optimizer="adam", n_steps=200, verbose=False
    )

    # Plot data vs model
    wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])
    pred = np.array(model.predict_photometry(posterior.params))

    axes[gal_idx].errorbar(
        wave_eff, gal["flux"], yerr=gal["error"], fmt="o", color="k", ms=5, label="Data", zorder=10
    )
    axes[gal_idx].plot(wave_eff, pred, "^", color="C3", ms=6, mfc="none", label="MAP", zorder=5)
    axes[gal_idx].set_xlabel("Wavelength [Å]")
    axes[gal_idx].set_yscale("log")
    ax_text = axes[gal_idx].text(
        0.05,
        0.95,
        gal["name"],
        transform=axes[gal_idx].transAxes,
        fontsize=9,
        verticalalignment="top",
    )

axes[0].set_ylabel(r"$F_\nu$ [erg/s/cm$^2$/Hz]")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")

fig.tight_layout()
plt.savefig("plot_recipe_load_real_csv.png", dpi=150, bbox_inches="tight")
