"""
Save and load a posterior to disk
==================================

How do I persist a posterior between sessions? This recipe runs a MAP fit,
saves the result to HDF5, reloads it, and demonstrates basic analysis.
Posterior objects can be checkpointed for long-running fits or multi-stage
analysis pipelines.
"""

import tempfile
import warnings
from pathlib import Path

import jax
import matplotlib.pyplot as plt

import tengri
from tengri.analysis.plotting import setup_style
from tengri.inference.posterior import Posterior

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
bands = ["sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# Build model and generate mock data
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": tengri.FREE},
    dust={"type": "two_component", "*": tengri.FREE},
    redshift=tengri.Fixed(0.08),
)

key = jax.random.PRNGKey(42)
true_params = dict(model.spec.sample(key))
true_params.update(
    {
        "sfh_tsnorm_peak_lbt_gyr": 2.5,
        "sfh_tsnorm_log_peak_sfr": 1.0,
    }
)
mock = model.mock(true_params, snr=25.0, key=key)

# Fit with MAP
forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs, mock.noise, method="map", optimizer="adam", n_steps=150, verbose=False
)

# Save to temporary file and reload
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = str(Path(tmpdir) / "posterior.h5")
    print(f"Saving posterior to {Path(save_path).name}")
    posterior.save(save_path)

    print(f"Loading posterior from {Path(save_path).name}")
    posterior_loaded = Posterior.load(save_path, model=model)
    print(f"Method: {posterior_loaded.method}")
    print(f"Parameters: {len(posterior_loaded.params)}")

    # Plot scatter of parameters from loaded posterior
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    sfr = posterior_loaded.params["sfh_tsnorm_log_peak_sfr"]
    met = posterior_loaded.params["met_logzsol"]

    ax.scatter(sfr, met, alpha=0.6, s=50, color="C0", edgecolors="k", linewidth=0.5)
    ax.set_xlabel(r"$\log_{10}(\mathrm{SFR}_{\rm peak})$ [M$_\odot$/yr]")
    ax.set_ylabel(r"$\log_{10}(Z/Z_\odot)$")

    fig.tight_layout()
    fig.savefig("plot_recipe_save_load_posterior.png", dpi=150, bbox_inches="tight")
