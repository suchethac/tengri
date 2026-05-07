"""
Save and Load a Posterior
==========================

How do I save a posterior to disk and load it later? This recipe demonstrates
running a NUTS fit, saving the Posterior to an HDF5 file, reloading it,
and analyzing the saved results.
"""

import tempfile
from pathlib import Path

import jax
import matplotlib.pyplot as plt

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)
from tengri.analysis.plotting import setup_style
from tengri.inference.posterior import Posterior

setup_style()


def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

# --- Setup and fit ---
bands = ["sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(bands))

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.08),
    mean_sfh_type="tsnorm",
)
model = SEDModel(spec, ssp, observation=obs)

# Generate mock data
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
true_params["sfh_tsnorm_peak_lbt_gyr"] = 2.5
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
mock = model.mock(true_params, snr=25.0, key=key)

# Fit with NUTS (small sample for speed)
fitter = Fitter(model, data=mock.flux_obs, noise=mock.noise)
fitter.run("map", optimizer="adam", n_steps=150, verbose=False)
posterior = fitter.run(
    "mcmc_nuts",
    n_warmup=50,
    n_samples=100,
    verbose=False,
)

# --- Save to temporary file ---
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = str(Path(tmpdir) / "posterior.h5")
    print(f"Saving posterior to: {save_path}")
    posterior.save(save_path)

    # --- Load from file ---
    print(f"Loading posterior from: {save_path}")
    posterior_loaded = Posterior.load(save_path, model=model)

    # --- Verify loaded posterior ---
    print(f"\nLoaded posterior method: {posterior_loaded.method}")
    print(f"Number of samples: {len(posterior_loaded.samples['met_logzsol'])}")
    print(f"Diagnostics: {posterior_loaded.diagnostics}")

    # --- Plot both originals and loaded ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Original posterior scatter
    if posterior.samples:
        axes[0].scatter(
            posterior.samples["sfh_tsnorm_log_peak_sfr"],
            posterior.samples["met_logzsol"],
            alpha=0.5,
            s=30,
            color="C0",
        )
        axes[0].set_xlabel("log peak SFR [Msun/yr]")
        axes[0].set_ylabel("log Z/Zsun")
        axes[0].set_title("Original Posterior (in memory)")

    # Loaded posterior scatter
    if posterior_loaded.samples:
        axes[1].scatter(
            posterior_loaded.samples["sfh_tsnorm_log_peak_sfr"],
            posterior_loaded.samples["met_logzsol"],
            alpha=0.5,
            s=30,
            color="C3",
        )
        axes[1].set_xlabel("log peak SFR [Msun/yr]")
        axes[1].set_ylabel("log Z/Zsun")
        axes[1].set_title("Loaded Posterior (from HDF5)")

    fig.tight_layout()
    plt.savefig("plot_recipe_save_load_posterior.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nSave/load test complete!")
