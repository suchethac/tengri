"""
Load and Fit Real CSV Photometry
================================

How do I load photometric data from a CSV file and fit it? This recipe
demonstrates loading a table of measured fluxes and uncertainties,
building observations per galaxy, and running a MAP fit on each.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
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
    load_ssp_data,
)
from tengri.analysis.plotting import setup_style

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

# --- Generate mock CSV with 5 SDSS bands x 3 galaxies ---
# In practice, load your own CSV via np.genfromtxt() or pd.read_csv()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs_config = Observation(photometry=Photometry.from_names(bands))
spec_template = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(0.5),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_diff=Fixed(0.3),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model_template = SEDModel(spec_template, ssp, observation=obs_config)

# Mock 3 galaxies with different parameters
key = jax.random.PRNGKey(42)
galaxy_data = []
for gal_id in range(3):
    params = spec_template.sample(jax.random.fold_in(key, gal_id))
    mock = model_template.mock(params, snr=20.0, key=jax.random.fold_in(key, 100 + gal_id))
    galaxy_data.append(
        {
            "name": f"galaxy_{gal_id}",
            "redshift": 0.1,
            "flux": np.array(mock.flux_obs),
            "error": np.array(mock.noise),
        }
    )

# --- Fit each galaxy ---
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

for gal_idx, gal in enumerate(galaxy_data):
    # Build free model: redshift fixed to measured, others free
    spec = Parameters(
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(gal["redshift"]),
        mean_sfh_type="tsnorm",
    )
    model = SEDModel(spec, ssp, observation=obs_config)

    # MAP fit
    fitter = Fitter(model, data=gal["flux"], noise=gal["error"])
    posterior = fitter.run("map", optimizer="adam", n_steps=200, verbose=False)

    # Plot
    wave_eff = np.array([float(jnp.mean(w)) for w in obs_config.photometry.filter_waves])
    pred = np.array(model.predict_photometry(posterior.params))

    axes[gal_idx].errorbar(
        wave_eff,
        gal["flux"],
        yerr=gal["error"],
        fmt="o",
        color="k",
        ms=5,
        label="Data",
    )
    axes[gal_idx].plot(wave_eff, pred, "^", color="C3", ms=7, mfc="none", label="MAP fit", lw=2.0)
    axes[gal_idx].set_xlabel("Wavelength [A]")
    axes[gal_idx].set_ylabel("Flux [erg/s/cm²/Hz]")
    axes[gal_idx].set_title(gal["name"])
    axes[gal_idx].legend(frameon=False, fontsize=9)
    axes[gal_idx].set_yscale("log")

fig.tight_layout()
plt.savefig("plot_recipe_load_real_csv.png", dpi=150, bbox_inches="tight")
plt.show()
