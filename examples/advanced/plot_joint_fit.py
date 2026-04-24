"""
Joint Photometry + Spectroscopy Fit
====================================

Demonstrates tengri's Observation API for joint fitting. Creates a mock galaxy
with both SDSS photometry and a low-resolution spectrum, fits with MAP, and
shows the combined constraint power. Requires SSP data.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
    generate_mock,
    load_ssp_data,
    setup_style,
)

setup_style()


# %%
# Locate SSP data
# ----------------


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

# %%
# Build Observation with joint photometry + spectroscopy
# -------------------------------------------------------

_FILTER_DIR = next(
    (
        str(d)
        for d in [
            Path("data/filters"),
            Path("../data/filters"),
            Path("../../data/filters"),
            Path("../../../data/filters"),
        ]
        if d.exists()
    ),
    "data/filters",
)

phot = Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    cache_dir=_FILTER_DIR,
)
wave_obs = jnp.linspace(3800.0, 9200.0, 200) * 1.1  # observed frame at z=0.1
spec_config = Spectroscopy(wave_obs=wave_obs, resolution=100.0)
obs = Observation(photometry=phot, spectroscopy=spec_config)

# %%
# Create model and generate mock data
# -------------------------------------

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Fixed(0.3),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = SEDModel(spec, ssp, observation=obs)

true_params = {
    "sfh_tsnorm_log_peak_sfr": 1.2,
    "sfh_tsnorm_peak_lbt_gyr": 1.5,  # Recent peak = star-forming
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.3,  # Positive skew = recent star formation
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.4,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

key = jax.random.PRNGKey(42)
mock = generate_mock(model, true_params, key=key, snr=20.0)

# %%
# Fit with MAP
# -------------

fitter = Fitter(model, mock["flux_obs"], mock["noise"])
posterior = fitter.run("map", optimizer="adam")

# %%
# Plot: 2-panel figure (photometry + spectrum)
# ----------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: photometry
ax = axes[0]
band_names = ["u", "g", "r", "i", "z"]
band_wave_um = np.array([3551, 4686, 6166, 7480, 8932]) / 1e4
phot_true = np.array(mock["flux_true"][:5])
phot_obs = np.array(mock["flux_obs"][:5])
phot_noise = np.array(mock["noise"][:5])

ax.errorbar(band_wave_um, phot_obs, yerr=phot_noise, fmt="o", color="k", ms=6, label="Observed")
ax.plot(band_wave_um, phot_true, "s", color="C0", ms=8, mfc="none", mew=1.5, label="True")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"Flux density [$\mu$Jy]")
ax.set_title("SDSS Photometry")
ax.legend(frameon=False)

# Right panel: spectrum
ax = axes[1]
wave_plot = np.array(wave_obs) / 1e4
n_phot = 5
if len(mock["flux_obs"]) > n_phot:
    spec_obs = np.array(mock["flux_obs"][n_phot:])
    spec_true = np.array(mock["flux_true"][n_phot:])
    ax.plot(wave_plot, spec_obs, color="grey", lw=0.5, alpha=0.7, label="Observed")
    ax.plot(wave_plot, spec_true, color="C0", lw=1.5, label="True")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"Flux density [$\mu$Jy]")
ax.set_title("Low-resolution Spectrum (R=100)")
ax.legend(frameon=False)

# --- SFH inset on photometry panel ---
sfh_true = model.predict_sfh(true_params)
sfh_fit = model.predict_sfh(posterior.params)
t_gyr = np.array(sfh_true["t_gyr"])
inset = axes[0].inset_axes([0.55, 0.55, 0.40, 0.40])
mask = t_gyr < 5.0
inset.plot(t_gyr[mask], np.array(sfh_true["sfr_mean"])[mask], "k-", lw=1.5, label="Truth")
inset.plot(t_gyr[mask], np.array(sfh_fit["sfr_mean"])[mask], "r--", lw=1.2, label="MAP")
inset.set_xlabel("Lookback [Gyr]", fontsize=10)
inset.set_ylabel("SFR", fontsize=10)
inset.tick_params(labelsize=5)
inset.legend(fontsize=10)

fig.suptitle("Joint Photometry + Spectroscopy Mock", fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig("joint_fit.png", dpi=150, bbox_inches="tight")
plt.show()
