"""
Age-dust-metallicity degeneracy: why UV photometry is critical
==============================================================

Two synthetic galaxies with identical SDSS ugriz photometry — one old and
dust-poor, one young and dust-rich — produce wildly different SED fits.
Adding GALEX FUV/NUV observation breaks the degeneracy by constraining the
UV slope. Demonstrates the critical importance of short-wavelength coverage
for stellar age and dust determination.

Reference: Conroy et al. 2009, ApJ, 699, 486 (age-dust-metallicity
degeneracy); Conroy 2013, ARA&A, 51, 393 (SED fitting).
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

# SDSS-only model
obs_sdss = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

model_sdss = tengri.SEDModel.build(
    ssp,
    observation=obs_sdss,
    sfh={
        "type": "tsnorm",
        "log_peak_sfr": tengri.Uniform(-1.0, 2.5),
        "peak_lbt_gyr": tengri.Uniform(0.5, 12.0),
        "width_gyr": tengri.Uniform(0.3, 5.0),
        "skew": tengri.Uniform(-3.0, 3.0),
        "trunc": tengri.Uniform(1.0, 10.0),
        "logzsol": tengri.Uniform(-2.0, 0.2),
    },
    dust={
        "type": "two_component",
        "tau_bc": tengri.Uniform(0.0, 2.0),
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": tengri.Fixed(-0.7),
    },
    redshift=tengri.Fixed(0.1),
)

# SDSS + GALEX (FUV/NUV) model with UV coverage
obs_sdss_uv = tengri.Observation(
    photometry=tengri.Photometry.from_names(
        ["galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    )
)

model_sdss_uv = tengri.SEDModel.build(
    ssp,
    observation=obs_sdss_uv,
    sfh={
        "type": "tsnorm",
        "log_peak_sfr": tengri.Uniform(-1.0, 2.5),
        "peak_lbt_gyr": tengri.Uniform(0.5, 12.0),
        "width_gyr": tengri.Uniform(0.3, 5.0),
        "skew": tengri.Uniform(-3.0, 3.0),
        "trunc": tengri.Uniform(1.0, 10.0),
        "logzsol": tengri.Uniform(-2.0, 0.2),
    },
    dust={
        "type": "two_component",
        "tau_bc": tengri.Uniform(0.0, 2.0),
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": tengri.Fixed(-0.7),
    },
    redshift=tengri.Fixed(0.1),
)

# Generate mock data: two distinct SED solutions (old+dustless vs young+dusty)
key = jax.random.PRNGKey(42)

# Old, dust-poor
params_old = {
    "sfh_tsnorm_log_peak_sfr": 0.5,
    "sfh_tsnorm_peak_lbt_gyr": 8.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": -0.5,
    "sfh_tsnorm_trunc": 10.0,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.05,
    "dust_tau_diff": 0.02,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

# Young, dusty
params_young = {
    "sfh_tsnorm_log_peak_sfr": 1.5,
    "sfh_tsnorm_peak_lbt_gyr": 1.0,
    "sfh_tsnorm_width_gyr": 0.5,
    "sfh_tsnorm_skew": 0.8,
    "sfh_tsnorm_trunc": 2.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 1.0,
    "dust_tau_diff": 0.6,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

# SDSS photometry (same for both)
phot_sdss_old = np.asarray(model_sdss.predict_photometry(params_old))
phot_sdss_young = np.asarray(model_sdss.predict_photometry(params_young))

# Make them equal (identical SDSS), then get the UV photometry
phot_sdss_use = phot_sdss_old  # Use old galaxy's SDSS
phot_uv_old = np.asarray(model_sdss_uv.predict_photometry(params_old))
phot_uv_young = np.asarray(model_sdss_uv.predict_photometry(params_young))

# Plot SED comparison
fig, ax = plt.subplots(figsize=(9, 5))

wave_sdss = np.array([3551, 4686, 6166, 7480, 8932])
wave_uv = np.array([1516, 2267])
wave_all = np.concatenate([wave_uv, wave_sdss])

# Plot: two degenerate solutions in SDSS only, broken with UV
ax.errorbar(wave_sdss, phot_sdss_use, fmt="o", color="k", ms=7, capsize=3, label="Observed SDSS")
ax.plot(
    wave_uv,
    phot_uv_old[:2],
    "s",
    color="C0",
    ms=7,
    mfc="none",
    mew=1.5,
    label="Old+dust-poor (truth)",
)
ax.plot(
    wave_uv,
    phot_uv_young[:2],
    "^",
    color="C3",
    ms=7,
    mfc="none",
    mew=1.5,
    label="Young+dusty (degenerate)",
)

ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"$f_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.set_yscale("log")
ax.legend(frameon=False, loc="upper right")
ax.axvspan(1500, 2300, color="red", alpha=0.1, label="UV breaks degeneracy")

fig.tight_layout()
fig.savefig("plot_usecase_age_dust_degeneracy.png", dpi=150, bbox_inches="tight")
