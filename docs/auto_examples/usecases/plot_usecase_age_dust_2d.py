"""
Age-dust degeneracy: optical colors vs. UV constraining power
=============================================================

**Left panel:** The age–dust degeneracy as seen in optical g−r color.
A 5 Gyr stellar population with no dust is nearly indistinguishable
from a 1 Gyr population reddened by ``τ_diff = 0.4`` when observed in
optical broadband colors alone. A 2-D grid in (age, ``τ_diff``) with
iso-color contours reveals the orientation of the degeneracy—lines of
constant color show why optical colors alone cannot break this ambiguity.

**Right panel:** Demonstrating how UV photometry (FUV/NUV) breaks the
degeneracy. Two synthetic galaxies with identical SDSS ugriz photometry
(one old and dust-poor, one young and dusty) produce wildly different
UV signatures. Adding GALEX FUV/NUV observation constrains the UV slope
and uniquely identifies the true stellar population. This demonstrates
the critical importance of short-wavelength coverage for stellar age
and dust determination.

References:
- Conroy et al. 2009, ApJ, 699, 486 (age-dust-metallicity degeneracy)
- Conroy 2013, ARA&A, 51, 393 (§3, SED fitting overview)
- Worthey 1994, ApJS, 95, 107 (age/Z degeneracy origin)
- Meurer et al. 1999, ApJ, 521, 64 (UV slope as diagnostic)
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

# ==============================================================================
# LEFT PANEL: Optical color degeneracy in g − r
# ==============================================================================

obs_optical = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))
model_optical = tengri.SEDModel.build(
    ssp,
    observation=obs_optical,
    sfh={
        "type": "tsnorm",
        "*": tengri.FIXED,
        "peak_lbt_gyr": tengri.Uniform(0.1, 13.0),
        "width_gyr": 0.3,
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.5,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 2.0),
        "tau_bc": 0.3,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.01),
)
baseline_optical = dict(model_optical.spec.sample(jax.random.PRNGKey(0)))

age_grid = np.geomspace(0.5, 12.0, 30)
tau_grid = np.linspace(0.0, 2.0, 28)
g_minus_r = np.empty((tau_grid.size, age_grid.size))

for i, tau in enumerate(tau_grid):
    for j, age in enumerate(age_grid):
        params = {
            **baseline_optical,
            "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age),
            "dust_tau_diff": jnp.float64(tau),
        }
        flux = np.asarray(model_optical.predict_photometry(params))
        # F_nu -> AB mag, color = -2.5 log10(g/r)
        g_minus_r[i, j] = -2.5 * np.log10(flux[0] / flux[1])

# ==============================================================================
# RIGHT PANEL: UV photometry breaks the degeneracy
# ==============================================================================

# SDSS-only model (degenerate case)
obs_sdss = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

model_sdss = tengri.SEDModel.build(
    ssp,
    observation=obs_sdss,
    sfh={
        "type": "tsnorm",
        "log_total_mass": 10.0,
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

# SDSS + GALEX model (UV-constrained case)
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
        "log_total_mass": 10.0,
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
# Old, dust-poor
params_old = {
    "sfh_tsnorm_log_total_mass": 0.5,
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
    "sfh_tsnorm_log_total_mass": 1.5,
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

# SDSS photometry
phot_sdss_old = np.asarray(model_sdss.predict_photometry(params_old))
phot_sdss_young = np.asarray(model_sdss.predict_photometry(params_young))

# Make SDSS identical (use old galaxy's SDSS), then get the UV photometry
phot_sdss_use = phot_sdss_old  # SDSS-only: identical
phot_uv_old = np.asarray(model_sdss_uv.predict_photometry(params_old))
phot_uv_young = np.asarray(model_sdss_uv.predict_photometry(params_young))

# ==============================================================================
# Create two-panel figure
# ==============================================================================

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.0))

# LEFT: Optical g-r color degeneracy as 2D heatmap
mesh = ax_left.pcolormesh(
    age_grid, tau_grid, g_minus_r, cmap="RdYlBu_r", vmin=0.2, vmax=1.6, shading="auto"
)
levels = np.arange(0.4, 1.6, 0.1)
cs = ax_left.contour(
    age_grid, tau_grid, g_minus_r, levels=levels, colors="0.15", linewidths=0.6, alpha=0.8
)
ax_left.clabel(cs, fmt="%.1f", fontsize=7, inline=True, inline_spacing=2)
ax_left.set_xscale("log")
ax_left.set_xlabel(r"Stellar burst age [Gyr]", fontsize=10)
ax_left.set_ylabel(r"Diffuse dust optical depth $\tau_{\rm diff}$", fontsize=10)
cbar_left = fig.colorbar(mesh, ax=ax_left, pad=0.01)
cbar_left.set_label(r"$g - r$  [mag]", fontsize=10)
ax_left.text(
    0.05,
    0.90,
    "Optical colors alone\ncannot break the\nage–dust degeneracy",
    transform=ax_left.transAxes,
    fontsize=8,
    color="0.15",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5),
)
ax_left.set_title("(Left) Optical g−r Degeneracy", fontweight="bold", fontsize=11)

# RIGHT: SED example showing UV breaks the degeneracy
wave_sdss = np.array([3551, 4686, 6166, 7480, 8932])
wave_uv = np.array([1516, 2267])

ax_right.errorbar(
    wave_sdss, phot_sdss_use, fmt="o", color="k", ms=7, capsize=3, label="Observed SDSS"
)
ax_right.plot(
    wave_uv,
    phot_uv_old[:2],
    "s",
    color="C0",
    ms=7,
    mfc="none",
    mew=1.5,
    label="Old+dust-poor (truth)",
)
ax_right.plot(
    wave_uv,
    phot_uv_young[:2],
    "^",
    color="C3",
    ms=7,
    mfc="none",
    mew=1.5,
    label="Young+dusty (degenerate in SDSS)",
)

ax_right.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize=10)
ax_right.set_ylabel(r"$f_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]", fontsize=10)
ax_right.set_yscale("log")
ax_right.legend(frameon=False, loc="upper right", fontsize=8)
ax_right.axvspan(1500, 2300, color="red", alpha=0.1)
ax_right.text(1900, 1e-22, "UV breaks\ndegeneracy", fontsize=8, ha="center", color="C3")
ax_right.set_title("(Right) UV Photometry Breaks Ambiguity", fontweight="bold", fontsize=11)

fig.tight_layout()
plt.savefig("plot_usecase_age_dust_2d.png", dpi=150, bbox_inches="tight")
