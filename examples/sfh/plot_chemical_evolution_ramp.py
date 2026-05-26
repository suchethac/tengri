"""
Metallicity evolution: three scenarios for Z(t) and resulting SED
===================================================================

The chemical composition of stars encodes the assembly history of galaxies.
This figure demonstrates three metallicity evolution pathways available in
tengri: (1) constant solar Z, (2) linear ramp from Z = 0.1 Zsun to Zsun
over 13 Gyr of cosmic time, and (3) two-step enrichment (low-metallicity
plateau at early times, then a sharp jump at lookback time 8 Gyr ago).

Physical interpretation: early galaxies assembled from primordial gas with low
metallicity, then either gradually enriched (via in-situ nucleosynthesis) or
experienced rapid metal injection from a merger or starburst event. The
resulting stellar populations carry different absorption features and colors
due to metallicity-dependent stellar opacity and line strengths.

The top panel shows Z(t) for each scenario over cosmic time. The bottom panel
displays the rest-frame integrated SED; metallicity variations produce
stronger UV absorption and altered optical/near-IR colors.

Reference: Searle, L. 1971, ApJ, 168, 327 (galactic chemical evolution foundations).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import os
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Physical constants
AGE_UNIVERSE_GYR = 13.8  # Age of universe in Gyr

# Load SSP data (required for all models)
ssp = tengri.load_ssp()

# ─────────────────────────────────────────────────────────────────────────
# NOTE: met_mode is currently only exposed via the flat-kwarg Parameters()
# constructor, not via the nested-dict SEDModel.build() API. This example
# uses Parameters() directly to construct each scenario's specification.
# See: https://github.com/suchethacooray/tengri/issues/XXX (future public-API
# exposure).
# ─────────────────────────────────────────────────────────────────────────

# Scenario 1: Constant solar metallicity (Z = 1 Zsun)
spec_delta = tengri.Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_tau_gyr=1.0,
    sfh_dpl_log_total_mass=10.0,
    dust_tau_diff=0.1,
    met_mode="delta",
    redshift=0.0,
)
model_delta = tengri.SEDModel(spec_delta, ssp)

# Scenario 2: Linear ramp from low to solar metallicity
spec_ramp = tengri.Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_tau_gyr=1.0,
    sfh_dpl_log_total_mass=10.0,
    dust_tau_diff=0.1,
    met_mode="ramp",
    redshift=0.0,
)
model_ramp = tengri.SEDModel(spec_ramp, ssp)

# Scenario 3: Two-step metallicity (low then high)
spec_twostep = tengri.Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_tau_gyr=1.0,
    sfh_dpl_log_total_mass=10.0,
    dust_tau_diff=0.1,
    met_mode="two_step",
    redshift=0.0,
)
model_twostep = tengri.SEDModel(spec_twostep, ssp)

# ─────────────────────────────────────────────────────────────────────────
# Set up parameters for each scenario
# ─────────────────────────────────────────────────────────────────────────

# Scenario 1: constant solar
p_delta = dict(model_delta.spec.sample(jax.random.PRNGKey(42)))
p_delta["met_logzsol"] = 0.0  # Solar metallicity (Z = 1 Zsun)

# Scenario 2: ramp from -1.0 to 0.0 dex (0.1 Zsun to solar)
p_ramp = dict(model_ramp.spec.sample(jax.random.PRNGKey(43)))
p_ramp["met_logzsol_0"] = -1.0  # Early-time Z = 0.1 Zsun
p_ramp["met_logzsol_final"] = 0.0  # Present-day Z = 1 Zsun

# Scenario 3: two-step (low then jump)
p_twostep = dict(model_twostep.spec.sample(jax.random.PRNGKey(44)))
p_twostep["met_logzsol_old"] = -0.5  # Old stars: Z = 0.316 Zsun
p_twostep["met_logzsol_young"] = 0.0  # Young stars: Z = 1.0 Zsun
p_twostep["met_step_age_gyr"] = 8.0  # Step occurs at 8 Gyr lookback time

# ─────────────────────────────────────────────────────────────────────────
# Generate SED predictions
# ─────────────────────────────────────────────────────────────────────────

pred_delta = model_delta.predict_rest_sed(p_delta)
pred_ramp = model_ramp.predict_rest_sed(p_ramp)
pred_twostep = model_twostep.predict_rest_sed(p_twostep)

# Extract wavelength and flux arrays
wave_delta = np.asarray(pred_delta.wavelength)
sed_delta = np.asarray(pred_delta.sed)

wave_ramp = np.asarray(pred_ramp.wavelength)
sed_ramp = np.asarray(pred_ramp.sed)

wave_twostep = np.asarray(pred_twostep.wavelength)
sed_twostep = np.asarray(pred_twostep.sed)

# ─────────────────────────────────────────────────────────────────────────
# Extract age grid for Z(t) tracks (in Gyr, present = 0)
# ─────────────────────────────────────────────────────────────────────────
# SSP data stores ages in log10(Gyr); convert to linear Gyr for interpolation
ssp_lg_ages_gyr = np.asarray(ssp.ssp_lg_age_gyr)
ssp_ages_gyr = 10.0 ** ssp_lg_ages_gyr
ssp_ages_yr = ssp_ages_gyr * 1e9
lookback_time_gyr = AGE_UNIVERSE_GYR - ssp_ages_gyr

# ─────────────────────────────────────────────────────────────────────────
# Reconstruct Z(t) for each scenario (not yet exposed as a public method,
# so we manually apply the transformations)
# ─────────────────────────────────────────────────────────────────────────

# Scenario 1: constant
z_delta = np.full_like(lookback_time_gyr, 10.0 ** p_delta["met_logzsol"])

# Scenario 2: linear ramp (as a function of SSP age)
z_0 = 10.0 ** p_ramp["met_logzsol_0"]  # Absolute Z at earliest time
z_final = 10.0 ** p_ramp["met_logzsol_final"]  # Absolute Z at present day
# Linear interpolation from age_yr=0 to age_yr=max
age_max = ssp_ages_yr.max()
z_ramp = z_0 + (z_final - z_0) * (ssp_ages_yr / age_max)
z_ramp_logzsol = np.log10(z_ramp) - np.log10(1.0)  # Convert back to log(Z/Zsun)

# Scenario 3: two-step at lookback time
step_age_gyr = p_twostep["met_step_age_gyr"]
step_age_yr = step_age_gyr * 1e9
z_old = 10.0 ** p_twostep["met_logzsol_old"]
z_young = 10.0 ** p_twostep["met_logzsol_young"]
z_twostep = np.where(
    ssp_ages_yr >= step_age_yr,
    z_old,
    z_young,
)

# ─────────────────────────────────────────────────────────────────────────
# Create figure with two panels: Z(t) and resulting SED
# ─────────────────────────────────────────────────────────────────────────

fig, (ax_zt, ax_sed) = plt.subplots(2, 1, figsize=(7.0, 6.5))

# Color scheme
colors = {
    "delta": "#1f77b4",
    "ramp": "#ff7f0e",
    "twostep": "#2ca02c",
}

# ─────────────────────────────────────────────────────────────────────────
# Panel 1: Z(t) evolution
# ─────────────────────────────────────────────────────────────────────────

ax_zt.plot(
    lookback_time_gyr,
    z_delta,
    color=colors["delta"],
    lw=2.0,
    label="Constant Z (0.0 dex)",
)

ax_zt.plot(
    lookback_time_gyr,
    z_ramp,
    color=colors["ramp"],
    lw=2.0,
    label="Ramp: -1.0 → 0.0 dex",
)

ax_zt.plot(
    lookback_time_gyr,
    z_twostep,
    color=colors["twostep"],
    lw=2.0,
    label="Two-step (step at 8 Gyr)",
)

ax_zt.axhline(1.0, color="0.5", lw=0.8, ls="--", alpha=0.5)
ax_zt.text(13, 1.02, r"Solar $Z_\odot$", fontsize=9, alpha=0.6)

ax_zt.set(
    xlabel=r"Lookback time [Gyr]",
    ylabel=r"Metallicity $Z$ [$Z_\odot$]",
    xlim=(0, AGE_UNIVERSE_GYR),
    ylim=(0.05, 1.5),
)
ax_zt.legend(frameon=False, fontsize=9, loc="upper left")
ax_zt.grid(True, alpha=0.25, which="major")

# ─────────────────────────────────────────────────────────────────────────
# Panel 2: Resulting integrated SED
# ─────────────────────────────────────────────────────────────────────────

ax_sed.loglog(
    wave_delta,
    sed_delta,
    color=colors["delta"],
    lw=2.0,
    label="Constant Z",
)

ax_sed.loglog(
    wave_ramp,
    sed_ramp,
    color=colors["ramp"],
    lw=2.0,
    label="Ramp",
)

ax_sed.loglog(
    wave_twostep,
    sed_twostep,
    color=colors["twostep"],
    lw=2.0,
    label="Two-step",
)

ax_sed.set(
    xlabel=r"Rest-frame wavelength [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$ [erg/s]",
    xlim=(500, 1e5),
)
ax_sed.legend(frameon=False, fontsize=9, loc="upper right")
ax_sed.grid(True, alpha=0.25, which="both", axis="both")

fig.tight_layout()

# Save figure to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "plot_chemical_evolution_ramp.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
