"""
Alpha-element enhancement shifts absorption features in old stellar populations
================================================================================

The stellar populations in massive elliptical galaxies are typically
α-enhanced ([α/Fe] > 0) due to rapid star formation timescales that
terminate before iron-peak elements fully enrich the gas. Increasing [α/Fe]
shifts absorption-feature strengths — particularly Mg b and Fe5270 — which
serve as diagnostics of the galaxy's star-formation history timescale. We
sweep [α/Fe] from 0.0 to 0.6 at fixed age (5 Gyr) and solar metallicity,
showing the full rest-frame SED and a zoom on the optical feature region.

References: Thomas et al. 2005 (MNRAS 357, 1113); Conroy & van Dokkum 2012 (ApJ 747, 69).
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

# Configuration
AGE_GYR = 5.0
MET_LOGZSOL = 0.0
REDSHIFT = 0.05
ALPHA_FE_SWEEP = np.array([0.0, 0.2, 0.4, 0.6])

# Broad wavelength coverage for full SED + zoom region (rest-frame)
WAVE_REST_ANGSTROM = np.linspace(3000.0, 10000.0, 2048)

# Create observation with spectroscopy at rest-frame wavelengths
spec_config = tengri.Spectroscopy(wave_obs=WAVE_REST_ANGSTROM)
obs = tengri.Observation(spectroscopy=spec_config)

# Load SSP data
ssp = tengri.load_ssp()

# Build base model: fixed age (via tsnorm), fixed metallicity, varying [α/Fe]
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": AGE_GYR,
        "width_gyr": 0.01,  # Narrow burst to isolate age
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.5,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(REDSHIFT),
)

# Sample baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(42)))

# Loop over [α/Fe] and build parameter dicts
seds = {}
for alpha_fe in ALPHA_FE_SWEEP:
    p = {
        **baseline,
        "sfh_tsnorm_peak_lbt_gyr": jnp.float64(AGE_GYR),
        "met_logzsol": jnp.float64(MET_LOGZSOL),
        "met_alpha_fe": jnp.float64(alpha_fe),
    }
    # Predict spectrum (F_nu in erg/s/cm^2/Hz at observer frame)
    flux_nu = np.asarray(model.predict_spectrum(p, WAVE_REST_ANGSTROM))
    # Convert F_nu to νL_ν using d = 1 pc (arbitrary; focus on shape)
    # For demonstration, just use flux in units of relative νL_ν
    freq_rest = 3.0e18 / WAVE_REST_ANGSTROM  # Hz
    sed_nulnu = flux_nu * freq_rest
    seds[alpha_fe] = (WAVE_REST_ANGSTROM, sed_nulnu)

# Create figure with two panels
fig, axes = plt.subplots(
    2, 1, figsize=(10, 8), sharex=False, gridspec_kw={"height_ratios": [1, 1.2]}
)

# Panel 1: Full rest-frame SED
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(ALPHA_FE_SWEEP)))
for alpha_fe, color in zip(ALPHA_FE_SWEEP, colors):
    wave_rest, sed_nulnu = seds[alpha_fe]
    ax1.loglog(wave_rest, sed_nulnu, label=f"[α/Fe] = {alpha_fe:.1f}", color=color, linewidth=2)

ax1.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]", fontsize=10)
ax1.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=10)
ax1.legend(loc="upper right", fontsize=9, frameon=False)
ax1.grid(True, which="both", alpha=0.25)

# Panel 2: Zoom into Mg b / Fe5270 region (5050–5350 Å)
ax2 = axes[1]
zoom_min, zoom_max = 5050, 5350
for alpha_fe, color in zip(ALPHA_FE_SWEEP, colors):
    wave_rest, sed_nulnu = seds[alpha_fe]
    # Filter to zoom region
    mask = (wave_rest >= zoom_min) & (wave_rest <= zoom_max)
    wave_zoom = wave_rest[mask]
    sed_zoom = sed_nulnu[mask]

    # Normalize by continuum for clarity (normalize to mean in region)
    continuum_level = np.median(sed_zoom)
    sed_norm = sed_zoom / continuum_level

    ax2.plot(
        wave_zoom,
        sed_norm,
        label=f"[α/Fe] = {alpha_fe:.1f}",
        color=color,
        linewidth=2.0,
    )

# Add reference lines for Mg b and Fe5270
ax2.axvline(5175, color="gray", linestyle="--", alpha=0.6, linewidth=1)
ax2.text(5175, 0.965, "Mg b", fontsize=9, ha="center", color="gray")
ax2.axvline(5270, color="gray", linestyle="--", alpha=0.6, linewidth=1)
ax2.text(5270, 0.965, "Fe5270", fontsize=9, ha="center", color="gray")

ax2.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]", fontsize=10)
ax2.set_ylabel(r"$\nu L_\nu$ (normalized)", fontsize=10)
ax2.set_xlim(zoom_min, zoom_max)
ax2.grid(True, alpha=0.25)
ax2.legend(loc="lower right", fontsize=9, frameon=False)

fig.tight_layout()
plt.savefig("plot_alpha_enhanced_population.png", dpi=150, bbox_inches="tight")
for alpha_fe in ALPHA_FE_SWEEP:
    print(f"  [α/Fe] = {alpha_fe:.1f} dex (typical for elliptical: 0.0–0.6)")
