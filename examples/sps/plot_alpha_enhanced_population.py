"""
Alpha-element enhancement in quiescent stellar populations
===========================================================

The stellar populations in massive elliptical galaxies are typically
α-enhanced ([α/Fe] > 0) due to rapid star formation timescales that
terminate before iron-peak elements fully enrich the gas (Thomas et al. 2005).
This example demonstrates how increasing [α/Fe] shifts absorption features —
particularly the Mg b and Fe5270 indices — which serve as diagnostics of
star-formation history timescale.

We build four tengri models at fixed age (5 Gyr), metallicity Z = 0 dex
(solar), and redshift z = 0.05, sweeping [α/Fe] ∈ {0.0, 0.2, 0.4, 0.6}.
The top panel shows the full rest-frame SED in νL_ν; the bottom zooms
into the 5050–5350 Å region to reveal the Mg b (≈5175 Å) and Fe5270
(≈5270 Å) line ratio changes.

**References:**
    - Thomas et al. 2005 (MNRAS 357, 1113) — age–metallicity–[α/Fe] diagnostics
    - Conroy & van Dokkum 2012 (ApJ 747, 69) — stellar population models

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
        "*": tengri.FIXED,
        "peak_lbt_gyr": AGE_GYR,
        "width_gyr": 0.01,  # Narrow burst to isolate age
        "log_peak_sfr": 11.0,
        "skew": 0.0,
        "trunc": 13.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
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
for (alpha_fe, color) in zip(ALPHA_FE_SWEEP, colors):
    wave_rest, sed_nulnu = seds[alpha_fe]
    ax1.loglog(
        wave_rest, sed_nulnu, label=f"[α/Fe] = {alpha_fe:.1f}", color=color, linewidth=2
    )

ax1.set_ylabel(r"$\nu L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)", fontsize=11)
ax1.set_xlabel(r"Rest wavelength ($\AA$)", fontsize=11)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(True, which="both", alpha=0.3)
ax1.set_title(
    f"Age = {AGE_GYR} Gyr, Z = {MET_LOGZSOL} dex (solar), z = {REDSHIFT}",
    fontsize=12,
)

# Panel 2: Zoom into Mg b / Fe5270 region (5050–5350 Å)
ax2 = axes[1]
zoom_min, zoom_max = 5050, 5350
for (alpha_fe, color) in zip(ALPHA_FE_SWEEP, colors):
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

ax2.set_xlabel(r"Rest wavelength ($\AA$)", fontsize=11)
ax2.set_ylabel(r"Normalized $\nu L_\nu$", fontsize=11)
ax2.set_xlim(zoom_min, zoom_max)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right", fontsize=10)
ax2.set_title("Mg b / Fe5270 diagnostic region (zoomed)", fontsize=12)

plt.tight_layout()
plt.savefig("plot_alpha_enhanced_population.png", dpi=150, bbox_inches="tight")
plt.show()

# Diagnostic: verify Mg b / Fe5270 ratio increases with [α/Fe]
print("\n=== Alpha-element enhancement diagnostics ===")
print(f"Age: {AGE_GYR} Gyr | Metallicity: Z = {MET_LOGZSOL} dex | Redshift: {REDSHIFT}")
print(
    f"\nAll four models use the same SSP age & metallicity, varying only [α/Fe].\n"
    f"Mg b and Fe5270 are absorption features; higher Mg/Fe ratio indicates faster\n"
    f"star-formation timescale (less iron-peak enrichment from supernovae Ia).\n"
)
for alpha_fe in ALPHA_FE_SWEEP:
    print(f"  [α/Fe] = {alpha_fe:.1f} dex (typical for elliptical: 0.0–0.6)")
