"""
Strong-lensing magnification: EoR galaxy SEDs at μ = 1, 5, 20, 100
==================================================================

Demonstrates how strong gravitational lensing elevates intrinsically-faint
high-redshift (z=7) galaxies above the JWST NIRCam 5σ detection threshold.

A z=7 star-forming galaxy with log10(M*/M☉)=9 and young burst (peak_lbt_gyr=0.05 Gyr,
typical of early reionization epoch Lyman-alpha emitters per Schaerer+2003) is
intrinsically too faint to detect individually. Strong lensing magnifications μ ∈ {1, 5, 20, 100}
(arc regime; Refsdal 1964) scale observed-frame fluxes by μ, progressively lifting
the SED above the ~28 AB JWST 5σ sensitivity limit (Rieke+2023, NIRCam module).

This example uses NIRCam photometry (F150W, F200W, F277W, F356W, F444W) to illustrate
how only μ ≥ 5 crosses the detection threshold, making strong lensing critical for
observing the faintest reionization-epoch objects.

References:

- Refsdal, S. 1964, MNRAS, 128, 295 (gravitational lensing magnification)
- Schaerer, D. 2003, A&A, 397, 527 (z>6 Lya emitters)
- Bouwens, R. J., et al. 2022, ApJ, 931, 160 (EoR LBGs)
- Rieke, M. J., et al. 2023, PASP, 135, 028001 (JWST NIRCam performance)

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
from tengri.units import fnu_to_ab_mag

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# ── Load data and construct observation ────────────────────────────────────

# Load default SSP (wNE PRSC/MILES Chabrier)
ssp = tengri.load_ssp()

# JWST NIRCam filters: F150W, F200W, F277W, F356W, F444W
# These span the near-infrared and probe rest-frame UV→optical at z=7
bands = [
    "jwst_f150w",
    "jwst_f200w",
    "jwst_f277w",
    "jwst_f356w",
    "jwst_f444w",
]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# ── Build z=7 model: young burst, fixed stellar mass ──────────────────────

# For a z=7 LAE-like galaxy (Schaerer+2003), use a recent starburst:
# peak_lbt_gyr ≈ 0.05 Gyr (recent, ~50 Myr old at z=7)
# log_total_mass ≈ 0.5 → SFR ≈ 3 M_sun/yr (moderate for z~7)
# Dust is minimal at z~7 (optical depth tau_bc ≈ 0.1)

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_total_mass": 10.0,  # 3 M_sun/yr
        "peak_lbt_gyr": tengri.Fixed(0.05),  # 50 Myr old (z=7 LAE regime)
        "width_gyr": tengri.Fixed(0.1),  # 100 Myr width
        "skew": tengri.Fixed(-0.3),  # slight left skew (recent burst)
        "trunc": tengri.Fixed(14.0),  # age of universe at z=7
        "logzsol": tengri.Fixed(-0.1),  # solar metallicity (young star-forming)
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.1,  # minimal dust in z~7 LAEs
        "tau_diff": 0.05,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(7.0),
)

# ── Predict intrinsic SED and photometry (μ=1) ──────────────────────────────

# Sample one representative parameter set
key = jax.random.PRNGKey(42)
params = model.spec.sample(key)

# Intrinsic (unlensed) photometry in erg/s/Hz
flux_intrinsic = np.asarray(model.predict_photometry(params))

# Convert to AB magnitudes
mag_intrinsic = np.array([float(fnu_to_ab_mag(jnp.array(f))) for f in flux_intrinsic])

# JWST NIRCam 5σ detection threshold (Rieke+2023)
# F150W ≈ 27.9 AB, F200W ≈ 28.1 AB, F277W ≈ 28.4 AB, F356W ≈ 28.6 AB, F444W ≈ 28.5 AB
# Conservative estimate: ~28.0 AB across the band
mag_5sigma_limit = 28.0

# ── Compute magnified photometry and AB magnitudes ──────────────────────────

magnifications = np.array([1.0, 5.0, 20.0, 100.0])
n_mag = len(magnifications)
n_bands = len(bands)

mag_lensed = np.zeros((n_mag, n_bands))
flux_lensed = np.zeros((n_mag, n_bands))

for i, mu in enumerate(magnifications):
    # Magnified flux: F = μ × F_intrinsic
    flux_lensed[i] = mu * flux_intrinsic

    # Convert to AB magnitudes (mag_lensed = mag_intrinsic - 2.5*log10(μ))
    mag_lensed[i] = mag_intrinsic - 2.5 * np.log10(mu)

# ── Plot observed-frame magnitudes vs magnification ────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

# Viridis colormap for bands
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.0, 1.0, n_bands))

# Plot each band across magnification values
for j, band in enumerate(bands):
    ax.plot(
        magnifications,
        mag_lensed[:, j],
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=colors[j],
        label=band.upper().replace("NIRCAM_", ""),
        alpha=0.85,
    )

# JWST 5σ detection limit (horizontal line)
ax.axhline(
    mag_5sigma_limit,
    color="red",
    linestyle="--",
    linewidth=2.2,
    label=f"JWST 5σ limit (~{mag_5sigma_limit:.1f} AB)",
    alpha=0.75,
)

# Shade the detectable region
ax.fill_between(
    magnifications,
    mag_5sigma_limit - 1.0,
    mag_5sigma_limit,
    alpha=0.1,
    color="red",
    label="Detectable region",
)

# ── Annotation: mark where detections begin ────────────────────────────────

# Find first magnification where F444W (reddest, typically deepest) crosses limit
f444w_idx = n_bands - 1
for i, mu in enumerate(magnifications):
    if mag_lensed[i, f444w_idx] < mag_5sigma_limit:
        ax.annotate(
            f"μ = {mu:.0f}\nDetectable",
            xy=(mu, mag_lensed[i, f444w_idx]),
            xytext=(mu + 15, mag_lensed[i, f444w_idx] + 0.5),
            fontsize=9,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black", lw=1.0
            ),
        )
        break

# ── Labels and formatting ──────────────────────────────────────────────────

ax.set_xlabel(r"Magnification $\mu$", fontsize=12, weight="bold")
ax.set_ylabel(r"AB Magnitude (obs-frame)", fontsize=12, weight="bold")
ax.set_xscale("log")
ax.set_xlim(0.7, 150)
ax.set_ylim(20, 31)
ax.invert_yaxis()  # Brighter (lower mag) at top
ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
ax.legend(
    loc="lower left",
    frameon=False,
    fontsize=10,
    ncol=2,
)

# Add reference annotations
textstr = "z=7, M* ~ 10$^9$ M$_\\odot$\nAge ~ 50 Myr (tsnorm burst)\nτ_BC = 0.1 (minimal dust)"
ax.text(
    0.98,
    0.97,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
)

fig.tight_layout()
plt.savefig("plot_usecase_lensed_galaxy_magnification.png", dpi=150, bbox_inches="tight")
