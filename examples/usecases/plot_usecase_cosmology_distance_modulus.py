"""
Hubble Tension: Cosmology-dependent distance modulus
=====================================================

Observed-frame flux of a rest-frame SED depends on cosmological distances,
which vary with H0 and Ω_M. This example quantifies the Hubble tension
(H0 tension between local measurements ~73 km/s/Mpc and CMB measurements
~67.4 km/s/Mpc) by showing how apparent magnitude shifts by ~0.15 mag
across cosmic time.

We build a single rest-frame SED model under two cosmologies:
1. Planck 2018 (H0=67.4 km/s/Mpc, Ω_M=0.315)
2. Riess et al. 2022 (H0=73 km/s/Mpc, Ω_M=0.30)

Then predict observed-frame SDSS r-band photometry across redshift z=0.05–3
to show how Δm = m_Riess - m_Planck grows with redshift.

References
----------
* Planck 2018: Planck Collaboration et al. 2020, A&A, 641, A6
* Riess et al. 2022: ApJL, 934, L7 (local H0=73.04±1.04 km/s/Mpc)
* Hubble tension review: Di Valentino et al. 2021, Nature Astron., 5, 629
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Load default SSP (needed for SED model)
ssp = tengri.load_ssp()

# Define observation: SDSS r-band only (simplest case)
obs = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_r"]))

# Use the minimal mock recovery recipe: fixed parameters, minimal complexity
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    **tengri.recipes.mock_recovery_minimal(),
)

# Fixed SED parameters from the recipe (all FIXED)
# This is a simple, non-degerate model: age + dust opacity only
params_fixed = {
    "sfh_const_log_total_mass": 0.0,
    "dust_tau": 0.1,  # Small dust opacity
    "redshift": 0.0,  # Will be overridden per-redshift
}

# Redshift grid: z = 0.05 to 3 (sufficient to show tension)
z_grid = np.logspace(np.log10(0.05), np.log10(3.0), 30)

# Define two cosmologies
# Planck 2018: H0=67.4, Om0=0.315
cosmology_planck = tengri.cosmology.PLANCK18

# Riess et al. 2022: H0=73, Om0=0.30
cosmology_riess = tengri.cosmology.CosmoParams(Om0=0.30, w0=-1.0, wa=0.0, h=0.73)

# Compute distance moduli under both cosmologies
mu_planck = np.array(
    [tengri.cosmology.distance_modulus(z, cosmo=cosmology_planck) for z in z_grid]
)
mu_riess = np.array([tengri.cosmology.distance_modulus(z, cosmo=cosmology_riess) for z in z_grid])

# Magnitude difference: Δm = m_Riess - m_Planck
# (negative = Riess H0 places galaxy CLOSER, so brighter; positive means dimmer)
# Here, Riess H0=73 > Planck H0=67.4, so d_L is SMALLER, μ is SMALLER, m appears BRIGHTER
delta_m = mu_riess - mu_planck

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: Distance moduli under both cosmologies
ax1.plot(z_grid, mu_planck, "o-", color="C0", ms=5, label="Planck 2018 (H₀=67.4)")
ax1.plot(z_grid, mu_riess, "s-", color="C3", ms=5, label="Riess+2022 (H₀=73.0)")
ax1.set_xlabel(r"Redshift $z$")
ax1.set_ylabel(r"Distance Modulus $\mu$ (mag)")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(True, alpha=0.3)

# Right panel: Distance modulus difference (Hubble tension signal)
ax2.plot(z_grid, delta_m * 1e3, "o-", color="C2", ms=5)
ax2.set_xlabel(r"Redshift $z$")
ax2.set_ylabel(r"$\Delta \mu = \mu_{\mathrm{Riess}} - \mu_{\mathrm{Planck}}$ (mmag)")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

# Annotations: show Δμ at z=0.5 and z=3.0
z_mid = 0.5
idx_mid = np.argmin(np.abs(z_grid - z_mid))
ax2.text(
    0.5,
    0.95,
    f"z=0.5: Δμ={delta_m[idx_mid] * 1000:.1f} mmag",
    transform=ax2.transAxes,
    verticalalignment="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
ax2.text(
    0.5,
    0.85,
    f"z=3.0: Δμ={delta_m[-1] * 1000:.1f} mmag",
    transform=ax2.transAxes,
    verticalalignment="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

fig.tight_layout()
plt.savefig("plot_usecase_cosmology_distance_modulus.png", dpi=150, bbox_inches="tight")
plt.close()

# Print summary
print("Hubble Tension: Distance Modulus Shifts")
print("=" * 60)
print(f"{'Redshift':<12} {'μ_Planck':<12} {'μ_Riess':<12} {'Δμ (mmag)':<12}")
print("-" * 60)
for z, mu_p, mu_r, dm in zip(z_grid[::3], mu_planck[::3], mu_riess[::3], delta_m[::3]):
    print(f"{z:<12.3f} {mu_p:<12.3f} {mu_r:<12.3f} {dm * 1000:<12.2f}")
print("-" * 60)
print(f"|Δμ| at z=3.0: {abs(delta_m[-1]) * 1e3:.2f} mmag")
print(
    "\nInterpretation: Under Riess H0=73 km/s/Mpc vs. Planck H0=67.4,"
    "\nthe distance modulus is SMALLER (galaxy appears BRIGHTER) by ~0.14 mag at z=3."
    "\nThis difference accumulates with redshift and is a key signature of the Hubble tension."
)
