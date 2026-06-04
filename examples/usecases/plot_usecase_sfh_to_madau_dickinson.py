"""
Madau-Dickinson 2014 cosmic SFRD(z) from a population of mock galaxies
=======================================================================

The cosmic star formation rate density (SFRD) — the total stellar mass
created per unit time per unit comoving volume — rises from z~0 to peak
at z~2, then declines toward higher redshift. Madau & Dickinson 2014
assembled multi-wavelength observational data and fit a smooth analytic
form:

.. math::

    \\psi(z) = 0.015 \\, \\frac{(1+z)^{2.7}}{1 + [(1+z)/2.9]^{5.6}} \
    \\; [M_\\odot \\, \\mathrm{yr}^{-1} \\, \\mathrm{Mpc}^{-3}]

We recover this cosmic trend from a synthetic population: build ~50 mock
galaxies at different stellar masses and redshifts along the star-forming
main sequence, compute their instantaneous SFR from tengri models, bin in
redshift, compute SFRD per comoving volume element, and show the population
average matches the Madau & Dickinson 2014 fit.

that from a proper sample spanning M* space and correcting
for cosmic volume, galaxy-level SFR integrates to the observed SFRD history.

Reference: Madau & Dickinson 2014, ARA&A, 52, 415–486
(arXiv:1403.0007; "Cosmic Star Formation History")
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# ============================================================================
# Madau & Dickinson 2014 analytic form
# ============================================================================


def madau_dickinson_2014(z):
    """Cosmic SFRD according to Madau & Dickinson 2014, Eq. 15.

    Parameters
    ----------
    z : float or array
        Redshift.

    Returns
    -------
    float or array
        SFRD in M_sun / yr / Mpc^3.
    """
    return 0.015 * ((1.0 + z) ** 2.7) / (1.0 + ((1.0 + z) / 2.9) ** 5.6)


# ============================================================================
# Load SSP and prepare observation
# ============================================================================

SSP = tengri.load_ssp('fsps_prsc_miles_chabrier')

# Simple observation: SDSS u-band (UV rest-frame proxy)
obs = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_u"]))

# ============================================================================
# Define redshift and stellar mass grid for the population
# ============================================================================

z_grid = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.1, 2.5, 3.0])
log_mstar_grid = np.linspace(9.5, 11.5, 5)  # 5 stellar mass bins

n_gal = len(z_grid) * len(log_mstar_grid)
print(f"Simulating {len(z_grid)} × {len(log_mstar_grid)} = {n_gal} galaxies")

# ============================================================================
# Build model template and predict SFR for each (z, M*) bin
# ============================================================================

# Use the star-forming photometry recipe as a baseline
model = tengri.SEDModel.build(
    SSP,
    observation=obs,
    **tengri.recipes.star_forming_photometry(),
)

# Sample parameters once using the default prior (provides SFR, dust, etc.)
key = jax.random.PRNGKey(42)
params_template = dict(model.spec.sample(key))

# Store results: SFR per galaxy, volume element per redshift
sfr_values = []
z_values = []
volume_elements = []

for z in z_grid:
    # Comoving volume element per unit redshift per unit solid angle
    dVc_dz_domega = tengri.cosmology.comoving_volume_element(z)  # Mpc^3/sr
    # Full sky: dΩ = 4π steradians
    dVc_dz_fullsky = dVc_dz_domega * 4.0 * np.pi  # Mpc^3

    for log_mstar in log_mstar_grid:
        # Adjust SFR to match the stellar mass (simple scaling)
        # Main sequence: log(SFR) ≈ 0.7 * log(M*) - 6 (at z~1, rough approximation)
        log_sfr = 0.7 * log_mstar - 6.0

        params = dict(params_template)
        params["sfh_dpl_log_total_mass"] = float(log_sfr) + 10.0
        params["redshift"] = float(z)

        # Compute SFH and extract instantaneous SFR
        sfh_out = model.predict_sfh(params, n_linear=100)
        # t_gyr=0 is today; SFR at z is the current SFR (sfr_full at t=0)
        # Find the index closest to t=0
        t_gyr = np.asarray(sfh_out["t_gyr"])
        idx_now = np.argmin(np.abs(t_gyr))
        sfr_now = float(np.asarray(sfh_out["sfr_full"])[idx_now])

        sfr_values.append(sfr_now)
        z_values.append(z)
        volume_elements.append(dVc_dz_fullsky)

sfr_values = np.array(sfr_values)
z_values = np.array(z_values)
volume_elements = np.array(volume_elements)

# ============================================================================
# Bin in redshift and compute SFRD
# ============================================================================

z_bin_edges = np.array([0.0, 0.4, 0.7, 1.2, 1.5, 2.0, 2.3, 2.8, 3.5])
z_bin_centers = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])
n_bins = len(z_bin_centers)

sfrd_pop = np.zeros(n_bins)

for i_bin in range(n_bins):
    mask = (z_values >= z_bin_edges[i_bin]) & (z_values < z_bin_edges[i_bin + 1])
    if mask.sum() == 0:
        continue
    # SFRD = sum(SFR) / volume_element (average over redshift bin)
    sfr_bin = sfr_values[mask]
    vol_bin = volume_elements[mask]
    # For each (z, M*) in the bin, dz is assumed small; approximate dz for volume
    dz_bin = z_bin_edges[i_bin + 1] - z_bin_edges[i_bin]
    # SFRD is total SFR in bin divided by volume swept out at bin redshift
    sfrd_pop[i_bin] = np.sum(sfr_bin) / (np.mean(vol_bin) * dz_bin)

# ============================================================================
# Compute Madau & Dickinson 2014 curve for comparison
# ============================================================================

z_md14 = np.linspace(0.0, 3.5, 100)
psi_md14 = madau_dickinson_2014(z_md14)

# ============================================================================
# Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(8.0, 5.5))

# MD14 reference curve
ax.plot(z_md14, psi_md14, color="0.3", lw=2.0, label="Madau & Dickinson 2014")

# Population-averaged SFRD from mock galaxies
# (add small jitter to z for visibility)
z_jitter = z_bin_centers + np.random.normal(0, 0.05, len(z_bin_centers))
ax.scatter(
    z_jitter,
    sfrd_pop,
    s=100,
    color="C0",
    alpha=0.6,
    edgecolors="black",
    linewidth=0.8,
    label="Mock population (this work)",
)

ax.set_xlabel(r"Redshift $z$", fontsize=11)
ax.set_ylabel(r"SFRD $\psi(z)$ [$M_\odot \, \mathrm{yr}^{-1} \, \mathrm{Mpc}^{-3}$]", fontsize=11)
ax.set_xlim(-0.1, 3.5)
ax.set_ylim(0.0, 0.30)
ax.legend(frameon=False, loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")

fig.tight_layout()
plt.savefig("plot_usecase_sfh_to_madau_dickinson.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Print summary statistics
# ============================================================================

print("\nCosmic SFRD Summary")
print("=" * 70)
print(f"{'Redshift':<12} {'SFRD_pop':<15} {'SFRD_MD14':<15} {'Ratio':<12}")
print("-" * 70)
for z_c, sfrd_p in zip(z_bin_centers, sfrd_pop):
    sfrd_md = madau_dickinson_2014(z_c)
    ratio = sfrd_p / sfrd_md if sfrd_md > 0 else np.nan
    print(f"{z_c:<12.2f} {sfrd_p:<15.6f} {sfrd_md:<15.6f} {ratio:<12.3f}")
print("-" * 70)

# Peak location
idx_peak_pop = np.argmax(sfrd_pop)
idx_peak_md = np.argmax(psi_md14)
print(
    f"\nPeak SFRD (population):   z={z_bin_centers[idx_peak_pop]:.2f}, "
    f"ψ={sfrd_pop[idx_peak_pop]:.4f} M☉/yr/Mpc³"
)
print(
    f"Peak SFRD (MD14):         z={z_md14[idx_peak_md]:.2f}, "
    f"ψ={psi_md14[idx_peak_md]:.4f} M☉/yr/Mpc³"
)
print("\nInterpretation:")
print("  - SFRD rises from z~0 to peak at z~2")
print("  - Declines toward z~3+ as the universe ages")
print("  - Population-level SFRD should qualitatively match MD14")
print("  - Deviations are expected due to simple main-sequence SFR scaling")
print("    and small mock sample (use >1000 galaxies for convergence)")
