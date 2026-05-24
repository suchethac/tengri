r"""
Modified blackbody dust temperature vs infrared luminosity
===========================================================

Show the L_IR–T_dust correlation for synthetic main-sequence galaxies.

For star-forming galaxies, the dust temperature and infrared luminosity are
tightly correlated: higher infrared luminosity (driven by higher SFR) is
associated with cooler average dust temperature, while lower-luminosity
starbursts show warmer dust. This example demonstrates the trend using a
suite of synthetic galaxies where dust temperature is manually varied across
the SFR grid.

We construct 10 synthetic galaxies spanning a SFR grid, each with a
modified-blackbody dust emission model with temperature co-varying with SFR,
predict the rest-frame SED, integrate L_IR in the 8–1000 μm band, and
visualize the correlation. Points are colored by SFR and overlaid against
the Symeonidis+2013 empirical fit.

References: Symeonidis+2013 (MNRAS 431, 2317), Casey+2014 (Physics Reports 541, 45).

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dust_temperature_vs_lir_001.png
   :alt: plot_dust_temperature_vs_lir
   :class: sphx-glr-single-img

"""

import os
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    load_ssp_data,
)
from tengri import cosmology, units
from tengri.analysis.plotting import setup_style

setup_style()

# Load SSP grid
_ssp_name = "fsps_prsc_miles_chabrier.h5"
_repo_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / _ssp_name).exists()
)
ssp = load_ssp_data(str(_repo_root / "data" / _ssp_name))

# Minimal photometry to speed up forward model
photometry = Photometry.from_names([
    "galex_nuv",  # UV constraint
    "sdss_g",     # Optical
    "sdss_i",     # Near-IR
    "wise_w3",    # MIR
])
observation = Observation(photometry=photometry)

# Galaxy parameters: redshift, stellar mass (fixed)
z_gal = 0.05
m_star = 1e10  # Msun (stellar mass, held constant)

# Star formation rates: logarithmic grid from 0.1 to 100 Msun/yr
# Reduced to 10 points for faster execution
sfr_grid = np.logspace(-1, 2, 10)

# Storage for results
lir_values = []  # [erg/s] at rest-frame
tdust_values = []  # [K]
sfr_values = []  # [Msun/yr]

print("Building synthetic galaxy grid...")
print(f"Redshift: z = {z_gal}")
print(f"SFR range: {sfr_grid[0]:.2e} – {sfr_grid[-1]:.2e} Msun/yr\n")

for idx, sfr in enumerate(sfr_grid):
    if (idx + 1) % 3 == 0 or idx == 0:
        print(f"  {idx + 1}/{len(sfr_grid)}")

    # Set dust temperature to co-vary with SFR: hotter dust at lower SFR,
    # cooler at higher SFR (realistic for local main-sequence galaxies).
    # Simple scaling: T_dust = 50 - 5 * log10(SFR)
    log_sfr = np.log10(sfr)
    tdust_model = 50.0 - 5.0 * log_sfr  # Hotter at low SFR, cooler at high SFR

    # Build model with modified-blackbody dust emission
    groups = {
        "sfh": {
            "type": "dpl",
            "log_peak_sfr": Fixed(log_sfr),
            "alpha": Fixed(1.0),
            "beta": Fixed(1.0),
            "tau_gyr": Fixed(1.0),
        },
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "tau_bc": Fixed(0.3),
            "tau_diff": Fixed(0.1),
            "slope": Fixed(-0.7),
            "emission": {
                "type": "modified_blackbody",
                "T": Fixed(tdust_model),  # Vary with SFR
                "beta_ir": Fixed(1.8),
            },
        },
        "neb": {"type": "cue"},
        "redshift": Fixed(z_gal),
        "apply_igm": False,
    }

    spec = Parameters.from_groups(**groups)
    model = SEDModel(spec, ssp, observation=observation)

    # Truth dict
    truth = {
        "sfh_dpl_log_peak_sfr": log_sfr,
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 1.0,
        "met_logzsol": -0.1,
        "dust_tau_bc": 0.3,
        "dust_tau_diff": 0.1,
        "dust_slope": -0.7,
        "dust_T": tdust_model,
        "dust_beta_ir": 1.8,
        "redshift": z_gal,
    }

    # Predict rest-frame SED
    sed = model.predict_rest_sed(truth)
    wave_rest_aa = np.asarray(sed.wavelength)
    lnu_rest = np.asarray(sed.sed)

    # Integrate L_IR in rest-frame 8–1000 μm band [erg/s]
    c_aa_per_s = 2.99792458e18
    mask_ir = (wave_rest_aa >= 8.0e4) & (wave_rest_aa <= 1.0e7)

    if np.sum(mask_ir) > 1:
        wave_ir = wave_rest_aa[mask_ir]
        nu_ir = c_aa_per_s / wave_ir
        lnu_ir = lnu_rest[mask_ir]

        # Sort by increasing frequency for integration
        sort_idx = np.argsort(nu_ir)
        nu_ir_sorted = nu_ir[sort_idx]
        lnu_ir_sorted = lnu_ir[sort_idx]

        lir = np.trapz(lnu_ir_sorted, nu_ir_sorted)
    else:
        lir = np.trapz(lnu_rest[mask_ir], wave_rest_aa[mask_ir])

    # Store results
    lir_values.append(lir)
    tdust_values.append(tdust_model)
    sfr_values.append(sfr)

lir_values = np.array(lir_values)
tdust_values = np.array(tdust_values)
sfr_values = np.array(sfr_values)

# Compute L_IR in Lsun (L_sun = 3.839e33 erg/s)
lsun = 3.839e33
lir_lsun = lir_values / lsun

print(f"\nResults:")
print(f"  L_IR range: {lir_lsun.min():.2e} – {lir_lsun.max():.2e} L_sun")
print(f"  T_dust range: {tdust_values.min():.1f} – {tdust_values.max():.1f} K")

# Plot
fig, ax = plt.subplots(figsize=(9, 6))

# Scatter plot: T_dust vs log L_IR, colored by SFR
scatter = ax.scatter(
    np.log10(lir_lsun),
    tdust_values,
    c=np.log10(sfr_values),
    s=80,
    alpha=0.6,
    cmap="viridis",
    edgecolors="black",
    linewidth=0.8,
)

# Symeonidis+2013 empirical fit: T = 27 + 6.5 * (log L_IR - 11.5) K
lir_range = np.logspace(10, 13, 100)
t_sym = 27.0 + 6.5 * (np.log10(lir_range) - 11.5)
ax.plot(
    np.log10(lir_range),
    t_sym,
    "r--",
    linewidth=2.0,
    label="Symeonidis+2013 fit",
    alpha=0.8,
)

ax.set_xlabel(r"$\log_{10}(L_{\rm IR} / L_\odot)$", fontsize=11)
ax.set_ylabel(r"$T_{\rm dust}$ [K]", fontsize=11)
ax.set_xlim(9.5, 13.5)
ax.set_ylim(20, 50)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=False, fontsize=10)

# Colorbar: SFR
cbar = plt.colorbar(scatter, ax=ax, label=r"$\log_{10}(\mathrm{SFR} / M_\odot \, \mathrm{yr}^{-1})$")

fig.tight_layout()
plt.savefig("plot_dust_temperature_vs_lir.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved: plot_dust_temperature_vs_lir.png")
