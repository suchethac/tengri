"""
Workflow: BPT Emission-Line Classification
=============================================

Demonstrates computing emission-line ratios for a mock galaxy catalog
with mixed star-forming and AGN fractions. Plots the BPT diagram
([OIII]/Hβ vs [NII]/Hα) and overlays Kewley+2001 and Kauffmann+2003
demarcation lines to show how emission-line diagnostics separate
ionization mechanisms.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_workflow_bpt_classification_001.png
   :alt: plot_workflow_bpt_classification
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fixed,
    Parameters,
    SEDModel,
    Uniform,
    load_ssp,
)
from tengri.analysis.plotting import setup_style

setup_style()


# --- SSP data ---


ssp = load_ssp()

# --- Build model ---
z = 0.1

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
    sfh_tsnorm_width_gyr=Fixed(1.0),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.1),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(z),
    mean_sfh_type="tsnorm",
)

model = SEDModel(spec, ssp)

# --- Generate mock catalog with AGN fractions ---
n_gal = 20
key = jax.random.PRNGKey(123)
agn_fracs = np.linspace(0.0, 0.8, n_gal)  # AGN contribution to ionization

log_nii_ha = []
log_oiii_hb = []

for agn_frac in agn_fracs:
    # Sample a star-forming galaxy, modulate AGN fraction
    key, subkey = jax.random.split(key)
    true_params = spec.sample(subkey)
    true_params["sfh_tsnorm_log_peak_sfr"] = 0.5  # Moderate SFR
    true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0

    # Generate synthetic emission line ratios via simple power-law approximation
    # (emission lines proportional to SFR and metallicity)
    sfr_peak = float(true_params["sfh_tsnorm_log_peak_sfr"])

    # Synthetic line fluxes (relative to H-alpha)
    ha = 1.0  # Normalized
    hb = 0.3  # Typical ratio
    nii = 0.1 * (1.0 + sfr_peak)  # Metallicity-sensitive
    oiii = 0.2 * (1.0 + sfr_peak)

    # Add AGN-like boost to ionization: AGN primarily enhances [OIII]
    if agn_frac > 0:
        oiii = oiii * (1.0 + 3.0 * agn_frac)  # AGN strengthens [OIII]
        nii = nii * (1.0 + 1.5 * agn_frac)  # and [NII] slightly

    if ha > 1e-12 and hb > 1e-12:
        log_nii_ha.append(np.log10(max(nii / ha, 1e-3)))
        log_oiii_hb.append(np.log10(max(oiii / hb, 1e-3)))

log_nii_ha = np.array(log_nii_ha)
log_oiii_hb = np.array(log_oiii_hb)

# --- Plot BPT diagram ---
fig, ax = plt.subplots(figsize=(8, 7))

# Kewley+2001 maximum starburst line
log_nii_ha_grid = np.linspace(-1.6, 0.5, 300)
log_oiii_hb_kewley = 0.61 / (log_nii_ha_grid - 0.47) + 1.19

# Kauffmann+2003 empirical SF line
log_oiii_hb_kauff = 0.61 / (log_nii_ha_grid - 0.05) + 1.3

mask_k = log_nii_ha_grid < 0.47
ax.plot(
    log_nii_ha_grid[mask_k],
    log_oiii_hb_kewley[mask_k],
    "k-",
    lw=2.0,
    label="Kewley+2001 (max starburst)",
)
mask_kauff = log_nii_ha_grid < 0.05
ax.plot(
    log_nii_ha_grid[mask_kauff],
    log_oiii_hb_kauff[mask_kauff],
    "k--",
    lw=1.8,
    label="Kauffmann+2003 (empirical SF)",
)

# Region labels
ax.text(-1.35, -0.6, "Star\nForming", fontsize=11, color="#1f77b4", fontweight="bold", ha="center")
ax.text(-0.1, 0.7, "Composite", fontsize=11, color="#ff7f0e", fontweight="bold", ha="center")
ax.text(
    0.25, 1.15, "Seyfert/\nLINER", fontsize=11, color="#d62728", fontweight="bold", ha="center"
)

# Galaxy catalog colored by AGN fraction
sc = ax.scatter(
    log_nii_ha,
    log_oiii_hb,
    c=agn_fracs[: len(log_nii_ha)],
    cmap="viridis",
    s=80,
    zorder=5,
    edgecolors="k",
    lw=0.5,
    label="Mock galaxies",
)
cbar = plt.colorbar(sc, ax=ax, label="AGN fraction")
cbar.ax.tick_params(labelsize=10)

ax.set_xlabel(r"log [NII]$\lambda$6583 / H$\alpha$", fontsize=12)
ax.set_ylabel(r"log [OIII]$\lambda$5007 / H$\beta$", fontsize=12)
ax.set_title("BPT Diagram: Star Formation vs AGN Classification", fontsize=12, fontweight="bold")
ax.set_xlim(-1.6, 0.6)
ax.set_ylim(-1.2, 1.5)
ax.legend(fontsize=10, frameon=False, loc="lower left")
fig.tight_layout()
plt.savefig("plot_workflow_bpt_classification.png", dpi=150, bbox_inches="tight")
plt.show()
