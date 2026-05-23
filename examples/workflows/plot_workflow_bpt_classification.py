"""
BPT diagram: emission lines separate star-forming from AGN ionization
====================================================================

Demonstrates BPT ([OIII]/Hbeta vs [NII]/Halpha) line-ratio diagnostics
on a mock galaxy catalog with varying AGN fraction. Overlays Kewley+2001
and Kauffmann+2003 demarcation lines to show the clean separation of
ionization mechanisms across the diagnostic plane.

Reference: Kewley et al. 2001, ApJ, 556, 121 (theoretical classification);
Kauffmann et al. 2003, MNRAS, 346, 1055 (empirical SF boundary).
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
z = 0.1

# Build a simple model to generate mock SFH samples
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "log_peak_sfr": tengri.Uniform(-1.0, 2.5),
        "peak_lbt_gyr": tengri.Fixed(2.0),
        "width_gyr": tengri.Fixed(1.0),
        "skew": tengri.Fixed(0.2),
        "trunc": tengri.Fixed(3.0),
        "logzsol": tengri.Fixed(-0.2),
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_bc": 0.1,
        "tau_diff": 0.1,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(z),
)

# Generate mock catalog with varying AGN fractions
n_gal = 20
key = jax.random.PRNGKey(123)
agn_fracs = np.linspace(0.0, 0.8, n_gal)

log_nii_ha = []
log_oiii_hb = []

for agn_frac in agn_fracs:
    key, subkey = jax.random.split(key)
    params = model.spec.sample(subkey)
    params["sfh_tsnorm_log_peak_sfr"] = 0.5
    params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
    sfr_peak = float(params["sfh_tsnorm_log_peak_sfr"])

    # Synthetic line fluxes (relative to H-alpha)
    ha = 1.0
    hb = 0.3
    nii = 0.1 * (1.0 + sfr_peak)
    oiii = 0.2 * (1.0 + sfr_peak)

    # AGN boost primarily to [OIII]
    if agn_frac > 0:
        oiii = oiii * (1.0 + 3.0 * agn_frac)
        nii = nii * (1.0 + 1.5 * agn_frac)

    if ha > 1e-12 and hb > 1e-12:
        log_nii_ha.append(np.log10(max(nii / ha, 1e-3)))
        log_oiii_hb.append(np.log10(max(oiii / hb, 1e-3)))

log_nii_ha = np.array(log_nii_ha)
log_oiii_hb = np.array(log_oiii_hb)

# Plot BPT diagram
fig, ax = plt.subplots(figsize=(8, 7))

# Kewley+2001 maximum starburst line
log_nii_grid = np.linspace(-1.6, 0.5, 300)
log_oiii_kewley = 0.61 / (log_nii_grid - 0.47) + 1.19

# Kauffmann+2003 empirical SF line
log_oiii_kauff = 0.61 / (log_nii_grid - 0.05) + 1.3

mask_k = log_nii_grid < 0.47
ax.plot(
    log_nii_grid[mask_k],
    log_oiii_kewley[mask_k],
    "k-",
    lw=2.0,
    label="Kewley+2001 (max starburst)",
)
mask_kauff = log_nii_grid < 0.05
ax.plot(
    log_nii_grid[mask_kauff],
    log_oiii_kauff[mask_kauff],
    "k--",
    lw=1.8,
    label="Kauffmann+2003 (empirical SF)",
)

# Region labels
ax.text(-1.35, -0.6, "Star\nForming", fontsize=11, color="#1f77b4", ha="center")
ax.text(-0.1, 0.7, "Composite", fontsize=11, color="#ff7f0e", ha="center")
ax.text(0.25, 1.15, "Seyfert/LINER", fontsize=11, color="#d62728", ha="center")

sc = ax.scatter(
    log_nii_ha,
    log_oiii_hb,
    c=agn_fracs[: len(log_nii_ha)],
    cmap="viridis",
    s=80,
    zorder=5,
    edgecolors="k",
    lw=0.5,
)
cbar = fig.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label("AGN fraction")

ax.set_xlabel(r"log [NII]$\lambda$6583 / H$\alpha$")
ax.set_ylabel(r"log [OIII]$\lambda$5007 / H$\beta$")
ax.set_xlim(-1.6, 0.6)
ax.set_ylim(-1.2, 1.5)
ax.legend(frameon=False, loc="lower left")

fig.tight_layout()
fig.savefig("plot_workflow_bpt_classification.png", dpi=150, bbox_inches="tight")
