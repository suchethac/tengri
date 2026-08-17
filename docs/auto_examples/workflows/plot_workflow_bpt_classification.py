"""
BPT diagram: emission lines from the baked-in nebular SSP
=========================================================

BPT ([OIII]/Hβ vs [NII]/Hα) line ratios computed directly from the
rest-frame SED via continuum-subtracted boxcar integration around each
line center, swept across a stellar metallicity grid. The Kewley+2001
and Kauffmann+2003 demarcation lines distinguish star-forming galaxies
from AGN.

References: Kewley+2001; Kauffmann+2003.
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

LINES = {
    "halpha": 6564.7,
    "hbeta": 4862.7,
    "oiii_5007": 5008.3,
    "nii_6584": 6585.4,
}
LINE_HALF_WIDTH = 8.0
CONT_OFFSET = 30.0
CONT_HALF_WIDTH = 8.0


def boxcar_line_flux(wave, sed, line_center):
    """Continuum-subtracted boxcar flux around a single line."""
    line_mask = np.abs(wave - line_center) < LINE_HALF_WIDTH
    blue_mask = np.abs(wave - (line_center - CONT_OFFSET)) < CONT_HALF_WIDTH
    red_mask = np.abs(wave - (line_center + CONT_OFFSET)) < CONT_HALF_WIDTH
    cont = 0.5 * (sed[blue_mask].mean() + sed[red_mask].mean())
    return float(np.trapezoid(sed[line_mask] - cont, wave[line_mask]))


ssp = tengri.load_ssp()
logzsol_grid = np.linspace(-1.0, 0.2, 15)

log_n2_ha = []
log_o3_hb = []

for logz in logzsol_grid:
    model = tengri.SEDModel.build(
        ssp,
        sfh={
            "type": "tsnorm",
            "all_params": tengri.FIXED,
            "log_total_mass": 10.0,
            "peak_lbt_gyr": 2.0,
            "width_gyr": 1.0,
            "skew": 0.2,
            "trunc": 3.0,
            "logzsol": float(logz),
        },
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_bc": 0.1,
            "tau_diff": 0.1,
            "slope": -0.7,
        },
    )

    pred = model.predict({"redshift": 0.1})
    wave = np.asarray(model.wavelengths)
    sed = np.asarray(pred.rest_sed())
    fluxes = {name: boxcar_line_flux(wave, sed, lam) for name, lam in LINES.items()}
    log_n2_ha.append(np.log10(max(fluxes["nii_6584"] / fluxes["halpha"], 1e-3)))
    log_o3_hb.append(np.log10(max(fluxes["oiii_5007"] / fluxes["hbeta"], 1e-3)))

log_n2_ha = np.array(log_n2_ha)
log_o3_hb = np.array(log_o3_hb)

fig, ax = plt.subplots(figsize=(8, 7))

log_nii_grid = np.linspace(-1.6, 0.5, 300)
log_oiii_kewley = 0.61 / (log_nii_grid - 0.47) + 1.19
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

ax.text(-1.35, -0.6, "Star\nForming", fontsize=11, color="#1f77b4", ha="center")
ax.text(-0.1, 0.7, "Composite", fontsize=11, color="#ff7f0e", ha="center")
ax.text(0.25, 1.15, "Seyfert/LINER", fontsize=11, color="#d62728", ha="center")

sc = ax.scatter(
    log_n2_ha,
    log_o3_hb,
    c=logzsol_grid,
    cmap="viridis",
    s=80,
    zorder=5,
    edgecolors="k",
    lw=0.5,
)
cbar = fig.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label(r"log $Z_\star / Z_\odot$")

ax.set_xlabel(r"log [NII]$\lambda$6584 / H$\alpha$")
ax.set_ylabel(r"log [OIII]$\lambda$5007 / H$\beta$")
ax.set_xlim(-1.6, 0.6)
ax.set_ylim(-1.2, 1.5)

fig.tight_layout()
plt.savefig("plot_workflow_bpt_classification.png", dpi=150, bbox_inches="tight")
