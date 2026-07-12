"""
BPT diagram: line ratios from the property catalog
==================================================

BPT ([OIII]/Hβ vs [NII]/Hα) line ratios swept across a stellar
metallicity grid, read straight off the property catalog. The
Kewley+2001 and Kauffmann+2003 demarcation lines are overlaid.

**Ask the model for a line; do not measure it off the continuum grid.**
An earlier version of this example integrated a continuum-subtracted
boxcar (half-width 8 Å) around each line centre on the SSP wavelength
grid. That grid is log-spaced and coarse — 64 Å per pixel at Hα — so an
8 Å box contains at most *one* sample, and ``np.trapezoid`` over one
point is exactly ``0.0``. The example divided by that zero and crashed.
It had been broken in the published gallery for a long time, because CI
never executed the gallery (#1146).

The lines are a *derived property*, not something to re-measure: build
with a photoionization backend (Cue, on a bare-stellar SSP) and read
``pred.halpha`` / ``pred.nii_6584`` / … from the catalog. Those are
per-line luminosities the backend actually solved for, at the correct
resolution.

Reference: Kewley et al. 2001, ApJ, 556, 121 (theoretical classification);
Kauffmann et al. 2003, MNRAS, 346, 1055 (empirical SF boundary).
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

# Cue is a photoionization backend: it solves for per-line luminosities and
# publishes them to the property catalog. It needs a BARE-STELLAR SSP — pairing
# it with a wNE grid (nebular already baked into the templates) would
# double-count the nebular emission, and tengri refuses.
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
logzsol_grid = np.linspace(-1.0, 0.2, 15)

log_n2_ha = []
log_o3_hb = []

for logz in logzsol_grid:
    model = tengri.SEDModel.build(
        ssp,
        sfh={
            "type": "tsnorm",
            "*": tengri.FIXED,
            "log_total_mass": 10.0,
            "peak_lbt_gyr": 2.0,
            "width_gyr": 1.0,
            "skew": 0.2,
            "trunc": 3.0,
            "logzsol": float(logz),
        },
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_bc": 0.1,
            "tau_diff": 0.1,
            "slope": -0.7,
        },
        neb={"type": "cue", "*": tengri.FIXED, "logZ_gas": float(logz)},
        redshift=tengri.Fixed(0.1),
    )

    # The lines come off the catalog by name — no continuum subtraction, no
    # boxcar, no dependence on the SSP grid spacing.
    pred = model.predict(model.spec.get_fixed_values())
    log_n2_ha.append(np.log10(float(pred.nii_6584) / float(pred.halpha)))
    log_o3_hb.append(np.log10(float(pred.oiii_5007) / float(pred.hbeta)))

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
