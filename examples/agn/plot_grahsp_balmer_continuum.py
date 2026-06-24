"""
GRAHSP Balmer continuum: building the small blue bump
======================================================

The GRAHSP AGN model (Buchner+ 2024) optionally adds a **Balmer continuum**
following Grandi (1982): a 15,000 K blackbody truncated at the Balmer edge
(3646 Å) and Gaussian-broadened by the line width. Together with the FeII
forest it builds the "small blue bump" seen blueward of ~4000 Å in type-1
quasars.

To isolate the effect we switch the emission lines, FeII forest and torus
**off** (``a_lines=0``, ``a_feii=0``, ``fcov=0``), leaving only the bending
power-law continuum, and sweep the strength parameter ``agn_grahsp_a_bc``
(upstream ``ABC``) from off to strong. The left panel shows the continuum +
Balmer bump zoomed on the edge; the right shows the Balmer contribution in
isolation (zero above the edge, rising blueward).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.agn.grahsp.model import (
    GRAHSPParams,
    compute_grahsp_sed,
    evaluate_grahsp_agn,
)
from tengri.components.agn.grahsp.templates import load_grahsp_templates

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Rest-frame near-UV/optical window bracketing the Balmer edge.
wave_aa = jnp.linspace(1500.0, 5000.0, 1400)
wave_um = np.asarray(wave_aa) / 1e4
templates = load_grahsp_templates()

A_BC = [0.0, 0.5, 1.0, 2.0]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(A_BC)))

# Precompute both panels so each can be normalized to O(1) (these are in
# arbitrary units; only the relative strengths are physical).
totals, balmers = [], []
for a_bc in A_BC:
    # Continuum + Balmer only (lines / FeII / torus suppressed).
    total = np.asarray(
        compute_grahsp_sed(
            wave_aa,
            agn_log_lbol=45.0,
            agn_grahsp_a_lines=0.0,
            agn_grahsp_a_feii=0.0,
            agn_grahsp_fcov=0.0,
            agn_grahsp_a_bc=a_bc,
        )
    )
    totals.append(wave_um * total)
    # Balmer contribution in isolation (erg/s/nm -> lambda*L_lambda).
    sed = evaluate_grahsp_agn(
        wave_aa * 0.1,  # nm
        GRAHSPParams(l5100=1e44, a_lines=0.0, a_feii=0.0, fcov=0.0, a_bc=a_bc),
        templates,
    )
    balmers.append(wave_um * np.asarray(sed.balmer))

# Panel 1 by the pure-continuum baseline (A_BC=0); panel 2 by the strongest
# isolated Balmer curve so both axes read O(1).
norm1 = totals[0].max()
norm2 = max(b.max() for b in balmers)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4), sharex=True)

for a_bc, c, total, balmer in zip(A_BC, colors, totals, balmers):
    ax1.plot(wave_um, total / norm1, color=c, lw=1.8, label=rf"$A_{{\rm BC}}={a_bc}$")
    ax2.plot(wave_um, balmer / norm2, color=c, lw=1.8)

for ax in (ax1, ax2):
    ax.axvline(3646.0 / 1e4, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel(r"rest wavelength [$\mu$m]")
ax1.text(3646.0 / 1e4, ax1.get_ylim()[1] * 0.9, " Balmer\n edge", fontsize=8, color="0.4")
ax1.set_ylabel(r"$\lambda L_\lambda$ [normalized]")
ax1.set_title("Bending power-law + Balmer continuum")
ax1.legend(frameon=False, fontsize=9)
ax2.set_title("Balmer continuum contribution (isolated)")
fig.suptitle("GRAHSP Balmer continuum (Grandi 1982) strength sweep", y=1.02)
fig.tight_layout()
plt.show()
