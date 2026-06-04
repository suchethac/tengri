"""
GRAHSP Balmer continuum: building the small blue bump
======================================================

The GRAHSP AGN model (Buchner+ 2024) optionally adds a **Balmer continuum**
following Grandi (1982): a 15,000 K blackbody truncated at the Balmer edge
(3646 Å) and Gaussian-broadened by the line width. Together with the FeII
forest it builds the "small blue bump" seen blueward of ~4000 Å in type-1
quasars.

Here we sweep the strength parameter ``agn_grahsp_a_bc`` (upstream ``ABC``,
the Balmer-continuum strength relative to the power-law at 3000 nm) from off
to strong, holding everything else fixed. Watch the step at the Balmer edge
and the way the Gaussian convolution smears it into the continuum.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.agn.grahsp.model import compute_grahsp_sed

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Rest-frame optical/UV window where the Balmer continuum lives.
wave_aa = jnp.logspace(np.log10(1500.0), np.log10(8000.0), 1200)
wave_um = np.asarray(wave_aa) / 1e4

A_BC = [0.0, 0.5, 1.0, 2.0]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(A_BC)))

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for a_bc, c in zip(A_BC, colors):
    # Disable FeII so the Balmer continuum is visible in isolation.
    lnu = np.asarray(
        compute_grahsp_sed(
            wave_aa,
            agn_log_lbol=45.0,
            agn_grahsp_a_feii=0.0,
            agn_grahsp_a_bc=a_bc,
            agn_grahsp_linewidth_kms=5000.0,
        )
    )
    ax.plot(wave_um, wave_um * lnu, color=c, lw=1.8, label=rf"$A_{{\rm BC}}={a_bc}$")

ax.axvline(3646.0 / 1e4, color="0.5", ls=":", lw=1.0)
ax.text(3646.0 / 1e4, ax.get_ylim()[1] * 0.92, " Balmer edge\n 3646 Å", fontsize=8, color="0.4")
ax.set_xlabel(r"rest wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_\lambda$ [erg s$^{-1}$, arb. norm.]")
ax.set_title("GRAHSP Balmer continuum (Grandi 1982) strength sweep")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
