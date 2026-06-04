"""
GRAHSP FeII forest: Bruhweiler+Verner 2008 vs Veron-Cetty 2004
===============================================================

The iron pseudo-continuum (the "FeII forest") is a defining feature of type-1
AGN optical/UV spectra. GRAHSP offers two templates: the photoionisation
model of **Bruhweiler & Verner (2008)** (the upstream default) and the
empirical **Veron-Cetty, Joly & Veron (2004)** template. They differ most in
the relative strength of the UV (2200-3000 Å) and optical (4400-5400 Å)
multiplet blends.

Here both are scaled to the same ``agn_grahsp_a_feii`` and overlaid on the
bending power-law continuum, with the other emission lines suppressed so the
iron blends are clear.
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

wave_aa = jnp.logspace(np.log10(1800.0), np.log10(7500.0), 1500)
wave_um = np.asarray(wave_aa) / 1e4


def feii_sed(template):
    return np.asarray(
        compute_grahsp_sed(
            wave_aa,
            agn_log_lbol=45.0,
            agn_grahsp_a_lines=0.0,  # suppress the Gaussian emission lines
            agn_grahsp_a_feii=10.0,  # strong iron so the blends are visible
            feii_template=template,
        )
    )


fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(wave_um, wave_um * feii_sed("bruhweiler2008"), lw=1.8, label="Bruhweiler & Verner 2008")
ax.plot(wave_um, wave_um * feii_sed("veroncetty2004"), lw=1.8, label="Veron-Cetty+ 2004")

# Mark the classic UV and optical FeII blend regions.
for lo, hi, lab in [(0.22, 0.30, "UV FeII"), (0.44, 0.54, "optical FeII")]:
    ax.axvspan(lo, hi, color="0.85", alpha=0.5, zorder=0)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95, lab, ha="center", fontsize=8, color="0.4")

ax.set_xlabel(r"rest wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_\lambda$ [erg s$^{-1}$, arb. norm.]")
ax.set_title("GRAHSP FeII forest templates ($A_{\\rm FeII}=10$)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
