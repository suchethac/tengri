"""
GRAHSP FeII forest: Bruhweiler+Verner 2008 vs Veron-Cetty 2004
===============================================================

The iron pseudo-continuum (the "FeII forest") is a defining feature of type-1
AGN optical/UV spectra. GRAHSP offers two templates: the photoionisation
model of **Bruhweiler & Verner (2008)** (the upstream default) and the
empirical **Veron-Cetty, Joly & Veron (2004)** template. They differ most in
the relative strength and shape of the UV (2200-3000 Å) and optical
(4400-5400 Å) multiplet blends.

To make the comparison clear we plot the FeII forest **in isolation** (the
``feii`` component returned by :func:`evaluate_grahsp_agn`, not buried under
the accretion-disc continuum), both scaled to the same ``agn_grahsp_a_feii``.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave_aa = jnp.linspace(1800.0, 7500.0, 2400)
wave_um = np.asarray(wave_aa) / 1e4
templates = load_grahsp_templates()


def feii_isolated(template):
    sed = evaluate_grahsp_agn(
        wave_aa * 0.1,  # nm
        GRAHSPParams(l5100=1e44, a_feii=10.0, feii_template=template),
        templates,
    )
    return np.asarray(sed.feii)  # erg/s/nm


fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(wave_um, wave_um * feii_isolated("bruhweiler2008"), lw=1.7, label="Bruhweiler & Verner 08")
ax.plot(wave_um, wave_um * feii_isolated("veroncetty2004"), lw=1.7, label="Veron-Cetty+ 2004")

for lo, hi, lab in [(0.22, 0.30, "UV FeII"), (0.44, 0.54, "optical FeII")]:
    ax.axvspan(lo, hi, color="0.88", alpha=0.6, zorder=0)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.93, lab, ha="center", fontsize=8, color="0.4")

ax.set_xlabel(r"rest wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_{\lambda,\,\rm FeII}$ [erg s$^{-1}$, arb. norm.]")
ax.set_title(r"GRAHSP FeII forest templates, isolated ($A_{\rm FeII}=10$)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
