"""
Richards+2006 empirical Type-1 quasar template alongside physical discs
========================================================================

Three accretion-disc backbones at the same bolometric anchor
(log L_bol / L_sun = 12.5): the Richards et al. 2006 empirical mean
Type-1 SDSS quasar template, the Temple, Hewett & Banerji 2021
empirical QSOgen, and the Shakura-Sunyaev multicolour disc (the
outer-disc component of Kubota & Done 2018). Each is normalised to the
same bolometric output so the differences are entirely in spectral
*shape* — Richards+2006 is broader than QSOgen and carries the
infrared bump from its host-galaxy-corrected composite, while the
multicolour disc cuts off sharply on either side of the big blue bump.

Reference: Richards et al. 2006, ApJS 166, 470 (composite SDSS Type-1
template); Temple, Hewett & Banerji 2021, MNRAS 508, 737 (QSOgen);
Kubota & Done 2018, MNRAS 480, 1247 (multicolour disc).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.agn import multicolor_disc, qsogen, richards2006

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
LOG_LBOL = 12.5
wavelength = jnp.logspace(jnp.log10(10.0), jnp.log10(1.0e6), 2000)
wave_aa = np.asarray(wavelength)
nu_factor = C_AA_PER_S / wave_aa

l_richards = richards2006(wavelength, agn_log_lbol=LOG_LBOL)
l_qsogen = qsogen(wavelength, agn_log_lbol=LOG_LBOL)
l_multicolor = multicolor_disc(
    wavelength,
    agn_log_lbol=LOG_LBOL,
    agn_log_mbh=9.0,
    agn_log_ledd=-1.0,
    agn_a_spin=0.5,
    agn_cos_inc=0.5,
)
MODELS = [
    ("Richards+2006 (empirical SDSS Type-1)", l_richards, "C0"),
    ("QSOgen (Temple+2021 empirical)", l_qsogen, "C3"),
    ("multicolour disc (K&D 2018 physical)", l_multicolor, "C2"),
]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for label, l_nu_jax, color in MODELS:
    l_nu = np.asarray(l_nu_jax)
    mask = l_nu > 0
    ax.loglog(wave_aa[mask] * 1.0e-4, (nu_factor * l_nu)[mask], color=color, lw=1.6, label=label)

ax.set(
    xlim=(0.001, 100.0),
    ylim=(1.0e42, 5.0e46),
    xlabel=r"Rest-frame wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.axvspan(1.0e-3, 0.012, alpha=0.08, color="0.5")
ax.axvspan(0.12, 0.4, alpha=0.08, color="C0")
ax.text(2.0e-3, 1.5e42, "X-ray", fontsize=8, color="0.4", ha="center")
ax.text(0.2, 1.5e42, "UV BBB", fontsize=8, color="C0", ha="center")
ax.legend(loc="lower center", fontsize=9, frameon=False)
fig.tight_layout()
plt.savefig("plot_richards2006_template.png", dpi=150, bbox_inches="tight")
