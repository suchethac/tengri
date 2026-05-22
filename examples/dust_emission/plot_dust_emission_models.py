"""
Dust Emission Models: Overview
==============================

Compares all dust emission models available in tengri at a fixed infrared
luminosity and fiducial temperature. Template-based models gracefully skip
if data files are unavailable.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dust_emission_models_001.png
   :alt: plot_dust_emission_models
   :class: sphx-glr-single-img

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.dust import (
    casey2012,
    dale2014,
    draine_li2007,
    draine_li2014,
    energy_balance_split,
    modified_blackbody,
)
from tengri.plot import setup_style

setup_style()

wave_aa = jnp.logspace(np.log10(1e4), np.log10(1e7), 2000)
wave_um = np.array(wave_aa) * 1e-4
L_ABS = 1e10 * 3.828e33


def _mbb():
    return modified_blackbody(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8)


def _casey():
    return casey2012(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0)


def _ebs():
    return energy_balance_split(wave_aa, L_ABS, dust_T_warm=35.0, dust_T_cold=20.0)


def _dl07():
    try:
        return draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
    except FileNotFoundError:
        return None


def _dl14():
    try:
        return draine_li2014(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
    except FileNotFoundError:
        return None


def _dale():
    try:
        return dale2014(wave_aa, L_ABS, dust_alpha_dale=2.0)
    except FileNotFoundError:
        return None


MODELS = [
    ("Modified BB", "C0", _mbb()),
    ("Casey (2012)", "C1", _casey()),
    ("Energy balance", "C3", _ebs()),
    ("DL07", "C4", _dl07()),
    ("DL14", "C5", _dl14()),
    ("Dale+2014", "C6", _dale()),
]

fig, ax = plt.subplots(figsize=(10, 6))
for label, color, lnu in MODELS:
    if lnu is None:
        continue
    y = np.array(lnu)
    mask = (wave_um > 1) & (y > 0)
    ax.loglog(wave_um[mask], y[mask], color=color, lw=2.0, label=label)

ax.set(
    xlim=(1, 1000),
    ylim=(1e27, 5e31),
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]",
    title=r"Dust Emission Models ($L_{\rm abs} = 10^{10}\,L_\odot$, $T = 35$ K)",
)
ax.legend(fontsize=10, frameon=False, ncol=2)

for wl_um, name in [(8, "PAH"), (25, "mid-IR"), (100, "far-IR peak"), (850, "submm")]:
    ax.axvline(wl_um, color="grey", ls=":", lw=0.5, alpha=0.5)
    ax.text(
        wl_um * 1.05,
        0.97,
        name,
        fontsize=10,
        color="grey",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        va="top",
    )

fig.tight_layout()
plt.savefig("plot_dust_emission_models.png", dpi=150, bbox_inches="tight")
plt.show()
