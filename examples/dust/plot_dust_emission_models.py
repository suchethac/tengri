"""
Dust Emission Models: Overview
================================

All tengri dust emission models evaluated at L_absorbed = 1e10 L☉ and
T = 35 K. Template-based models (DL07, DL14, Dale+2014) use their
analytic fallbacks when data files are unavailable.
"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message=".*template.*not found.*")
warnings.filterwarnings("ignore", message=".*DL07.*")
warnings.filterwarnings("ignore", message=".*Failed to load.*")

from tengri import setup_style
from tengri.models.dust.emission import (
    _dale2014_analytic_fallback,
    _draine_li2007_analytic_fallback,
    _draine_li2014_analytic_fallback,
    casey2012,
    energy_balance_split,
    magphys_dc08,
    modified_blackbody,
)

setup_style()

# --- Wavelength grid: 1–1000 μm ---
wave_aa = jnp.logspace(np.log10(1e4), np.log10(1e7), 2000)
wave_um = np.array(wave_aa) * 1e-4
L_ABS = 1e10  # Lsun

def _mbb():
    return modified_blackbody(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8)

def _casey():
    return casey2012(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0)

def _magphys():
    return magphys_dc08(wave_aa, L_ABS, dust_T_warm=35.0, dust_T_cold=20.0,
                        dust_T_hot=180.0, dust_xi_pah=0.06,
                        dust_xi_mir=0.07, dust_xi_warm=0.25)

def _ebs():
    return energy_balance_split(wave_aa, L_ABS, dust_T_warm=35.0, dust_T_cold=20.0)

def _dl07():
    return _draine_li2007_analytic_fallback(
        wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5
    )

def _dl14():
    return _draine_li2014_analytic_fallback(
        wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5
    )

def _dale():
    return _dale2014_analytic_fallback(wave_aa, L_ABS, dust_alpha_dale=2.0)

MODELS = [
    ("Modified BB",          "C0", _mbb()),
    ("Casey (2012)",         "C1", _casey()),
    ("MAGPHYS (dC+08)",      "C2", _magphys()),
    ("Energy balance",       "C3", _ebs()),
    ("DL07 (fallback)",      "C4", _dl07()),
    ("DL14 (fallback)",      "C5", _dl14()),
    ("Dale+2014 (fallback)", "C6", _dale()),
]

fig, ax = plt.subplots(figsize=(9, 5.5))
for label, color, lnu in MODELS:
    y = np.array(lnu)
    mask = (wave_um > 1) & (y > 0)
    ax.loglog(wave_um[mask], y[mask], color=color, lw=1.8, label=label)

ax.set_xlim(1, 1000)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
ax.set_title(r"Dust Emission Models ($L_{\rm abs} = 10^{10}\,L_\odot$, $T = 35$ K)")
ax.legend(fontsize=8, frameon=False, ncol=2)

# Mark key wavelengths
for wl_um, name in [(8, "PAH"), (25, "mid-IR"), (100, "far-IR peak"), (850, "submm")]:
    ax.axvline(wl_um, color="grey", ls=":", lw=0.5, alpha=0.5)
    ax.text(wl_um * 1.05, ax.get_ylim()[1] * 0.6, name, fontsize=7, color="grey", rotation=90)

fig.tight_layout()
plt.savefig("plot_dust_emission_models.png", dpi=150, bbox_inches="tight")
plt.show()
