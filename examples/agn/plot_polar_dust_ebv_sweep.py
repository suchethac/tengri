"""
Polar-dust E(B-V) sweep for Type 1 and Type 2 AGN sightlines (X-CIGALE)
========================================================================

Reproduces Figure 1 of Yang et al. 2020 (the X-CIGALE polar-dust
introduction): SMC-law attenuation of the AGN disc by dust above the
torus, plus an energy-conserving mid-IR greybody re-emission. Two
panels at cos_inc = 0.95 (Type-1, face-on into the polar cone) and
cos_inc = 0.10 (Type-2, edge-on view of the torus) for opening angle
40°. We sweep ``agn_polar_ebv`` from 0.00 to 0.30 — covering the
empirical range Yang+2020 anchor against red quasars.

The contrast between the panels is the key physics: for Type-1
sightlines, polar dust sits directly in the line of sight and the
UV/optical disc dims sharply with E(B-V); for Type-2 sightlines the
disc is already obscured by the torus and polar dust contributes
mostly to mid-IR re-emission, leaving the observed shape nearly
unchanged.

Reference: Yang et al. 2020, MNRAS 491, 740 (X-CIGALE), Fig. 1.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.agn import multicolor_disc
from tengri.components.agn.polar_dust import polar_dust_total

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

LOG_LBOL = 12.0
EBV_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
EBV_COLORS = ["#8856a7", "#1f77b4", "#2ca02c", "#ff7f0e", "#e377c2", "#d62728"]
OPENING_ANGLE_DEG = 40.0

# Wide rest-frame grid: 100 Å (FUV) to 1000 µm (FIR)
wavelength_aa = jnp.logspace(2.0, 7.0, 1200)
log_lam_nm = np.log10(np.asarray(wavelength_aa) / 10.0)


def predict_with_polar(cos_inc: float, ebv: float) -> np.ndarray:
    """Multicolour disc + polar-dust attenuation + greybody re-emission."""
    l_disc = multicolor_disc(
        wavelength_aa,
        agn_log_lbol=LOG_LBOL,
        agn_log_mbh=8.5,
        agn_log_ledd=-1.0,
        agn_a_spin=0.5,
        agn_cos_inc=cos_inc,
    )
    l_atten, l_reemit = polar_dust_total(
        l_disc,
        wavelength_aa,
        cos_inc=cos_inc,
        opening_angle_deg=OPENING_ANGLE_DEG,
        ebv=float(ebv),
    )
    return np.asarray(l_atten + l_reemit)


# Normalise each curve by its maximum (X-CIGALE style — "normed")
def _norm(l_nu: np.ndarray) -> np.ndarray:
    return l_nu / np.max(l_nu)


fig, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

for ebv, color in zip(EBV_VALUES, EBV_COLORS):
    l_t1 = predict_with_polar(cos_inc=0.95, ebv=ebv)
    l_t2 = predict_with_polar(cos_inc=0.10, ebv=ebv)
    ax_t1.plot(log_lam_nm, np.log10(_norm(l_t1)), color=color, lw=1.6, label=f"{ebv:.2f}")
    ax_t2.plot(log_lam_nm, np.log10(_norm(l_t2)), color=color, lw=1.6, label=f"{ebv:.2f}")

for ax, label in [(ax_t1, "Type 1"), (ax_t2, "Type 2")]:
    ax.set(
        xlim=(2.0, 6.0),
        ylim=(-4.0, 1.2),
        xlabel=r"$\log\lambda$  [nm, rest-frame]",
    )
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(title=r"$E(B-V)$", loc="lower right", frameon=False, ncol=2, fontsize=8)

ax_t1.set_ylabel(r"$\log L_\nu$  (normed)")
fig.tight_layout()
plt.savefig("plot_polar_dust_ebv_sweep.png", dpi=150, bbox_inches="tight")
