"""
delta_alpha_OX pivots the X-ray spectrum about the disc UV anchor
==================================================================

Reproduces the UV-to-X-ray connection panel from Yang et al. 2020
(X-CIGALE Fig. 3): the X-ray corona is normalised through the Just+07
alpha_OX-L_2500 relation, anchored at the disc-derived L_2500. Offsets
``delta_alpha_OX`` from -0.3 to +0.3 dex pivot the X-ray power-law
about the 2500 A anchor — the disc UV stays fixed (single curve at
log lam > 1), only the X-ray normalisation moves.

The "observable X-ray" annotation marks the practical lower-energy
limit (around 0.5 keV / 25 A) below which interstellar absorption
makes the soft band difficult to measure.

Reference: Yang et al. 2020, MNRAS 491, 740 (X-CIGALE Fig. 3);
Just et al. 2007, ApJ 665, 1004.
"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.agn import multicolor_disc
from tengri.xray import xray_agn_corona

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

LOG_LBOL = 12.0
# Hopkins+2007 BC=5.15 at 2500 A
_BC_NU = 5.15 * 1.199e15
L_BOL_ERG = 10.0**LOG_LBOL * 3.828e33
L_2500 = L_BOL_ERG / _BC_NU

DELTA_VALUES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
COLORS = ("#d62728", "#e377c2", "#ff7f0e", "#2ca02c", "#17becf", "#1f77b4", "#9467bd")

# Disc grid (10 nm to 1000 nm rest-frame, in Angstrom)
disc_wave = jnp.logspace(np.log10(100.0), np.log10(1.0e4), 600)
log_lam_nm_disc = np.log10(np.asarray(disc_wave) / 10.0)

l_disc = multicolor_disc(
    disc_wave,
    agn_log_lbol=LOG_LBOL,
    agn_log_mbh=8.5,
    agn_log_ledd=-1.0,
    agn_a_spin=0.5,
    agn_cos_inc=0.95,
)
l_disc = np.asarray(l_disc)

# X-ray grid (0.04 to 10 nm, i.e. ~0.1 - 300 keV in Angstrom)
xray_wave = jnp.logspace(np.log10(0.4), np.log10(100.0), 400)
log_lam_nm_xray = np.log10(np.asarray(xray_wave) / 10.0)
wave_keV_xray = 12.398 / np.asarray(xray_wave)

fig, ax = plt.subplots(figsize=(7.5, 5.0))

# Plot disc UV/optical once (it doesn't depend on delta_alpha_ox)
ax.plot(log_lam_nm_disc, np.log10(l_disc), color="#7f7f7f", lw=2.0, label="_nolegend_")

for delta, color in zip(DELTA_VALUES, COLORS):
    l_xray = np.asarray(
        xray_agn_corona(
            xray_wave,
            l_2500_30deg_erg_hz=L_2500,
            delta_alpha_ox=float(delta),
        )
    )
    mask = l_xray > 0
    ax.plot(
        log_lam_nm_xray[mask],
        np.log10(l_xray[mask]),
        color=color,
        lw=1.6,
        label=rf"${delta:+.1f}$",
    )

ax.set(
    xlim=(-1.0, 3.0),
    ylim=(24.5, 30.5),
    xlabel=r"$\log\lambda$  [nm, rest-frame]",
    ylabel=r"$\log L_\nu$  (cgs)",
)
ax.legend(title=r"$\Delta\alpha_{\rm OX}$", loc="lower right", frameon=False, ncol=2, fontsize=9)


# Twin x-axis: log E (keV) on top
def _lam_nm_to_log_E_kev(log_lam_nm: float) -> float:
    lam_aa = 10.0 ** (log_lam_nm + 1.0)  # nm to Å
    return np.log10(12.398 / lam_aa)


secax = ax.secondary_xaxis(
    "top",
    functions=(_lam_nm_to_log_E_kev, lambda x: np.log10(12.398 / 10.0 ** (x + 1.0))),
)
secax.set_xlabel(r"$\log E$  [keV, rest-frame]")

# "observable X-ray" annotation around 0.5 keV (= 25 Å = log lam_nm ~ 0.4)
ax.axvline(0.4, color="0.2", lw=0.7)
ax.annotate(
    "",
    xy=(-0.5, 28.0),
    xytext=(0.35, 28.0),
    arrowprops=dict(arrowstyle="->", color="0.2", lw=1.0),
)
ax.text(0.05, 28.15, "observable X-ray", fontsize=9, color="0.2")

fig.tight_layout()
plt.savefig("plot_alpha_ox_uv_xray_connection.png", dpi=150, bbox_inches="tight")
