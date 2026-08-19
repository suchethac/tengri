"""
N_H column density sweep: from unobscured to Compton-thick
==========================================================

The line-of-sight column density ``N_H`` reshapes the AGN X-ray spectrum
in two regimes: photoelectric absorption (``zphabs``) suppresses the soft
band roughly as :math:`\\exp(-\\sigma(E)\\,N_H)` with cross-section
:math:`\\sigma \\propto E^{-3}`, while Compton down-scattering (``cabs``)
adds an energy-independent suppression :math:`\\exp(-\\sigma_T\\,N_H)`
that becomes dominant once ``log N_H ≳ 24`` (the Compton-thick boundary).
A constant warm-electron scattered fraction (~1 % of the intrinsic
continuum) is added back, which is the only flux observable in the soft
band for nearly opaque columns and explains why Compton-thick AGN are
still marginally detectable in soft-band stacks
(Matsumoto et al. 2026 Fig. 11/12).

This sweep reproduces the spectral evolution behind Matsumoto+2026
Figure 11 (hardness-ratio model tracks) for log N_H from 20 to 25.

References
----------

- Ricci et al. 2017, Nature 549, 488 (zphabs × cabs × cut-off PL +
  scattered component spectral model).

- Matsumoto et al. 2026, Eq. B6 and §3.3 (X-ray properties of MIPS-
  selected obscured AGN at z > 3).

- Morrison & McCammon 1983, ApJ 270, 119 (photoelectric cross-sections).

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.plot import setup_style
from tengri.xray import xray_agn_corona

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# ---------------------------------------------------------------------- grid
wavelength = jnp.logspace(np.log10(0.124), np.log10(124.0), 512)  # 0.1 – 100 keV
wave_keV = 12.398 / np.array(wavelength)

L_BOL = 1e45  # erg/s — luminous AGN
# Hopkins+2007 BC=5.15 at 2500 A
L_2500 = L_BOL / (5.15 * 1.199e15)
log_nh_vals = np.array([20.0, 22.0, 23.0, 24.0, 24.5, 25.0])

# ---------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
cmap = plt.get_cmap("plasma")
norm = mpl.colors.Normalize(vmin=20.0, vmax=25.0)

# Panel a: spectral evolution with N_H
ax = axes[0]
l_intr = np.array(xray_agn_corona(wavelength, l_2500_30deg_erg_hz=L_2500, log_nh=15.0))
ax.loglog(wave_keV, l_intr, color="0.5", ls="--", lw=1.3, label="intrinsic")
for log_nh in log_nh_vals:
    l_obs = np.array(xray_agn_corona(wavelength, l_2500_30deg_erg_hz=L_2500, log_nh=float(log_nh)))
    ax.loglog(
        wave_keV,
        l_obs,
        lw=1.5,
        color=cmap(norm(log_nh)),
        label=rf"$\log N_H = {log_nh:.1f}$",
    )

ax.set_xlim(0.2, 100)
ax.set_ylim(1e21, 5e25)
ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.axvspan(0.5, 2.0, alpha=0.10, color="C0")
ax.axvspan(2.0, 10.0, alpha=0.10, color="C2")
ax.text(0.7, 1.5e21, "soft\n(0.5–2 keV)", fontsize=7, color="C0", ha="center")
ax.text(4.0, 1.5e21, "hard\n(2–10 keV)", fontsize=7, color="C2", ha="center")
ax.text(
    0.04,
    0.96,
    rf"$L_{{\rm bol}} = 10^{{{np.log10(L_BOL):.0f}}}$ erg/s",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)
ax.legend(fontsize=8, frameon=False, loc="lower right")

# Panel b: 2–10 keV vs 0.5–2 keV band ratios as a function of N_H
ax = axes[1]
log_nh_grid = np.linspace(20.0, 26.0, 60)
soft_mask = (wave_keV >= 0.5) & (wave_keV <= 2.0)
hard_mask = (wave_keV >= 2.0) & (wave_keV <= 10.0)

soft_frac, hard_frac = [], []
for log_nh in log_nh_grid:
    l_obs = np.array(xray_agn_corona(wavelength, l_2500_30deg_erg_hz=L_2500, log_nh=float(log_nh)))
    soft_frac.append(
        np.trapezoid(l_obs[soft_mask], wave_keV[soft_mask])
        / np.trapezoid(l_intr[soft_mask], wave_keV[soft_mask])
    )
    hard_frac.append(
        np.trapezoid(l_obs[hard_mask], wave_keV[hard_mask])
        / np.trapezoid(l_intr[hard_mask], wave_keV[hard_mask])
    )

ax.semilogy(log_nh_grid, soft_frac, color="C0", lw=2.0, label="0.5–2 keV (soft)")
ax.semilogy(log_nh_grid, hard_frac, color="C2", lw=2.0, label="2–10 keV (hard)")
ax.axhline(0.01, color="0.5", ls=":", lw=1.0)
ax.text(20.3, 0.012, "scattered floor (1%)", color="0.5", fontsize=8)
ax.axvline(24.0, color="k", ls=":", lw=0.8)
ax.text(24.05, 1.5e-4, "Compton-thick", fontsize=8)

ax.set_xlim(20.0, 26.0)
ax.set_ylim(1e-5, 2.0)
ax.set_xlabel(r"$\log N_H$ [cm$^{-2}$]")
ax.set_ylabel("observed / intrinsic band flux")
ax.legend(fontsize=9, frameon=False, loc="lower left")

fig.tight_layout()
plt.savefig("plot_xray_nh_sweep.png", dpi=150, bbox_inches="tight")
