"""
X-ray binary luminosity scales with SFR (HMXB) and stellar mass (LMXB)
========================================================================

X-ray binaries (XRBs) are the dominant X-ray sources in star-forming
galaxies once an AGN is excluded. High-mass XRBs trace the recent
star-formation rate (Mineo+2012), while low-mass XRBs trace the
integrated stellar mass (Lehmer+2019). The two scalings have different
spectral shapes too: HMXBs are slightly harder, LMXBs slightly softer.
Two side-by-side sweeps — SFR (left) at fixed M_star = 1e11 M☉, and
M_star (right) at fixed SFR = 10 M☉/yr — separate the two channels
on the same axes.

References
----------

- Mineo, Gilfanov & Sunyaev 2012, MNRAS 419, 2095 (HMXB-SFR).
- Lehmer et al. 2019, ApJ 878, 122 (combined L_X-SFR-M_star).

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.xray import xray_xrb

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wavelength = jnp.logspace(np.log10(0.1), np.log10(100.0), 512)
wave_keV = 12.398 / np.array(wavelength)

cmap = plt.get_cmap("viridis")
sfr_values = np.logspace(-1.0, 2.0, 9)
mstar_values = np.logspace(8.0, 12.5, 9)
sfr_norm = mpl.colors.LogNorm(vmin=sfr_values.min(), vmax=sfr_values.max())
mstar_norm = mpl.colors.LogNorm(vmin=mstar_values.min(), vmax=mstar_values.max())

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)

# Left: SFR sweep at fixed M_star (HMXB-dominated when SFR is high)
ax = axes[0]
for sfr in sfr_values:
    l_xrb = np.asarray(xray_xrb(wavelength, sfr=float(sfr), stellar_mass=1.0e11))
    ax.loglog(wave_keV, l_xrb, color=cmap(sfr_norm(sfr)), lw=1.3)
ax.set(
    xlim=(0.1, 100.0),
    ylim=(1.0e20, 1.0e32),
    xlabel="Energy [keV]",
    ylabel=r"$L_\nu$  [erg s$^{-1}$ Hz$^{-1}$]",
)
cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=sfr_norm, cmap=cmap), ax=ax, pad=0.01)
cbar1.set_label(r"SFR  [M$_\odot$ yr$^{-1}$]")
ax.text(0.04, 0.95, r"$M_\star = 10^{11}\,M_\odot$", transform=ax.transAxes, va="top", fontsize=9)

# Right: M_star sweep at fixed SFR (LMXB grows with old population)
ax = axes[1]
for m_star in mstar_values:
    l_xrb = np.asarray(xray_xrb(wavelength, sfr=10.0, stellar_mass=float(m_star)))
    ax.loglog(wave_keV, l_xrb, color=cmap(mstar_norm(m_star)), lw=1.3)
ax.set(xlim=(0.1, 100.0), xlabel="Energy [keV]")
cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=mstar_norm, cmap=cmap), ax=ax, pad=0.01)
cbar2.set_label(r"$M_\star$  [M$_\odot$]")
ax.text(0.04, 0.95, r"SFR $= 10\,M_\odot$ yr$^{-1}$", transform=ax.transAxes, va="top", fontsize=9)

fig.tight_layout()
plt.savefig("plot_xray_sf.png", dpi=150, bbox_inches="tight")
