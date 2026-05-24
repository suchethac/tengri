"""
Star-forming X-ray budget: metallicity-dependent HMXBs + diffuse hot gas
========================================================================

A star-forming galaxy's diffuse X-ray budget below ~ 10 keV has three
ingredients: high-mass X-ray binaries (HMXB) tracking the *current* SFR,
low-mass X-ray binaries (LMXB) tracking the *integrated* stellar mass,
and a thermal soft-X-ray plume from hot ISM/CGM. The metallicity
dependence of HMXBs is the dominant systematic at z > 3 — sub-solar
stellar populations form more massive black holes per unit SFR, raising
``L_X / SFR`` by up to ~ 0.5 dex at ``Z = 0.001`` (Fragos+13, Lehmer+19).

This sweep mirrors the CIGALE ``yang20`` decomposition adopted in
Yang+22 and Matsumoto+26 — the spectral budget you should expect from
a typical SFR ~ 10 M☉/yr starburst, separated by component and swept
through the metallicity range where high-z galaxies live.

References
----------
- Lehmer, B. D. et al. 2019, ApJ, 883, 109. HMXB metallicity quartic.
- Lehmer, B. D. et al. 2014, ApJ, 789, 52. LMXB age quartic.
- Mineo, S. et al. 2012, MNRAS, 426, 1870. Hot-gas SFR scaling.
- Yang, G. et al. 2022, ApJ, 927, 192. CIGALE ``yang20`` X-ray module.
"""

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.xray import xray_hotgas, xray_xrb

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# ------------------------------------------------------------------- grid
wave = jnp.logspace(np.log10(0.5), np.log10(124.0), 1024)  # 0.1 – 25 keV
E_keV = 12.398 / np.asarray(wave)

SFR = 10.0  # M☉/yr — modest starburst
MSTAR = 5e10  # M☉    — like the Milky Way

# ------------------------------------------------------------------- panels
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) Three-component decomposition at Z = Z_sun
ax = axes[0]
hmxb = xray_xrb(wave, sfr=SFR, stellar_mass=0.0, metallicity_z=0.02)
lmxb = xray_xrb(wave, sfr=0.0, stellar_mass=MSTAR, stellar_age_gyr=8.0)
hot = xray_hotgas(wave, sfr=SFR)

ax.loglog(E_keV, np.asarray(hmxb), color="C0", lw=2.0, label="HMXB (tracks SFR)")
ax.loglog(E_keV, np.asarray(lmxb), color="C1", lw=2.0, label="LMXB (tracks M*)")
ax.loglog(E_keV, np.asarray(hot), color="C2", lw=2.0, label="hot gas (Mineo+12)")
ax.loglog(E_keV, np.asarray(hmxb + lmxb + hot), color="k", lw=1.4, ls="--", label="total")

ax.set_xlim(0.3, 30)
ax.set_ylim(1e21, 1e25)
ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.text(
    0.04,
    0.96,
    f"SFR = {SFR:.0f} M⊙/yr\nM* = {np.log10(MSTAR):.1f} dex\n Z = Z⊙",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=9,
)
ax.legend(fontsize=9, frameon=False, loc="lower right")

# (b) HMXB metallicity sequence — the new physics
ax = axes[1]
Z_grid = np.array([0.0001, 0.001, 0.004, 0.008, 0.02, 0.04])
cmap = plt.get_cmap("viridis")
norm = mpl.colors.LogNorm(vmin=Z_grid.min(), vmax=Z_grid.max())


# 2-10 keV band integral for each Z
def band_int(L_nu, e_lo, e_hi):
    mask = (E_keV >= e_lo) & (E_keV <= e_hi)
    nu = (E_keV[mask] / 12.398) * 2.998e18
    order = np.argsort(nu)
    return float(np.trapezoid(np.asarray(L_nu)[mask][order], nu[order]))


for Z in Z_grid:
    L = xray_xrb(wave, sfr=SFR, stellar_mass=0.0, metallicity_z=Z)
    ax.loglog(E_keV, np.asarray(L), color=cmap(norm(Z)), lw=1.6)

ax.set_xlim(0.3, 30)
ax.set_ylim(1e21, 1e25)
ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu^{\rm HMXB}$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.text(
    0.04,
    0.96,
    "HMXB only,  SFR = 10 M⊙/yr",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=9,
)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label("Stellar metallicity Z (mass fraction)")

# Annotate the Fragos+13 / Lehmer+19 dex enhancement
L_subsol = band_int(xray_xrb(wave, sfr=SFR, stellar_mass=0.0, metallicity_z=0.001), 2, 10)
L_sun = band_int(xray_xrb(wave, sfr=SFR, stellar_mass=0.0, metallicity_z=0.02), 2, 10)
ax.text(
    0.96,
    0.04,
    rf"$\Delta \log L_X(2\!-\!10)$"
    rf" $(Z\!=\!0.001 \to Z_\odot) = {np.log10(L_sun / L_subsol):+.2f}$ dex",
    transform=ax.transAxes,
    va="bottom",
    ha="right",
    fontsize=9,
)

fig.tight_layout()
plt.savefig("plot_xray_metallicity_hotgas.png", dpi=150, bbox_inches="tight")
