"""
X-ray Binary Population from Star Formation
=============================================

Plot X-ray luminosity from high and low-mass X-ray binaries (HMXB + LMXB)
as a function of star formation rate and stellar mass. Shows the different
scaling relations for HMXB (SFR-dependent) vs LMXB (mass-dependent).

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_xray_sf_001.png
   :alt: plot_xray_sf
   :class: sphx-glr-single-img

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.xray import xray_xrb

setup_style()

# Wavelength grid: hard X-ray (keV) to soft X-ray
wavelength = jnp.logspace(np.log10(0.1), np.log10(100), 512)  # Angstrom
wave_keV = 1.2398e-4 / (np.array(wavelength) * 1e-8)  # Convert to keV

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- Panel 1: SFR dependence (fixed stellar mass) ---
ax = axes[0, 0]

stellar_mass = 1e11  # Solar masses
for sfr in [0.1, 1.0, 10.0, 100.0]:
    l_xrb = xray_xrb(wavelength, sfr=sfr, stellar_mass=stellar_mass)
    ax.loglog(wave_keV, np.array(l_xrb), lw=1.5, label=f"SFR={sfr:.1f} M_sun/yr")

ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title(f"X-ray Binary SFR Dependence (M_*={stellar_mass:.0e} M_sun)")
ax.legend(fontsize=10, frameon=False)
ax.set_xlim(0.1, 100)
ax.set_ylim(1e20, 1e32)

# --- Panel 2: Stellar mass dependence (fixed SFR) ---
ax = axes[0, 1]

sfr = 10.0  # Solar masses/yr
for m_star in [1e9, 1e10, 1e11, 1e12]:
    l_xrb = xray_xrb(wavelength, sfr=sfr, stellar_mass=m_star)
    ax.loglog(wave_keV, np.array(l_xrb), lw=1.5, label=f"M_*={m_star:.0e} M_sun")

ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title(f"X-ray Binary Mass Dependence (SFR={sfr:.1f} M_sun/yr)")
ax.legend(fontsize=10, frameon=False)
ax.set_xlim(0.1, 100)
ax.set_ylim(1e20, 1e32)

# --- Panel 3: HMXB vs LMXB ratio ---
ax = axes[1, 0]

# Total XRB spectrum
sfr_vals = np.array([1.0, 10.0, 100.0])
m_star_vals = np.array([1e10, 1e11, 1e12])

# Show total X-ray luminosity as function of parameters
for sfr in sfr_vals:
    l_xrb = xray_xrb(wavelength, sfr=sfr, stellar_mass=1e11)
    ax.loglog(wave_keV, np.array(l_xrb), lw=1.5, label=f"SFR={sfr:.1f}")

ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("X-ray Binary Spectral Shape")
ax.legend(fontsize=10, frameon=False)
ax.set_xlim(0.1, 100)
ax.set_ylim(1e20, 1e32)

# --- Panel 4: L_X vs SFR and M_* (heatmap via lines) ---
ax = axes[1, 1]

# Calculate integrated L_X in 2-10 keV band
sfr_range = np.logspace(-1, 2, 20)
m_star_ref = 1e11

colors = plt.cm.viridis(np.linspace(0, 1, len(sfr_range)))

for sfr, color in zip(sfr_range, colors):
    l_xrb = xray_xrb(wavelength, sfr=sfr, stellar_mass=m_star_ref)
    ax.loglog(wave_keV, np.array(l_xrb), lw=1.0, color=color, alpha=0.6)

ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title(f"X-ray Binary SFR Range (M_*={m_star_ref:.0e})")
ax.set_xlim(0.1, 100)
ax.set_ylim(1e20, 1e32)

# Add colorbar-like legend
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=sfr_range.min(), vmax=sfr_range.max())
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.12, aspect=25)
cbar.set_label(r"SFR [M$_\odot$/yr]")

fig.suptitle("X-ray Binaries: SFR and Stellar Mass Dependencies", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("plot_xray_sf.png", dpi=100, bbox_inches="tight")
plt.show()
