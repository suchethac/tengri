"""
AGN X-ray spectral hardness: photon index γ controls power-law steepness
=========================================================================

The X-ray photon index γ controls how rapidly the AGN corona's power-law
spectrum falls off above a few keV. Flat spectra (low γ ~1.4) extend more
photons to high energies; steep spectra (high γ ~2.4) drop quickly. We vary
γ across its typical observational range at fixed bolometric luminosity.

Reference: Wilkins et al. 2020, MNRAS, 493, 5548 (α_ox relation and
photon index dependence).
"""

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.xray import xray_agn_corona

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wavelength = jnp.logspace(np.log10(0.0124), np.log10(124.0), 512)
wave_keV = 12.398 / np.array(wavelength)

gamma_values = np.array([1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
norm = mpl.colors.Normalize(vmin=gamma_values.min(), vmax=gamma_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for gamma in gamma_values:
    l_xray = xray_agn_corona(wavelength, L_agn_bol=1e45, gamma=gamma, E_cut=300.0, alpha_ox=-1.4)
    ax.loglog(wave_keV, np.array(l_xray), lw=1.4, color=cmap(norm(gamma)))

ax.set_xlim(0.1, 1000)
ax.set_ylim(1e21, 5e24)
ax.set_xlabel(r"Energy [keV]")
ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Photon index $\gamma$")

fig.tight_layout()
fig.savefig("plot_xray_gamma_sweep.png", dpi=150, bbox_inches="tight")
