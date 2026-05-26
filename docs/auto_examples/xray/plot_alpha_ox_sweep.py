"""
AGN UV-to-X-ray power-law slope α_ox controls relative X-ray normalisation
===========================================================================

The UV-to-X-ray spectral slope α_ox (defined as log(F_X) − log(F_UV) /
log(ν_X) − log(ν_UV)) separates "X-ray loud" quasars (α_ox ~ −1.2, strong
X-ray relative to UV continuum) from "X-ray quiet" systems (α_ox ~ −1.8,
suppressed X-ray). More negative α_ox suppresses the X-ray continuum and
weakens the high-energy tail. We vary α_ox at fixed bolometric luminosity,
showing the anticorrelation of X-ray strength and UV continuum slope.

Reference: Wilkins et al. 2020, MNRAS, 493, 5548 (α_ox correlation study).
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

alpha_ox_values = np.array([-1.0, -1.2, -1.4, -1.6, -1.8])
norm = mpl.colors.Normalize(vmin=alpha_ox_values.min(), vmax=alpha_ox_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for alpha_ox in alpha_ox_values:
    l_xray = xray_agn_corona(wavelength, L_agn_bol=1e45, gamma=1.8, E_cut=300.0, alpha_ox=alpha_ox)
    ax.loglog(wave_keV, np.array(l_xray), lw=1.4, color=cmap(norm(alpha_ox)))

ax.set_xlim(0.1, 1000)
ax.set_ylim(1e20, 1e27)
ax.set_xlabel(r"Energy [keV]")
ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"UV-to-X-ray slope $\alpha_{\rm ox}$")

fig.tight_layout()
plt.savefig("plot_alpha_ox_sweep.png", dpi=150, bbox_inches="tight")
