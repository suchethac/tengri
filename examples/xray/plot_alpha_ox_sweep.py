"""
AGN UV-to-X-ray power-law slope alpha_OX controls X-ray normalization
======================================================================

The UV-to-X-ray spectral slope alpha_OX (defined as log F_X minus
log F_UV divided by log nu_X minus log nu_UV) separates X-ray-loud
quasars (alpha_OX around -1.2, strong X-ray relative to the UV
continuum) from X-ray-quiet systems (alpha_OX around -1.8, suppressed
X-ray). The CIGALE-faithful corona derives alpha_OX from L_2500 via
the Just+2007 relation by default; here we sweep ``delta_alpha_ox``
to apply offsets from -0.4 to +0.4 around that empirical value, at
fixed L_2500 (= L_bol = 1e45 erg/s through the standard Hopkins+2007
bolometric correction). More positive delta brightens the corona;
more negative suppresses it.

Reference: Just et al. 2007, ApJ 665, 1004 (alpha_OX-L_2500);
Wilkins et al. 2020, MNRAS 493, 5548 (alpha_OX scatter study).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.plot import setup_style
from tengri.xray import alpha_ox_from_l2500, xray_agn_corona

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Hopkins+2007 bolometric correction at 2500 A
_BC_2500 = 5.15
_NU_2500 = 1.199e15
L_BOL = 1.0e45
L_2500 = L_BOL / (_BC_2500 * _NU_2500)
ALPHA_OX_J07 = float(alpha_ox_from_l2500(L_2500))

wavelength = jnp.logspace(np.log10(0.0124), np.log10(124.0), 512)
wave_keV = 12.398 / np.array(wavelength)

delta_values = np.linspace(-0.4, 0.4, 5)
alpha_ox_effective = ALPHA_OX_J07 + delta_values
norm = mpl.colors.Normalize(vmin=alpha_ox_effective.min(), vmax=alpha_ox_effective.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for delta, alpha_eff in zip(delta_values, alpha_ox_effective):
    l_xray = xray_agn_corona(
        wavelength,
        l_2500_30deg_erg_hz=L_2500,
        gamma=1.8,
        E_cut=300.0,
        delta_alpha_ox=float(delta),
    )
    ax.loglog(wave_keV, np.asarray(l_xray), lw=1.4, color=cmap(norm(alpha_eff)))

ax.set(
    xlim=(0.1, 1000.0),
    ylim=(1.0e20, 1.0e27),
    xlabel="Energy [keV]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"effective $\alpha_{\rm OX}$  (Just+07 baseline $+\delta$)")

fig.tight_layout()
plt.savefig("plot_alpha_ox_sweep.png", dpi=150, bbox_inches="tight")
