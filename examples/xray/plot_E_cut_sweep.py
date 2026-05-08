"""
High-Energy Cutoff Variation
=============================

Sweep the exponential cutoff E_cut ∈ {100, 200, 300, 500, 1000} keV at
fixed γ = 1.8, α_ox = −1.4, L_bol = 10⁴⁵ erg/s. The spectrum departs from
the power-law above ~0.3 × E_cut and rolls over rapidly at higher energies.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_E_cut_sweep_001.png
   :alt: plot_E_cut_sweep
   :class: sphx-glr-single-img

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.xray import xray_agn_corona

setup_style()

# Wavelength grid spans 0.1 keV (124 A) to 1000 keV (0.0124 A).
# E[keV] = 12.398 / lambda[A].
wavelength = jnp.logspace(np.log10(0.0124), np.log10(124.0), 512)  # Angstrom
wave_keV = 12.398 / np.array(wavelength)

fig, ax = plt.subplots(figsize=(8, 6))

# High-energy cutoff sweep
E_cut_values = [100, 200, 300, 500, 1000]
L_bol = 1e45  # erg/s (fixed)
n_curves = len(E_cut_values)

colors = plt.cm.viridis(np.linspace(0.0, 0.85, n_curves))

for E_cut, color in zip(E_cut_values, colors):
    l_xray = xray_agn_corona(wavelength, L_agn_bol=L_bol, gamma=1.8, E_cut=E_cut, alpha_ox=-1.4)
    ax.loglog(wave_keV, np.array(l_xray), lw=2.0, color=color, label=f"E_cut = {E_cut} keV")

ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title(
    r"AGN X-ray Corona: High-Energy Cutoff E$_{\mathrm{cut}}$ Variation"
    "\n"
    r"(γ=1.8, L$_{\mathrm{bol}}=10^{45}$ erg/s)"
)
ax.set_xlim(0.1, 1000)
ax.set_ylim(1e21, 5e24)
ax.legend(fontsize=11, frameon=False)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_E_cut_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
