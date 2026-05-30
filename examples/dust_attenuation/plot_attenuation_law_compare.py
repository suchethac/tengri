"""
Dust attenuation laws: family comparison across UV to near-IR
==============================================================

Dust attenuation laws encode how interstellar dust preferentially absorbs
short-wavelength (UV) starlight relative to optical/IR. The wavelength
dependence is empirically calibrated to extinction measurements in the Milky Way
(Cardelli+1989), Large/Small Magellanic Clouds (Pei 1992), and starburst
galaxies (Calzetti+2000, Kriek+Conroy 2013).

This gallery overlays six textbook attenuation laws on two scales:
- Top panel: linear scale (5500 Å to 1 μm), showing normalized A_λ/A_V
- Bottom panel: log-log scale (1000 Å to 30 μm), revealing the 2175 Å bump
  (characteristic of Milky Way dust) and UV slope steepness. SMC shows the
  steepest UV rise and no bump; Cardelli MW and Kriek+Conroy retain the
  2175 Å bump; Calzetti and Salim flatten the UV slope.

References:
- Calzetti et al. 2000, ApJ, 533, 682 (starburst attenuation)
- Cardelli, Clayton & Mathis 1989, ApJ, 345, 245 (MW extinction)
- Charlot & Fall 2000, ApJ, 539, 718 (power-law families)
- Kriek & Conroy 2013, ApJL, 775, L16 (flexible Calzetti+bump+slope)
- Pei 1992, ApJ, 395, 130 (SMC, LMC extinction)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.dust import list_laws

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave_linear = jnp.linspace(1000.0, 10000.0, 2000)
wave_log_aa = jnp.logspace(3.0, 4.477, 1500)

fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(7.5, 8.0))

# Top panel: linear scale
laws = list_laws()
for label, fn in laws.items():
    ax_lin.plot(wave_linear / 1e4, fn(wave_linear), label=label, lw=1.4)

ax_lin.axvline(0.55, ls=":", color="grey", lw=0.8, alpha=0.5)
ax_lin.axvline(0.2175, ls=":", color="red", lw=0.8, alpha=0.5)
ax_lin.set_xlim(0.08, 1.0)
ax_lin.set_ylim(0, 3.5)
ax_lin.set_xlabel(r"Wavelength [$\mu$m]")
ax_lin.set_ylabel(r"Attenuation $k(\lambda) = A_\lambda / A_V$")
ax_lin.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
ax_lin.grid(True, alpha=0.2)

# Bottom panel: log-log scale with color palette (from plot_attenuation_law_family.py)
colors = {
    "Calzetti+2000": "#1f77b4",
    "Charlot & Fall (slope=-0.7)": "#ff7f0e",
    "Cardelli+1989 (MW, Rv=3.1)": "#2ca02c",
    "SMC (Gordon+2003)": "#d62728",
    "Kriek & Conroy 2013": "#9467bd",
    "Salim+2018": "#8c564b",
}

laws_headline = list_laws(headline=True)
for label, fn in laws_headline.items():
    k_wave = fn(wave_log_aa)
    ax_log.loglog(
        np.array(wave_log_aa),
        np.array(k_wave),
        lw=2.2,
        label=label,
        color=colors.get(label, "C0"),
    )

ax_log.axvline(2175.0, color="gray", lw=1.0, ls=":", alpha=0.5)
ax_log.text(2175.0, 0.3, "2175 Å bump", fontsize=8, ha="center", color="gray", alpha=0.6)

ax_log.axvline(5500.0, color="black", lw=0.8, ls="--", alpha=0.3)
ax_log.text(5500.0, 0.15, "V-band norm", fontsize=8, ha="center", color="black", alpha=0.4)

ax_log.set_xlim(1000, 30000)
ax_log.set_ylim(0.1, 10)
ax_log.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize=11)
ax_log.set_ylabel(r"Attenuation curve $k(\lambda) = A_\lambda / A_V$", fontsize=11)
ax_log.legend(loc="upper right", frameon=True, fontsize=8)
ax_log.grid(True, alpha=0.2, which="both")

fig.tight_layout()
plt.savefig("plot_attenuation_law_compare.png", dpi=150, bbox_inches="tight")
