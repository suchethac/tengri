"""
The six headline dust attenuation laws span MW, SMC, and starburst geometries
=============================================================================

The tengri library offers six attenuation laws covering the morphology-geometry
spectrum: Milky Way (Cardelli), SMC (Pei), starburst (Calzetti, Conroy),
and theoretical models (Kriek & Conroy, power law). At fixed τ_V = 1, their
curves expose the 2175 Å bump (MW/Cardelli), slope differences (SMC is greyer,
Calzetti is redder), and parametric extensions (Kriek & Conroy).

Reference: Cardelli et al. 1989, ApJ, 345, 245 (MW); Pei 1992, ApJ, 395, 130 (SMC).
"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt

from tengri.analysis.plotting import setup_style
from tengri.dust import list_laws

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave = jnp.linspace(1000.0, 10000.0, 2000)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for label, fn in list_laws().items():
    ax.plot(wave / 1e4, fn(wave), label=label, lw=1.4)

ax.axvline(0.55, ls=":", color="grey", lw=0.8, alpha=0.5)
ax.axvline(0.2175, ls=":", color="red", lw=0.8, alpha=0.5)

ax.set_xlim(0.08, 1.0)
ax.set_ylim(0, 3.5)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$k(\lambda)$ (normalized at 5500 $\mathrm{\AA}$)")
ax.legend(fontsize=9, frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig("plot_attenuation_law_compare.png", dpi=150, bbox_inches="tight")
