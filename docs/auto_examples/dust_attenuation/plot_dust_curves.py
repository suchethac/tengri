"""
Dust attenuation laws from UV through near-infrared
====================================================

The six headline dust attenuation laws plotted over the full UV-through-NIR
range (0.1–3 μm), extending beyond the 2175 Å bump region to show how curves
flatten in the infrared. Red-shifted galaxies observe longer wavelengths at
rest frame, so the IR slope controls K-correction factors and SED fitting degeneracies.

Reference: Cardelli et al. 1989, ApJ, 345, 245 (extended optical extinction).
"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt

import tengri
from tengri.analysis.plotting import setup_style
from tengri.dust import list_laws

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave = jnp.linspace(1000.0, 30000.0, 2000)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for label, fn in list_laws().items():
    ax.plot(wave / 1e4, fn(wave), label=label, lw=1.4)

ax.axvline(0.55, ls=":", color="grey", lw=0.5, alpha=0.5)
ax.axvline(0.2175, ls=":", color="grey", lw=0.5, alpha=0.5)

ax.set_xlim(0.08, 3.0)
ax.set_ylim(0, None)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$k(\lambda)$ (normalized at 5500 $\mathrm{\AA}$)")
ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)

fig.tight_layout()
fig.savefig("plot_dust_curves.png", dpi=150, bbox_inches="tight")
