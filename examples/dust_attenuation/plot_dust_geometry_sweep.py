"""
Dust geometry shapes the extinction: screen vs mixed vs clumpy
==============================================================

Three dust geometries—foreground screen (power-law), mixed slab (Calzetti),
and clumpy two-phase (SMC)—proxy different physical arrangements via their
attenuation laws. At fixed τ_V = 1, geometry controls the spectral shape:
screens are reddest, clumpy geometries are greyest. Transmission curves show
how each law transforms a stellar continuum.

Reference: Witt & Gordon 2000, ApJ, 528, 799 (dust geometry classification).
"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style
from tengri.dust import resolve_dust_law

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave = jnp.linspace(1000.0, 10000.0, 2000)
tau_v = 1.0

geometries = {
    "Screen (foreground)": resolve_dust_law("power_law")(wave, n_slope=-0.7),
    "Mixed (slab)": resolve_dust_law("calzetti")(wave),
    "Clumpy (two-phase)": resolve_dust_law("smc")(wave),
}

colors = ["C0", "C1", "C2"]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for (label, k), color in zip(geometries.items(), colors):
    ax.plot(wave / 1e4, np.exp(-tau_v * np.array(k)), lw=1.4, color=color, label=label)

ax.axhline(1.0, ls="--", color="black", lw=0.8, alpha=0.3)
ax.axvline(0.55, ls=":", color="grey", lw=0.8, alpha=0.5)

ax.set_xlim(0.08, 1.0)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"Transmission: $\exp(-\tau_V \, k(\lambda))$")
ax.legend(fontsize=9, frameon=False, loc="lower left")

fig.tight_layout()
fig.savefig("plot_dust_geometry_sweep.png", dpi=150, bbox_inches="tight")
