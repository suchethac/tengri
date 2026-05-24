"""
The 2175 Å UV bump traces small-grain dust populations
=======================================================

The 2175 Å UV bump from PAHs and small graphite grains sweeps from absent to
Milky-Way strength via the ``dust_bump_strength`` knob. At zero, the attenuation
curve is a smooth power law; at MW-like values, the bump dominates the UV. We
show the attenuation law (not a galaxy SED) to isolate the curve shape.

Reference: Kriek & Conroy 2013, ApJ, 775, L16 (extended attenuation model).
"""

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.dust import resolve_dust_law

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave = jnp.linspace(1000.0, 10000.0, 2000)
dust_fn = resolve_dust_law("kriek_conroy")
bump_values = np.linspace(0.0, 4.0, 7)
norm = mpl.colors.Normalize(vmin=bump_values.min(), vmax=bump_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for bump in bump_values:
    k_lambda = dust_fn(wave, dust_bump_strength=bump, dust_delta=0.0)
    ax.plot(
        wave / 1e4,
        k_lambda,
        lw=1.4,
        color=cmap(norm(bump)),
    )

ax.axvline(0.2175, ls=":", color="red", lw=1.0, alpha=0.6)
ax.set_xlim(0.08, 1.0)
ax.set_ylim(0, 3.5)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$k(\lambda)$ (normalized at 5500 $\mathrm{\AA}$)")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"UV bump strength")

fig.tight_layout()
fig.savefig("plot_uv_bump_sweep.png", dpi=150, bbox_inches="tight")
