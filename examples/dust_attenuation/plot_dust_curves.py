"""
Dust Attenuation Curves (UV to NIR)
====================================

Same six headline laws as :doc:`plot_attenuation_law_compare`, but plotted
over the full UV-through-NIR range (0.1–3 μm) instead of zooming into the
UV bump region.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dust_curves_001.png
   :alt: plot_dust_curves
   :class: sphx-glr-single-img

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from tengri.dust import list_laws
from tengri.plot import setup_style

setup_style()

wave = jnp.linspace(1000.0, 30000.0, 2000)

fig, ax = plt.subplots(figsize=(9, 5))
for label, fn in list_laws().items():
    ax.plot(wave / 1e4, fn(wave), label=label, lw=1.5)

ax.axvline(0.55, ls=":", color="grey", lw=0.5, alpha=0.5)
ax.annotate(
    "V-band", xy=(0.56, 0.05), xycoords=("data", "axes fraction"), fontsize=10, color="grey"
)
ax.axvline(0.2175, ls=":", color="grey", lw=0.5, alpha=0.5)
ax.annotate(
    "2175 Å bump", xy=(0.23, 0.85), xycoords=("data", "axes fraction"), fontsize=10, color="grey"
)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$k(\lambda)$ (normalized at 5500 $\AA$)",
    title="Dust Attenuation Curves in tengri",
    xlim=(0.1, 3.0),
    ylim=(0, None),
)
ax.legend(fontsize=10, frameon=False, ncol=2)
fig.tight_layout()
plt.savefig("plot_dust_curves.png", dpi=150, bbox_inches="tight")
plt.show()
