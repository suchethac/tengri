"""
Attenuation Law Comparison
==========================

The six headline attenuation laws at fixed τ_V = 1, UV through NIR.
Highlights the 2175 Å bump and the slope differences between Milky Way,
SMC, and starburst families.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_attenuation_law_compare_001.png
   :alt: plot_attenuation_law_compare
   :class: sphx-glr-single-img

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from tengri.dust import list_laws
from tengri.plot import setup_style

setup_style()

wave = jnp.linspace(1000.0, 10000.0, 2000)

fig, ax = plt.subplots(figsize=(10, 6))
for label, fn in list_laws().items():
    ax.plot(wave / 1e4, fn(wave), label=label, lw=2.0)

ax.axvline(0.55, ls=":", color="grey", lw=0.8, alpha=0.5)
ax.text(0.56, 3.0, "V-band", fontsize=9, color="grey")
ax.axvline(0.2175, ls=":", color="red", lw=1.0, alpha=0.6)
ax.text(0.23, 2.7, "2175 Å bump", fontsize=9, color="red")

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$k(\lambda)$ (normalized at 5500 $\AA$)",
    title="Dust Attenuation Law Comparison (τ_V = 1.0)",
    xlim=(0.1, 1.0),
    ylim=(0, 3.5),
)
ax.legend(fontsize=10, frameon=False, loc="upper left")
fig.tight_layout()
plt.savefig("plot_attenuation_law_compare.png", dpi=150, bbox_inches="tight")
plt.show()
