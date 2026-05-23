r"""
Stellar Metallicity (log Z/Z_⊙)
================================

Stellar metallicity sets the UV-optical SED shape: metal-poor stars are
hotter and bluer (less line blanketing), metal-rich are redder. Sweep
``met_logzsol`` from −2 (0.01 Z_⊙) to +0.2 (1.6 Z_⊙) at fixed SFH and dust.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_logzsol_sweep_001.png
   :alt: plot_logzsol_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import SEDModel, load_ssp, recipes
from tengri.analysis.plotting import setup_style, sweep_parameter

setup_style()

# Intermediate-age galaxy with modest dust (recipe defaults). Sweep met_logzsol.
model = SEDModel.build(ssp_data=load_ssp(), **recipes.dust_demo())

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "met_logzsol",
    [-2.0, -1.5, -1.0, -0.5, -0.3, 0.0, 0.2],
    ax=ax,
    cmap="viridis",
    label_fmt=r"$\log Z/Z_\odot$ = {:.1f}",
    wave_range=(1000, 12000),
)
ax.set(
    title=r"Stellar Metallicity: $\log Z/Z_\odot$ sweep",
    ylabel=r"$\lambda F_\lambda$ (normalized at 5500 Å)",
    ylim=(0, 7.5e4),
)
fig.tight_layout()
plt.savefig("plot_logzsol_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
