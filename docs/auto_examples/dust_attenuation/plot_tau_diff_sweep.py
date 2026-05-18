"""
Diffuse ISM Optical Depth (τ_diff)
==================================

Diffuse ISM dust τ_diff attenuates all stellar light (young + old).
Higher τ_diff reddens the optical continuum and weakens the 4000 Å break
— a signature of aging stellar populations.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_tau_diff_sweep_001.png
   :alt: plot_tau_diff_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import SEDModel, load_ssp, recipes
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()

model = SEDModel.from_groups(ssp_data=load_ssp(), **recipes.dust_demo())

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "dust_tau_diff",
    [0.0, 0.2, 0.5, 1.0, 2.0],
    ax=ax,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$\tau_{{\rm diff}}$ = {:.1f}",
    wave_range=(1000, 10000),
)
ax.set(
    yscale="log",
    ylim=(1e-1, 1e5),
    title="Diffuse ISM Dust on Typical Galaxy SED",
    ylabel=r"$\lambda F_\lambda$ (normalized at 5500 Å, log scale)",
)
fig.tight_layout()
plt.savefig("plot_tau_diff_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
