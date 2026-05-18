"""
Birth Cloud Optical Depth (τ_BC)
================================

Birth-cloud dust τ_BC controls how much of the youngest stellar light
escapes the cocoon. Higher τ_BC reddens the UV and suppresses nebular
emission from embedded HII regions.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_tau_bc_sweep_001.png
   :alt: plot_tau_bc_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import SEDModel, load_ssp, recipes
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()

# τ_BC effects are clearest on a young population — override the recipe's
# typical-galaxy SFH with a 500 Myr-old burst.
recipe = recipes.dust_demo()
recipe["sfh"].update(peak_lbt_gyr=0.5, width_gyr=0.3)
model = SEDModel.from_groups(ssp_data=load_ssp(), **recipe)

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "dust_tau_bc",
    [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
    ax=ax,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$\tau_{{BC}}$ = {:.1f}",
    wave_range=(1000, 10000),
)
ax.set(
    yscale="log",
    ylim=(1e-1, 1e5),
    title="Birth Cloud Dust on Young Star-Forming Galaxy SED",
    ylabel=r"$\lambda F_\lambda$ (normalized at 5500 Å, log scale)",
)
fig.tight_layout()
plt.savefig("plot_tau_bc_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
