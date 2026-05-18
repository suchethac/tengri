r"""
α/Fe Enhancement (met_alpha_fe)
================================

[α/Fe] records the chemical enrichment history: high [α/Fe] signals rapid
enrichment by core-collapse SNe before Type Ia SNe can dilute the alpha
elements. In the SED, enhanced alpha suppresses iron absorption features
in the optical.

Old-passive recipe + sweep ``met_alpha_fe`` from -0.2 to +0.6.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_alpha_fe_sweep_001.png
   :alt: plot_alpha_fe_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import SEDModel, load_ssp, recipes
from tengri.analysis.plotting import setup_style, sweep_parameter

setup_style()

# Old-passive galaxy — [α/Fe] effects are clearest where iron features dominate.
recipe = recipes.dust_demo()
recipe["sfh"].update(peak_lbt_gyr=8.0, log_peak_sfr=0.5, width_gyr=1.5, skew=0.0, trunc=10.0)
recipe["dust"].update(tau_bc=0.0, tau_diff=0.1)
model = SEDModel.from_groups(ssp_data=load_ssp(), **recipe)

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "met_alpha_fe",
    [-0.2, 0.0, 0.2, 0.4, 0.6],
    ax=ax,
    cmap="magma",
    label_fmt=r"$[\alpha/\mathrm{{Fe}}]$ = {:.1f}",
    wave_range=(3500, 9000),
)
ax.set(
    title=r"$\alpha$-element Enhancement: Impact on Optical Absorption Features",
    ylabel=r"$\lambda F_\lambda$ (normalized at 5500 Å)",
    ylim=(0, 2.5e4),
)
fig.tight_layout()
plt.savefig("plot_alpha_fe_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
