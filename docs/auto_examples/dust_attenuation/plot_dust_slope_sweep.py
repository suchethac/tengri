"""
Attenuation Curve Slope (δ)
===========================

The power-law slope δ steepens (negative) or flattens (positive) UV
attenuation relative to the optical. Controls whether dust absorbs
more or less light at short wavelengths.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dust_slope_sweep_001.png
   :alt: plot_dust_slope_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import SEDModel, load_ssp, recipes
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()

# Bump τ_BC and τ_diff above their recipe defaults — slope effects are clearest
# on a dust-attenuated continuum (and a nonzero τ_BC prevents the unattenuated
# nebular spike from dominating the λF_λ display at high δ).
recipe = recipes.dust_demo()
recipe["dust"].update(tau_bc=1.0, tau_diff=0.5)
model = SEDModel.build(ssp_data=load_ssp(), **recipe)

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "dust_slope",
    [-1.5, -0.7, 0.0, 0.5],
    ax=ax,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$\delta$ = {:.1f}",
    wave_range=(1000, 10000),
)
ax.set(
    title="Dust Attenuation Curve Slope: UV vs Optical Hardness",
    ylabel=r"$\lambda F_\lambda$ (normalized at 5500 Å)",
)
fig.tight_layout()
plt.savefig("plot_dust_slope_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
