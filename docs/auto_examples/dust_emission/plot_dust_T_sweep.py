"""
Modified Blackbody Dust Temperature
====================================

Dust temperature T sets the far-infrared peak via Wien's displacement law.
Higher T shifts the peak blueward into the mid-IR; lower T shifts it
redward toward the submillimeter.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dust_T_sweep_001.png
   :alt: plot_dust_T_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import FIXED, SEDModel, load_ssp, recipes
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()

recipe = recipes.dust_demo()
recipe["dust"]["emission"] = {
    "type": "modified_blackbody",
    "*": FIXED,
    "T": 35.0,
    "beta_ir": 1.6,
}
model = SEDModel.build(ssp_data=load_ssp(), **recipe)

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "dust_T",
    [20, 30, 40, 60, 80],
    ax=ax,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"T = {:.0f} K",
    wave_range=(1e5, 1e7),
    normalize_at=None,
)
ax.set(
    xscale="log",
    yscale="log",
    ylim=(1e31, 1e40),
    title="Dust Temperature: Far-IR Peak Position and Shape",
    ylabel=r"$\lambda F_\lambda$ (not normalized)",
)
fig.tight_layout()
plt.savefig("plot_dust_T_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
