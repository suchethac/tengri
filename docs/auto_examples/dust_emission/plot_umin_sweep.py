"""
Draine & Li Minimum Radiation Field (U_min)
============================================

Minimum radiation field intensity U_min controls diffuse dust heating.
Higher U_min implies hotter dust and FIR peak shifted blueward toward
shorter wavelengths.
"""

import warnings

import matplotlib.pyplot as plt

from tengri import FIXED, SEDModel, load_ssp, recipes
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

recipe = recipes.dust_demo()
recipe["dust"]["emission"] = {
    "type": "draine_li2007",
    "*": FIXED,
    "umin": 1.0,
    "gamma_dl": 0.01,
    "qpah": 2.5,
}
model = SEDModel.build(ssp_data=load_ssp(), **recipe)

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "dust_umin",
    [0.1, 1.0, 5.0, 25.0],
    ax=ax,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$U_{{min}}$ = {:.1f}",
    wave_range=(1e5, 1e7),
    normalize_at=None,
)
ax.set(
    xscale="log",
    yscale="log",
    ylim=(1e32, 1e40),
    ylabel=r"$\lambda F_\lambda$ (not normalized)",
)
fig.tight_layout()
plt.savefig("plot_umin_sweep.png", dpi=150, bbox_inches="tight")
