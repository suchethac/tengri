"""
PAH Mass Fraction (q_PAH)
=========================

PAH mass fraction controls strength of polycyclic aromatic hydrocarbon
mid-infrared emission features. Higher q_PAH produces stronger features at
3.3, 6.2, 7.7, 8.6, 11.3 μm. Range varies by dust model.
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
    "dust_qpah",
    [0.5, 1.5, 2.5, 4.5, 6.0],
    ax=ax,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$q_{{PAH}}$ = {:.1f}%",
    wave_range=(3e4, 2e5),
    normalize_at=None,
)
ax.set(
    xscale="log",
    yscale="log",
    ylim=(1e31, 1e37),
    ylabel=r"$\lambda F_\lambda$ (not normalized)",
)
fig.tight_layout()
fig.savefig("plot_qpah_sweep.png", dpi=150, bbox_inches="tight")
