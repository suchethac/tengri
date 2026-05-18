r"""
Metallicity Sweep: Panchromatic SED
====================================

Sweep ``met_logzsol`` ∈ {−1.5, −0.7, 0, +0.5} on a composite SED from
912 Å (Lyman limit) to 30 μm (mid-IR), with modified-blackbody dust
emission turned on so the metallicity-vs-energy-balance interaction is
visible.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_logzsol_panchromatic_001.png
   :alt: plot_logzsol_panchromatic
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import FIXED, Fixed, SEDModel, load_ssp, recipes
from tengri.analysis.plotting import setup_style, sweep_parameter

setup_style()

# Star-forming galaxy with modified-blackbody dust emission at z=0.2.
recipe = recipes.dust_demo()
recipe["sfh"].update(peak_lbt_gyr=1.0, log_peak_sfr=1.0, width_gyr=0.8, trunc=3.0)
recipe["dust"].update(tau_bc=1.0, tau_diff=0.5)
recipe["dust"]["emission"] = {"type": "modified_blackbody", "*": FIXED, "T": 30.0, "beta_ir": 1.8}
recipe["redshift"] = Fixed(0.2)
model = SEDModel.from_groups(ssp_data=load_ssp(), **recipe)

fig, ax = plt.subplots(figsize=(8, 5))
sweep_parameter(
    model,
    "met_logzsol",
    [-1.5, -0.7, 0.0, 0.5],
    ax=ax,
    cmap="viridis",
    label_fmt=r"$\log Z/Z_\odot$ = {:.1f}",
    wave_range=(912, 3e5),  # 912 Å (Lyman limit) → 30 μm (mid-IR)
)
ax.set(
    xscale="log",
    yscale="log",
    title=r"Metallicity Impact on Panchromatic SED",
    xlabel=r"Wavelength [$\AA$]",
    ylabel=r"$\lambda F_\lambda$ (not normalized)",
)

for wl in (1215, 5500, 1e4, 1e5):
    ax.axvline(wl, color="grey", ls=":", lw=0.5, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_logzsol_panchromatic.png", dpi=150, bbox_inches="tight")
plt.show()
