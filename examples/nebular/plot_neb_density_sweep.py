"""
Nebular Gas Density: Metallicity Variation
===========================================

Gas phase metallicity affects ionization balance and emission line strengths.
Higher metallicity increases cooling efficiency, affecting the nebular continuum
and emission-line ratios through recombination rate changes.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_neb_density_sweep_001.png
   :alt: plot_neb_density_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, load_ssp
from tengri.analysis.plotting import setup_style, sweep_parameter

setup_style()


ssp = load_ssp()

# --- Build model: young star-forming galaxy with variable metallicity ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),  # Peak ~500 Myr ago (young)
    sfh_tsnorm_width_gyr=Fixed(0.3),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),  # Will sweep this
    dust_tau_bc=Fixed(0.1),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)
model = SEDModel(spec, ssp)

# --- Sweep stellar metallicity (proxy for nebular Z) ---
values = [-1.0, -0.7, -0.3, 0.0, 0.2]

fig, ax = sweep_parameter(
    model,
    "met_logzsol",
    values,
    cmap="viridis",
    label_fmt=r"$\log(Z/Z_\odot)$ = {:.1f}",
    wave_range=(4000, 8000),
)
ax.set_title("Metallicity Impact on Stellar SED and Nebular Lines", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
ax.set_ylim(0, 50_000)
plt.tight_layout()
plt.savefig("plot_neb_density_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
