"""
Ionization Parameter (logU)
============================

Higher ionisation parameter `log U` drives stronger [OIII] emission
and pulls the galaxy toward the Seyfert region on the BPT diagram.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_logu_sweep_001.png
   :alt: plot_logu_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, load_ssp
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()


ssp = load_ssp()

# --- Build model: young star-forming galaxy ---
spec = Parameters(
    nebular_cue=True,
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),  # Peak ~500 Myr ago (young)
    sfh_tsnorm_width_gyr=Fixed(0.3),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),  # Solar-ish
    dust_tau_bc=Fixed(0.0),  # No dust
    dust_tau_diff=Fixed(0.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    neb_logU=Fixed(-3.0),  # Will sweep this
    neb_logZ_gas=Fixed(-0.3),  # Match stellar metallicity
)
model = SEDModel(spec, ssp)

# --- Sweep ionization parameter (logU) ---
values = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5]

fig, ax = sweep_parameter(
    model,
    "neb_logU",
    values,
    cmap=SWEEP_CMAPS["nebular"],
    label_fmt=r"$\log U$ = {:.1f}",
    wave_range=(4000, 8000),
)
ax.set_title("Ionization Parameter: Impact on Optical Emission Lines", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
ax.set_ylim(0, 50_000)
plt.tight_layout()
plt.savefig("plot_logu_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
