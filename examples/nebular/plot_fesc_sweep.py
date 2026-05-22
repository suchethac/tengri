"""
Ionizing Photon Escape Fraction (f_esc)
========================================

The escape fraction sets how many ionizing photons reach the ISM vs escape
into the IGM. Higher f_esc suppresses all nebular emission lines since fewer
photons remain to ionize the interstellar gas. f_esc = 0 (all photons stay),
f_esc = 1 (all photons escape).

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_fesc_sweep_001.png
   :alt: plot_fesc_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, load_ssp
from tengri.analysis.plotting import SWEEP_CMAPS, sweep_parameter
from tengri.plot import setup_style

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
    neb_logU=Fixed(-3.0),  # Fixed ionization
    neb_logZ_gas=Fixed(-0.3),
    neb_fesc=Fixed(0.0),  # Will sweep this
)
model = SEDModel(spec, ssp)

# --- Sweep escape fraction ---
values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

fig, ax = sweep_parameter(
    model,
    "neb_fesc",
    values,
    cmap=SWEEP_CMAPS["nebular"],
    label_fmt=r"$f_{{\mathrm{{esc}}}}$ = {:.1f}",
    wave_range=(1000, 8000),
)
ax.set_title("Ionizing Photon Escape Fraction: UV to Optical", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
plt.tight_layout()
plt.savefig("plot_fesc_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
