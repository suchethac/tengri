"""
Gas Metallicity (log Z/Zsun)
=============================

Gas metallicity controls [NII]/Hα and [OIII]/Hβ — the primary optical
metallicity diagnostics. These ratios move galaxies on the BPT diagram and
are used to measure oxygen abundances in star-forming galaxies.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_logz_gas_sweep_001.png
   :alt: plot_logz_gas_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, load_ssp
from tengri.analysis.plotting import sweep_parameter
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
    met_logzsol=Fixed(-0.3),  # Solar-ish stellar metallicity
    dust_tau_bc=Fixed(0.0),  # No dust
    dust_tau_diff=Fixed(0.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    neb_logU=Fixed(-3.0),  # Fixed ionization
    neb_logZ_gas=Fixed(-0.3),  # Will sweep this
)
model = SEDModel(spec, ssp)

# --- Sweep gas metallicity ---
values = [-1.5, -0.7, -0.3, 0.0, 0.3]

fig, ax = sweep_parameter(
    model,
    "neb_logZ_gas",
    values,
    cmap="YlGn",
    label_fmt=r"$\log Z/Z_\odot$ = {:.1f}",
    wave_range=(4500, 7500),
)
ax.set_title("Gas Metallicity: Impact on Optical Forbidden Lines", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
ax.set_ylim(0, 30_000)
plt.tight_layout()
plt.savefig("plot_logz_gas_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
