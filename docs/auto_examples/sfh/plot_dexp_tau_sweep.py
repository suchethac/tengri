"""
Delayed-τ SFH: Star Formation Timescale
========================================

The delayed-exponential timescale `τ` sets how quickly the SFH falls
after its peak. Shorter `τ` means faster quenching, older mean stellar age.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dexp_tau_sweep_001.png
   :alt: plot_dexp_tau_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, Uniform, load_ssp, setup_style
from tengri.analysis.plotting import sfh_sed_comparison

setup_style()


ssp = load_ssp()

# Build Parameters with delayed exponential SFH
spec = Parameters(
    mean_sfh_type="dexp",
    sfh_dexp_log_peak_sfr=Fixed(1.0),
    sfh_dexp_tau_gyr=Uniform(0.1, 10.0),  # will be overridden
    sfh_dexp_start_gyr=Fixed(10.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model = SEDModel(spec, ssp)

# Sweep parameter
values = [0.5, 1.0, 2.0, 5.0, 10.0]

fig = sfh_sed_comparison(model, "sfh_dexp_tau_gyr", values, cmap="Blues")
fig.suptitle("Delayed Exponential SFH: Timescale τ", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_dexp_tau_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
