"""
Double Power-Law SFH: Rising Slope α
=====================================

The rising slope α controls how quickly star formation builds up before the peak.
Steeper α = more abrupt onset, younger mean age.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_dpl_alpha_sweep_001.png
   :alt: plot_dpl_alpha_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, Uniform, load_ssp, setup_style
from tengri.analysis.plotting import SWEEP_CMAPS, sfh_sed_comparison

setup_style()


ssp = load_ssp()

# Build Parameters with double power-law SFH
spec = Parameters(
    mean_sfh_type="dpl",
    sfh_dpl_alpha=Uniform(0.3, 6.0),  # will be overridden
    sfh_dpl_beta=Fixed(1.0),
    sfh_dpl_tau_gyr=Fixed(3.0),
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model = SEDModel(spec, ssp)

# Sweep parameter
values = [0.3, 0.7, 1.5, 3.0, 6.0]

fig = sfh_sed_comparison(model, "sfh_dpl_alpha", values, cmap=SWEEP_CMAPS["sfh"])
fig.suptitle("Double Power-Law SFH: Rising Slope α", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_dpl_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
