"""
Log-Normal SFH: Peak Lookback Time
===================================

When did this galaxy form most of its stars? The peak lookback time shifts
the SFH and changes UV slope, 4000 Å break, and NIR mass.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_lnorm_peak_sweep_001.png
   :alt: plot_lnorm_peak_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, Uniform, load_ssp, setup_style
from tengri.analysis.plotting import SWEEP_CMAPS, sfh_sed_comparison

setup_style()


ssp = load_ssp()

# Build Parameters with log-normal SFH
spec = Parameters(
    mean_sfh_type="lnorm",
    sfh_lnorm_log_peak_sfr=Fixed(1.0),
    sfh_lnorm_peak_lbt_gyr=Uniform(1.0, 11.0),  # will be overridden
    sfh_lnorm_width_gyr=Fixed(0.3),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model = SEDModel(spec, ssp)

# Sweep parameter
values = [1.0, 3.0, 5.0, 8.0, 11.0]

fig = sfh_sed_comparison(model, "sfh_lnorm_peak_lbt_gyr", values, cmap=SWEEP_CMAPS["sfh"])
fig.suptitle("Log-Normal SFH: Peak Lookback Time", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_lnorm_peak_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
