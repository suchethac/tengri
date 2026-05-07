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

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Parameters, SEDModel, Uniform, load_ssp_data, setup_style
from tengri.analysis.plotting import sfh_sed_comparison

setup_style()


def _find_ssp():
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

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

# # The sweep_parameter helper creates a single SEDModel instance and calls
# # model.predict_rest_sed(...) in a loop. JAX JIT compilation is cached
# # automatically via tengri's persistent compilation cache (enabled at
# # import time), so repeated forward model calls reuse the compiled kernel.
fig = sfh_sed_comparison(model, "sfh_lnorm_peak_lbt_gyr", values, cmap="Purples")
fig.suptitle("Log-Normal SFH: Peak Lookback Time", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_lnorm_peak_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
