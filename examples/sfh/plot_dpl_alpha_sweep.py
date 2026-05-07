"""
Double Power-Law SFH: Rising Slope α
=====================================

The rising slope α controls how quickly star formation builds up before the peak.
Steeper α = more abrupt onset, younger mean age.
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

# # The sweep_parameter helper creates a single SEDModel instance and calls
# # model.predict_rest_sed(...) in a loop. JAX JIT compilation is cached
# # automatically via tengri's persistent compilation cache (enabled at
# # import time), so repeated forward model calls reuse the compiled kernel.
fig = sfh_sed_comparison(model, "sfh_dpl_alpha", values, cmap="Oranges")
fig.suptitle("Double Power-Law SFH: Rising Slope α", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_dpl_alpha_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
