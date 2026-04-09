"""
Double Power-Law SFH: Falling Slope β
======================================

The falling slope β controls post-peak quenching. Large β gives rapid quenching;
small β gives a gentle tail.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Model, ParamSpec, Uniform, load_ssp_data, setup_style
from tengri.plotting import sfh_sed_comparison

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

# Build ParamSpec with double power-law SFH
spec = ParamSpec(
    mean_sfh_type="dpl",
    sfh_dpl_alpha=Fixed(1.5),
    sfh_dpl_beta=Uniform(0.3, 10.0),  # will be overridden
    sfh_dpl_tau_gyr=Fixed(3.0),
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model = Model(spec, ssp)

# Sweep parameter
values = [0.3, 1.0, 2.0, 5.0, 10.0]
fig = sfh_sed_comparison(model, "sfh_dpl_beta", values, cmap="Reds")
fig.suptitle("Double Power-Law SFH: Falling Slope β", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_dpl_beta_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
