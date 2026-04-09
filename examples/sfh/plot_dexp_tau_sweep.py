"""
Delayed-τ SFH: Star Formation Timescale
========================================

How does the e-folding timescale τ reshape the star formation history and the
resulting galaxy SED? Shorter τ = faster quenching after the peak.
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

# Build ParamSpec with delayed exponential SFH
spec = ParamSpec(
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

model = Model(spec, ssp)

# Sweep parameter
values = [0.5, 1.0, 2.0, 5.0, 10.0]
fig = sfh_sed_comparison(model, "sfh_dexp_tau_gyr", values, cmap="Blues")
fig.suptitle("Delayed Exponential SFH: Timescale τ", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_dexp_tau_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
