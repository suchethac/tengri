"""
Ionization Parameter (logU)
============================

How does the ionization parameter sweep the emission line strengths and SED shape?
Higher logU drives stronger [OIII] emission and shifts galaxies toward the Seyfert
region on the BPT diagram.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Model, ParamSpec, Uniform, load_ssp_data
from tengri.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

setup_style()


def _find_ssp():
    """Find SSP data file in standard locations."""
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

# --- Build model: young star-forming galaxy ---
spec = ParamSpec(
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
model = Model(spec, ssp)

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
plt.tight_layout()
plt.savefig("plot_logu_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
