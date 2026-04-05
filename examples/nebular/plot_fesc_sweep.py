"""
Ionizing Photon Escape Fraction (f_esc)
========================================

The escape fraction sets how many ionizing photons reach the ISM vs escape
into the IGM. Higher f_esc suppresses all nebular emission lines since fewer
photons remain to ionize the interstellar gas. f_esc = 0 (all photons stay),
f_esc = 1 (all photons escape).
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
    neb_logU=Fixed(-3.0),  # Fixed ionization
    neb_logZ_gas=Fixed(-0.3),
    neb_fesc=Fixed(0.0),  # Will sweep this
)
model = Model(spec, ssp)

# --- Sweep escape fraction ---
values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
fig, ax = sweep_parameter(
    model,
    "neb_fesc",
    values,
    cmap="Purples",
    label_fmt=r"$f_{{\mathrm{{esc}}}}$ = {:.1f}",
    wave_range=(1000, 8000),
)
ax.set_title("Ionizing Photon Escape Fraction: UV to Optical", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
plt.tight_layout()
plt.savefig("plot_fesc_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
