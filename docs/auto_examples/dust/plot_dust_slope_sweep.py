"""
Attenuation Curve Slope (δ)
============================

The power-law slope δ modifies the Calzetti attenuation curve shape.
Negative δ steepens UV attenuation; positive δ flattens it. This controls
whether dust absorbs more or less light at short wavelengths relative to
optical.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Parameters, SEDModel, load_ssp_data
from tengri.analysis.plotting import SWEEP_CMAPS, setup_style, sweep_parameter

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

# --- Build model: typical galaxy ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
    sfh_tsnorm_width_gyr=Fixed(1.5),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),  # Will sweep this
    redshift=Fixed(0.1),
)
model = SEDModel(spec, ssp)

# --- Sweep dust_slope ---
values = [-1.5, -0.7, 0.0, 0.5]
fig, ax = sweep_parameter(
    model,
    "dust_slope",
    values,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$\delta$ = {:.1f}",
    wave_range=(1000, 10000),
    normalize_at=5500.0,
)
ax.set_title("Dust Attenuation Curve Slope: UV vs. Optical Hardness", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
plt.tight_layout()
plt.savefig("plot_dust_slope_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
