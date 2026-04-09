"""
Diffuse ISM Optical Depth (τ_diff)
===================================

The diffuse ISM attenuation affects all stellar light (not just young
stars). Higher τ_diff reddens the optical continuum and weakens the
4000 Å break, a signature of aging stellar populations.
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

# --- Build model: typical star-forming galaxy ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),  # Peak ~2 Gyr ago
    sfh_tsnorm_width_gyr=Fixed(1.5),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.5),  # Modest birth cloud dust
    dust_tau_diff=Fixed(0.3),  # Will sweep this
    dust_slope=Fixed(-0.7),  # Calzetti-like
    redshift=Fixed(0.1),
)
model = Model(spec, ssp)

# --- Sweep τ_diff ---
values = [0.0, 0.3, 0.7, 1.5, 3.0]
fig, ax = sweep_parameter(
    model,
    "dust_tau_diff",
    values,
    cmap=SWEEP_CMAPS["dust"],
    label_fmt=r"$\tau_{diff}$ = {:.1f}",
    wave_range=(1000, 10000),
)
ax.set_title("Diffuse ISM Dust: Impact on Galaxy Attenuation", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
plt.tight_layout()
plt.savefig("plot_tau_diff_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
