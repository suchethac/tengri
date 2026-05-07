"""
Nebular Gas Density: Metallicity Variation
===========================================

Gas phase metallicity affects ionization balance and emission line strengths.
Higher metallicity increases cooling efficiency, affecting the nebular continuum
and emission-line ratios through recombination rate changes.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Parameters, SEDModel, load_ssp_data
from tengri.analysis.plotting import setup_style, sweep_parameter

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

# --- Build model: young star-forming galaxy with variable metallicity ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),  # Peak ~500 Myr ago (young)
    sfh_tsnorm_width_gyr=Fixed(0.3),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),  # Will sweep this
    dust_tau_bc=Fixed(0.1),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)
model = SEDModel(spec, ssp)

# --- Sweep stellar metallicity (proxy for nebular Z) ---
values = [-1.0, -0.7, -0.3, 0.0, 0.2]
fig, ax = sweep_parameter(
    model,
    "met_logzsol",
    values,
    cmap="viridis",
    label_fmt=r"$\log(Z/Z_\odot)$ = {:.1f}",
    wave_range=(4000, 8000),
)
ax.set_title("Metallicity Impact on Stellar SED and Nebular Lines", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
ax.set_ylim(0, 50_000)
plt.tight_layout()
plt.savefig("plot_neb_density_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
