"""
Stellar Metallicity (log Z/Z_⊙)
================================

Stellar metallicity sets the UV-optical SED shape: metal-poor stars are
hotter and bluer (less line blanketing), while metal-rich stars have
stronger absorption features and redder continua. This plot sweeps
``met_logzsol`` from :math:`-2` (0.01 :math:`Z_\\odot`) to :math:`0.2`
(1.6 :math:`Z_\\odot`) at fixed SFH and dust.
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

# --- Build model: intermediate-age galaxy with modest dust ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(0.5),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
    sfh_tsnorm_width_gyr=Fixed(1.0),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(8.0),
    met_logzsol=Fixed(-0.3),  # Will sweep this
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)
model = SEDModel(spec, ssp)

# --- Sweep stellar metallicity ---
values = [-2.0, -1.5, -1.0, -0.5, -0.3, 0.0, 0.2]
fig, ax = sweep_parameter(
    model,
    "met_logzsol",
    values,
    cmap="viridis",
    label_fmt=r"$\log Z/Z_\odot$ = {:.1f}",
    wave_range=(1000, 12000),
)
ax.set_title(r"Stellar Metallicity: $\log\,Z/Z_\odot$ sweep", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
plt.tight_layout()
plt.savefig("plot_logzsol_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
