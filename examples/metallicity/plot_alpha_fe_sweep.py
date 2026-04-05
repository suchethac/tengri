"""
α/Fe Enhancement (met_alpha_fe)
=================================

Alpha-element enhancement :math:`[\\alpha/\\rm Fe]` records the chemical
enrichment history: high :math:`[\\alpha/\\rm Fe]` signals rapid enrichment
by core-collapse supernovae before Type Ia SNe can dilute the alpha
elements. In the SED, enhanced alpha suppresses iron absorption features
and alters the optical mass-to-light ratio.

Enable with ``alpha_enhancement=True`` in ``ParamSpec``.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Model, ParamSpec, load_ssp_data
from tengri.plotting import setup_style, sweep_parameter

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

# --- Build model: old passive galaxy where [α/Fe] matters ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(0.5),
    sfh_tsnorm_peak_lbt_gyr=Fixed(8.0),  # peaked early
    sfh_tsnorm_width_gyr=Fixed(1.5),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(10.0),
    met_logzsol=Fixed(0.0),  # solar
    met_alpha_fe=Fixed(0.0),  # will sweep this
    dust_tau_bc=Fixed(0.0),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    alpha_enhancement=True,
)
model = Model(spec, ssp)

# --- Sweep [α/Fe] ---
values = [-0.2, 0.0, 0.2, 0.4, 0.6]
fig, ax = sweep_parameter(
    model,
    "met_alpha_fe",
    values,
    cmap="magma",
    label_fmt=r"$[\alpha/\mathrm{{Fe}}]$ = {:.1f}",
    wave_range=(3500, 9000),
)
ax.set_title(r"$\alpha$-element Enhancement: Impact on Optical Absorption Features", fontsize=12)
ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)")
plt.tight_layout()
plt.savefig("plot_alpha_fe_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
