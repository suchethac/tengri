"""
SKIRTOR Torus: Opening Angle Sweep
===================================

Sweep `agn_oa_skirtor` from 20° to 60° at fixed inclination, optical
depth, and power. A more open torus exposes more of the disc and shifts
the IR emission peak.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_agn_oa_sweep_001.png
   :alt: plot_agn_oa_sweep
   :class: sphx-glr-single-img

"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Locate SKIRTOR grid file
_grid_path = None
for p in [
    Path("data/skirtor_templates_v3.h5"),
    Path("../data/skirtor_templates_v3.h5"),
    Path("../../data/skirtor_templates_v3.h5"),
    Path("../../../data/skirtor_templates_v3.h5"),
    Path("data/skirtor_templates_v2.h5"),
    Path("../data/skirtor_templates_v2.h5"),
    Path("../../data/skirtor_templates_v2.h5"),
    Path("../../../data/skirtor_templates_v2.h5"),
]:
    if p.exists():
        _grid_path = str(p)
        break

if _grid_path is None:
    raise SystemExit(
        "Skipping: SKIRTOR grid not found. Run: python scripts/download_skirtor_templates.py"
    )

from tengri.agn import create_skirtor_from_grid
from tengri.plot import setup_style

setup_style()

# Load the SKIRTOR interpolator
skirtor_fn = create_skirtor_from_grid(_grid_path)

# Wavelength grid: 0.5 - 500 micron (IR torus dominated)
wavelength = jnp.logspace(np.log10(5e3), np.log10(5e6), 512)
wave_um = np.array(wavelength) / 1e4

# Opening angle values to sweep (degrees)
oa_values = [20, 30, 40, 50, 60]

# Create figure with single panel
fig, ax = plt.subplots(figsize=(8, 5))

# Generate colors from colormap
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(oa_values)))

# Fixed parameters: typical values
cos_inc = 0.5  # Edge-on orientation (60 degrees)
agn_log_lbol = 11.0
agn_tau_skirtor = 7.0
agn_p_skirtor = 1.0  # Radial power
agn_q_skirtor = 1.0

# Sweep opening angle
for oa, color in zip(oa_values, colors):
    try:
        sed = skirtor_fn(
            wavelength,
            agn_log_lbol=agn_log_lbol,
            agn_tau_skirtor=agn_tau_skirtor,
            agn_p_skirtor=agn_p_skirtor,
            agn_q_skirtor=agn_q_skirtor,
            agn_oa_skirtor=float(oa),
            agn_cos_inc=cos_inc,
        )
        label = rf"$\theta_{{\mathrm{{oa}}}} = {oa}°$"
        ax.loglog(wave_um, np.array(sed), lw=2.0, color=color, label=label)
    except Exception:
        continue

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("SKIRTOR torus: opening-angle sweep", fontsize=12)
ax.legend(fontsize=10, frameon=False, loc="best")
ax.set_xlim(0.5, 500)
ax.set_ylim(1e21, 1e32)
ax.grid(False)

fig.tight_layout()
plt.savefig("plot_agn_oa_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
