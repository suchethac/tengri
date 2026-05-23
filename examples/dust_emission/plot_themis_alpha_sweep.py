r"""
THEMIS: power-law slope alpha sweep
====================================

Sweep radiation-field distribution slope across the THEMIS grid at fixed
grain content and minimum intensity. Lower alpha shifts weight toward high U,
warming dust and shifting FIR peak blueward; higher alpha approaches single-U.
"""

import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tengri import data_path
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

with h5py.File(data_path("themis_templates.h5"), "r") as f:
    wave_aa = np.asarray(f["wavelength_aa"][:])
    qhac_grid = np.asarray(f["qhac_grid"][:])
    umin_grid = np.asarray(f["umin_grid"][:])
    alpha_grid = np.asarray(f["alpha_grid"][:])
    powerlaw_alpha = np.asarray(f["powerlaw_alpha"][:])

wave_um = wave_aa * 1.0e-4
i_qhac = int(np.argmin(np.abs(qhac_grid - 0.17)))
i_umin = int(np.argmin(np.abs(umin_grid - 1.0)))
c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

fig, ax = plt.subplots(figsize=(8.0, 5.5))
cmap = plt.get_cmap("plasma")
idx_show = np.arange(0, len(alpha_grid), 2)
for k, ia in enumerate(idx_show):
    L_nu = powerlaw_alpha[i_qhac, i_umin, ia]
    ax.plot(
        wave_um,
        nu * L_nu,
        color=cmap(k / max(1, len(idx_show) - 1)),
        lw=1.3,
        label=rf"$\alpha={alpha_grid[ia]:.1f}$",
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\nu L_\nu\ [\mathrm{normalised}\ \int L_\nu d\nu = 1]$",
    xlim=(2.0, 1.0e3),
    ylim=(1.0e-26, 1.0e-22),
)
ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig("plot_themis_alpha_sweep.png", dpi=150, bbox_inches="tight")
