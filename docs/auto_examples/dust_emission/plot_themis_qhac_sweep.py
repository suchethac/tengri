r"""
THEMIS: q_HAC sweep at fixed U_min
===================================

Sweep hydrocarbon grain content across the THEMIS grid at fixed minimum
radiation field strength. PAH-like mid-IR features strengthen with q_HAC
while FIR continuum remains essentially unchanged.
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
    single_u = np.asarray(f["single_u"][:])

wave_um = wave_aa * 1.0e-4
i_umin = int(np.argmin(np.abs(umin_grid - 1.0)))

c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

fig, ax = plt.subplots(figsize=(7.0, 5.0))
cmap = plt.get_cmap("viridis")
for k, qhac in enumerate(qhac_grid):
    L_nu = single_u[k, i_umin]
    nu_Lnu = nu * L_nu
    ax.plot(
        wave_um,
        nu_Lnu,
        color=cmap(k / max(1, len(qhac_grid) - 1)),
        lw=1.3,
        label=rf"$q_{{\rm HAC}} = {qhac:.2f}$",
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\nu L_\nu\ [\mathrm{arbitrary,\ normalised}]$",
    xlim=(2.0, 1.0e3),
    ylim=(1.0e-26, 1.0e-23),
)
ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=3)
fig.tight_layout()
plt.savefig("plot_themis_qhac_sweep.png", dpi=150, bbox_inches="tight")
