r"""
THEMIS: U_min sweep at fixed q_HAC
===================================

Sweep minimum radiation field strength across the THEMIS grid at fixed
hydrocarbon grain content. Higher U warms dust, shifting FIR peak
blueward and strengthening mid-IR grain emission relative to far-IR.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_themis_umin_sweep_001.png
   :alt: plot_themis_umin_sweep
   :class: sphx-glr-single-img

"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tengri import data_path
from tengri.analysis.plotting import setup_style

setup_style()

with h5py.File(data_path("themis_templates.h5"), "r") as f:
    wave_aa = np.asarray(f["wavelength_aa"][:])
    qhac_grid = np.asarray(f["qhac_grid"][:])
    umin_grid = np.asarray(f["umin_grid"][:])
    single_u = np.asarray(f["single_u"][:])

wave_um = wave_aa * 1.0e-4
i_qhac = int(np.argmin(np.abs(qhac_grid - 0.17)))
c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

fig, ax = plt.subplots(figsize=(8.0, 5.5))
cmap = plt.get_cmap("plasma")
idx_show = np.linspace(0, len(umin_grid) - 1, 12).astype(int)
for k, iu in enumerate(idx_show):
    L_nu = single_u[i_qhac, iu]
    ax.plot(
        wave_um,
        nu * L_nu,
        color=cmap(k / max(1, len(idx_show) - 1)),
        lw=1.3,
        label=rf"$U_{{\rm min}}={umin_grid[iu]:.2f}$",
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\nu L_\nu\ [\mathrm{normalised}\ \int L_\nu d\nu = 1]$",
    xlim=(2.0, 1.0e3),
    ylim=(1.0e-26, 1.0e-22),
    title=rf"THEMIS (Jones+2017) at $q_{{\rm HAC}}={qhac_grid[i_qhac]:.2f}$, $\alpha=2$",
)
ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
plt.savefig("plot_themis_umin_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
