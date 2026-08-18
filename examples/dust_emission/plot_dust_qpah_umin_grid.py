"""
The q_PAH and U_min knobs move PAH amplitude and FIR peak independently
========================================================================

A 2-D grid on the Draine & Li 2007 template library: rows step through
PAH mass fraction q_PAH (controls mid-IR PAH-feature strength),
columns through the minimum radiation field U_min (sets the diffuse
dust temperature, i.e. the FIR peak position). The two axes act
nearly orthogonally — a surprise for anyone who would lump them
together as "PAH knobs."

Reference: Draine & Li 2007, ApJ, 657, 810.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.dust import draine_li2007
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

wave_aa = jnp.logspace(np.log10(1e4), np.log10(1e7), 2000)
wave_um = np.array(wave_aa) * 1e-4

L_ABS = 1e10 * 3.828e33
qpah_values = [0.5, 2.5, 4.5]
umin_values = [0.5, 2.0, 10.0]

colors_grid = plt.cm.viridis(np.linspace(0.0, 0.85, 3))

fig, axes = plt.subplots(3, 3, figsize=(15, 13))

for i, umin in enumerate(umin_values):
    for j, qpah in enumerate(qpah_values):
        ax = axes[i, j]

        try:
            lnu = draine_li2007(wave_aa, L_ABS, dust_umin=umin, dust_gamma_dl=0.01, dust_qpah=qpah)
        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                "Data not found\n(use synthetic)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set(
                xlabel=r"Wavelength [$\mu$m]",
                ylabel=r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]",
            )
            continue

        y = np.array(lnu)
        mask = (wave_um > 1) & (y > 0)
        ax.loglog(wave_um[mask], y[mask], color=colors_grid[j], lw=2.0)

        ax.set(
            xlabel=r"Wavelength [$\mu$m]",
            ylabel=r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]",
            xlim=(1, 1000),
            ylim=(1e29, 1e32),
        )
        ax.tick_params(labelsize=11)
        if i == 0 and j == 0:
            for wl_um, _wl_label in [(3, "PAH"), (25, "mid-IR"), (100, "far-IR")]:
                ax.axvline(wl_um, color="gray", ls=":", lw=0.5, alpha=0.4)

fig.tight_layout()
plt.savefig("plot_dust_qpah_umin_grid.png", dpi=150, bbox_inches="tight")
