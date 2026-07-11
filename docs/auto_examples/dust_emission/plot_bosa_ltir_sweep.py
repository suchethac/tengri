r"""
BOSA: log L_TIR sweep at fixed log sSFR
========================================

Sweep infrared luminosity across the BOSA grid at fixed specific star
formation rate. Increasing L_TIR heats dust, shifting FIR peak blueward
and enhancing PAH relative to continuum. Library is normalized by
∫Lν dν=1; shape variation with L_TIR is intentionally small.
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

with h5py.File(data_path("bosa_templates.h5"), "r") as f:
    wave_aa = np.asarray(f["wavelength_aa"][:])
    log_ltir = np.asarray(f["log_ltir_grid"][:])
    log_ssfr = np.asarray(f["log_ssfr_grid"][:])
    spectra = np.asarray(f["spectra"][:])

wave_um = wave_aa * 1.0e-4
i_ssfr = int(np.argmin(np.abs(log_ssfr - (-9.6))))
c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

fig, ax = plt.subplots(figsize=(8.0, 5.5))
cmap = plt.get_cmap("viridis")
idx_show = np.linspace(0, len(log_ltir) - 1, 6).astype(int)
for k, il in enumerate(idx_show):
    L_nu = spectra[il, i_ssfr]
    ax.plot(
        wave_um,
        nu * L_nu,
        color=cmap(k / max(1, len(idx_show) - 1)),
        lw=1.3,
        label=rf"$\log_{{10}} L_{{\rm TIR}} = {log_ltir[il]:.1f}$",
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\nu L_\nu\ [\mathrm{normalized}\ \int L_\nu d\nu = 1]$",
    xlim=(50.0, 200.0),
    ylim=(1.0e-3, 2.0e0),
)
ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=1)
fig.tight_layout()
plt.savefig("plot_bosa_ltir_sweep.png", dpi=150, bbox_inches="tight")
