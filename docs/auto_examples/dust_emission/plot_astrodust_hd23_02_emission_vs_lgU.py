r"""
Astrodust+PAH emission vs log U
================================

Emission per H per ionization parameter U across the Hensley & Draine 2023
grid. Dividing by U reveals its effect: PAH-to-FIR ratio plateaus in FIR
(U-independent) but rises steeply with U in MIR.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

tpl = tengri.load_astrodust_hd23()
wave_um = np.asarray(tpl.wavelength_um)
lgU = np.asarray(tpl.lgU)
L_nu_total = np.asarray(tpl.L_nu_total)

c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
lam_I_lam = L_nu_total * c_cgs / (4.0 * np.pi * lam_cm[None, :])

fig, ax = plt.subplots(figsize=(7.0, 5.0))
cmap = plt.get_cmap("viridis")
targets = np.arange(-3.0, 6.0, 1.15)
for k, tg in enumerate(targets):
    i = int(np.argmin(np.abs(lgU - tg)))
    U = 10.0 ** lgU[i]
    ax.plot(
        wave_um,
        lam_I_lam[i] / U,
        color=cmap(k / max(1, len(targets) - 1)),
        lw=1.4,
        label=rf"$\log_{{10}} U={lgU[i]:+.2f}$",
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\lambda I_\lambda / (N_{\rm H}\,U)\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    xlim=(2.0, 1000.0),
    ylim=(1.0e-28, 5.0e-25),
)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
plt.savefig("plot_astrodust_hd23_02_emission_vs_lgU.png", dpi=150, bbox_inches="tight")
