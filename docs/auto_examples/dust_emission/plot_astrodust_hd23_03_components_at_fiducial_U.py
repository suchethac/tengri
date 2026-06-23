r"""
Astrodust+PAH per-component decomposition
=========================================

Per-component breakdown (Astrodust continuum, PAHs, spinning dust) at the
Hensley & Draine 2023 fiducial ionization parameter :math:`\log_{10} U = 0.2`.
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

i = int(np.argmin(np.abs(lgU - 0.2)))

c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
factor = c_cgs / (4.0 * np.pi * lam_cm)
li_total = np.asarray(tpl.L_nu_total[i]) * factor
li_astro = np.asarray(tpl.L_nu_astrodust[i]) * factor
li_pah = np.asarray(tpl.L_nu_pah[i]) * factor
spd = 0.28 * (np.asarray(tpl.L_nu_spdust_Ad_CNM) + np.asarray(tpl.L_nu_spdust_PAH_CNM)) + 0.72 * (
    np.asarray(tpl.L_nu_spdust_Ad_WNM) + np.asarray(tpl.L_nu_spdust_PAH_WNM)
)
li_spd = spd * factor

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.plot(wave_um, li_astro, color="#e41a1c", ls="--", lw=1.5, label="Astrodust")
ax.plot(wave_um, li_pah, color="#0868ac", ls="--", lw=1.5, label="PAHs")
mask = li_spd > 1.0e-32
ax.plot(
    wave_um[mask],
    li_spd[mask],
    color="#984ea3",
    ls=":",
    lw=1.5,
    label=r"Spinning dust ($f_{\rm CNM}=0.28$)",
)
ax.plot(wave_um, li_total, color="k", lw=2, label="Total thermal", zorder=0)
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\lambda I_\lambda / N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    xlim=(5.0, 3.0e4),
    ylim=(1.0e-32, 1.0e-24),
)
ax.legend(loc="lower left", frameon=False, fontsize=10)
fig.tight_layout()
plt.savefig("plot_astrodust_hd23_03_components_at_fiducial_U.png", dpi=150, bbox_inches="tight")
