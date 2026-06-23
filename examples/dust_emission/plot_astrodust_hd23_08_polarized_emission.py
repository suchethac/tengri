r"""
Astrodust+PAH polarized emission and polarization fraction
==========================================================

Polarized emission and polarization fraction from Astrodust grains at the
Hensley & Draine 2023 fiducial ionization parameter.
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
lambda_P_lambda = np.asarray(tpl.lambda_P_lambda_polarized[i])

c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
factor = c_cgs / (4.0 * np.pi * lam_cm)
li_astrodust = np.asarray(tpl.L_nu_astrodust[i]) * factor

with np.errstate(divide="ignore", invalid="ignore"):
    p_frac = np.where(li_astrodust > 0, lambda_P_lambda / li_astrodust, np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(wave_um, lambda_P_lambda, color="k", lw=1.5)
ax1.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\lambda\,P_\lambda/N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    xlim=(10.0, 1.0e4),
    ylim=(1.0e-32, 1.0e-25),
)

ax2.plot(wave_um, p_frac, color="k", lw=1.5)
ax2.set(
    xscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$P_\lambda / I_\lambda^{\rm Astrodust}$",
    xlim=(50.0, 3.0e3),
    ylim=(0.0, 0.30),
)

fig.tight_layout()
plt.savefig("plot_astrodust_hd23_08_polarized_emission.png", dpi=150, bbox_inches="tight")
