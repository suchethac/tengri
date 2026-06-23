r"""
Astrodust+PAH extinction, scattering, and albedo
================================================

Extinction opacity, polarized extinction, and single-scattering albedo for
the Hensley & Draine 2023 fiducial size distribution.
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

_tpl = tengri.load_astrodust_hd23()
ext = np.asarray(_tpl.tau_per_H)
scatt = np.asarray(_tpl.sigma_sca_per_H)
extpol = np.asarray(_tpl.p_pol_per_H)

wave_um = ext[:, 0]
tau_Ad = ext[:, 1]
tau_PAH = ext[:, 2]
tau_total = ext[:, 3]
sca_Ad = scatt[:, 1]
sca_PAH = scatt[:, 2]
pol_Ad_max = extpol[:, 1]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

ax1.plot(wave_um, tau_Ad, color="#e41a1c", ls="--", label="Astrodust")
ax1.plot(wave_um, tau_PAH, color="#0868ac", ls="--", label="PAHs")
ax1.plot(wave_um, tau_total, color="k", lw=1.5, label="Total", zorder=0)
ax1.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\tau_\lambda/N_{\rm H}\ [\mathrm{cm}^2\,\mathrm{H}^{-1}]$",
    xlim=(0.1, 40.0),
    ylim=(5.0e-25, 3.0e-21),
)
ax1.legend(loc="upper right", frameon=False, fontsize=9)

ax2.plot(wave_um, pol_Ad_max, color="k", lw=1.5)
ax2.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$(p_\lambda/N_{\rm H})^{\rm max}\ [\mathrm{cm}^2\,\mathrm{H}^{-1}]$",
    xlim=(0.1, 40.0),
    ylim=(5.0e-25, 3.0e-23),
)

with np.errstate(invalid="ignore", divide="ignore"):
    albedo_Ad = np.where(tau_Ad > 0, sca_Ad / tau_Ad, np.nan)
    albedo_PAH = np.where(tau_PAH > 0, sca_PAH / tau_PAH, np.nan)
    sca_total = scatt[:, 3]
    albedo_total = np.where(tau_total > 0, sca_total / tau_total, np.nan)
inv_lam = 1.0 / wave_um
ax3.plot(inv_lam, albedo_Ad, color="#e41a1c", lw=1.5, label="Astrodust")
ax3.plot(inv_lam, albedo_PAH, color="#0868ac", lw=1.5, label="PAHs")
ax3.plot(inv_lam, albedo_total, color="k", lw=1.2, label="Total", zorder=0)
ax3.set(
    xlabel=r"$\lambda^{-1}\ [\mu\mathrm{m}^{-1}]$",
    ylabel="Albedo  $\\omega$",
    xlim=(0.0, 8.0),
    ylim=(0.0, 1.0),
)
ax3.legend(loc="upper left", frameon=False, fontsize=9)

fig.tight_layout()
plt.savefig("plot_astrodust_hd23_06_extinction_and_scattering.png", dpi=150, bbox_inches="tight")
