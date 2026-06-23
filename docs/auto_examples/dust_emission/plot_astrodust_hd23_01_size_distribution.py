r"""
Astrodust+PAH size distribution
================================

Per-H grain volume distribution versus grain radius for the Hensley & Draine
2023 fiducial size distribution (MW high-latitude :math:`R_V=3.1` sightline).
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

size_dist = np.asarray(tengri.load_astrodust_hd23().size_distribution)

rad_um = size_dist[:, 0]
dn_Ad_per_H = size_dist[:, 1]
dn_PAH_per_H = size_dist[:, 2]
rad_cm = rad_um * 1.0e-4

dlna = np.log(rad_um[20] / rad_um[0]) / 20.0
vol_Ad = (4.0 / 3.0) * np.pi * rad_cm**3 * dn_Ad_per_H / dlna
vol_PAH = (4.0 / 3.0) * np.pi * rad_cm**3 * dn_PAH_per_H / dlna

fig, ax = plt.subplots(figsize=(7.0, 5.0))
mask = dn_Ad_per_H > 0
ax.plot(rad_um[mask], vol_Ad[mask], lw=2, color="#e41a1c", label="Astrodust")
mask = dn_PAH_per_H > 0
ax.plot(rad_um[mask], vol_PAH[mask], lw=2, color="#0868ac", label="PAHs")
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$a\ [\mu\mathrm{m}]$",
    ylabel=r"$(4\pi/3)\,a^3\,dn/d\ln a / n_{\rm H}\ [\mathrm{cm}^3\,\mathrm{H}^{-1}]$",
    xlim=(3.0e-4, 1.0),
    ylim=(1.0e-30, 1.0e-26),
)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
plt.savefig("plot_astrodust_hd23_01_size_distribution.png", dpi=150, bbox_inches="tight")
