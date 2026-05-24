r"""
Astrodust+PAH ionization fraction and alignment
===============================================

Ionization fraction and alignment efficiency versus grain size for the
Hensley & Draine 2023 fiducial size distribution.
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

with h5py.File(data_path("astrodust_templates.h5"), "r") as f:
    size_dist = np.asarray(f["size_distribution"])

rad_um = size_dist[:, 0]
f_ion = size_dist[:, 3]
f_align = size_dist[:, 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for ax, y, lab in (
    (ax1, f_ion, r"$f_{\rm ion}$"),
    (ax2, f_align, r"$f_{\rm align}$"),
):
    ax.plot(rad_um, y, lw=2, color="#1f77b4")
    ax.set(
        xscale="log",
        xlabel=r"$a\ [\mu\mathrm{m}]$",
        ylabel=lab,
        xlim=(3.0e-4, 5.0),
        ylim=(0.0, 1.05),
    )

ax1.set_title("PAH ionization fraction (Eq. 20)", fontsize=10)
fig.tight_layout()
plt.savefig("plot_astrodust_hd23_05_ionization_alignment.png", dpi=150, bbox_inches="tight")
