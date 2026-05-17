r"""
Astrodust+PAH ionization fraction and alignment
===============================================

f_ion(a) and f_align(a) versus grain size — H&D 2023 fiducials.

Reproduces the two diagnostic panels from the model_file_tutorial.ipynb
showing the ionization fraction and alignment efficiency that were
adopted in the published fiducial size distribution.

These functions live in HDU 1 columns 4 (f_ion) and 5 (f_align).
"""


# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HDF5 = Path("data/astrodust_templates.h5")

with h5py.File(HDF5, "r") as f:
    size_dist = np.asarray(f["size_distribution"])  # (167, 5)

rad_um = size_dist[:, 0]
f_ion = size_dist[:, 3]
f_align = size_dist[:, 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for ax, y, lab in (
    (ax1, f_ion, r"$f_{\rm ion}$"),
    (ax2, f_align, r"$f_{\rm align}$"),
):
    ax.set_xscale("log")
    ax.set_xlabel(r"$a\ [\mu\mathrm{m}]$", fontsize=12)
    ax.set_ylabel(lab, fontsize=14)
    ax.set_xlim(3.0e-4, 5.0)
    ax.set_ylim(0.0, 1.05)
    ax.plot(rad_um, y, lw=2, color="#1f77b4")

ax1.set_title("PAH ionization fraction (Eq. 20)", fontsize=10)
ax2.set_title("Astrodust alignment efficiency", fontsize=10)
plt.show()

