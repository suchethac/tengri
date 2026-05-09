"""Reproduce Figure 1 of model_file_tutorial.ipynb (Hensley & Draine 2023).

Plots the per-H grain volume distribution
:math:`(4\\pi/3)a^3 \\, dn/d\\ln a / n_{\\rm H}` versus radius for
astrodust and PAHs, using the size-distribution metadata embedded
in tengri's HDF5 grid.

Reference
---------
* Notebook: brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb
* Paper:    Hensley & Draine 2023, ApJ 948, 55 (arXiv:2208.12365).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HDF5 = Path("data/astrodust_templates.h5")
OUT = Path("examples/astrodust_hd23/fig_size_distribution.png")


def main() -> None:
    with h5py.File(HDF5, "r") as f:
        size_dist = np.asarray(f["size_distribution"])  # (167, 5)

    rad_um = size_dist[:, 0]  # μm
    dn_Ad_per_H = size_dist[:, 1]  # number / H per bin
    dn_PAH_per_H = size_dist[:, 2]
    rad_cm = rad_um * 1.0e-4

    # The bins are log-spaced; recover dlna from the first 20 entries.
    dlna = np.log(rad_um[20] / rad_um[0]) / 20.0
    # Volume per H per d ln a, in cm^3/H.
    vol_Ad = (4.0 / 3.0) * np.pi * rad_cm**3 * dn_Ad_per_H / dlna
    vol_PAH = (4.0 / 3.0) * np.pi * rad_cm**3 * dn_PAH_per_H / dlna

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a\ [\mu\mathrm{m}]$", fontsize=18)
    ax.set_ylabel(
        r"$(4\pi/3)\,a^3\,dn/d\ln a / n_{\rm H}\ "
        r"[\mathrm{cm}^3\,\mathrm{H}^{-1}]$",
        fontsize=14,
    )
    ax.set_xlim(3.0e-4, 1.0)
    ax.set_ylim(1.0e-30, 1.0e-26)

    mask = dn_Ad_per_H > 0
    ax.plot(rad_um[mask], vol_Ad[mask], lw=2, color="#e41a1c", label="Astrodust")
    mask = dn_PAH_per_H > 0
    ax.plot(rad_um[mask], vol_PAH[mask], lw=2, color="#0868ac", label="PAHs")
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    ax.set_title("Hensley & Draine 2023 fiducial size distribution", fontsize=12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
