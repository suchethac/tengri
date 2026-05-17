r"""
Astrodust+PAH spinning-dust microwave emission
==============================================

Spinning dust microwave emission — H&D 2023 fiducial.

Reproduces the AME panel from the model_file_tutorial.ipynb showing
the spinning-dust :math:`I_\\nu/N_H` per H atom in the 10-100 GHz
range, with the per-phase decomposition:

* Astrodust grains in the CNM and WNM
* PAHs in the CNM and WNM
* Total at the published :math:`f_{\\rm CNM} = 0.28`

Reference
---------
* Notebook: brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb
* Paper:    Hensley & Draine 2023, ApJ 948, 55 (arXiv:2208.12365)
"""


# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.dust.astrodust_hd23 import load_astrodust_hd23_or_raise

setup_style()


def _find_h5():
    for p in (
        Path("data/astrodust_templates.h5"),
        Path("../data/astrodust_templates.h5"),
        Path("../../data/astrodust_templates.h5"),
    ):
        if p.exists():
            return str(p)
    return None


_PATH = _find_h5()
if _PATH is None:
    raise FileNotFoundError(
        "Astrodust HDF5 not found. Build with "
        "`python scripts/build_astrodust_hdf5.py --download`."
    )

HDF5 = _PATH

tpl = load_astrodust_hd23_or_raise(HDF5)
wave_um = np.asarray(tpl.wavelength_um)
c_cgs = 2.99792458e10
nu_hz = c_cgs / (wave_um * 1.0e-4)

# Convert L_nu (erg/s/Hz/H) -> I_nu/N_H (Jy cm^2 sr^-1 H^-1).
# I_nu = L_nu / (4*pi*r^2), but the file stores per-H so we want
# I_nu/N_H = (L_nu_per_H) * (1 / 4*pi).  Then 1 Jy = 1e-23 erg/s/Hz/cm^2,
# so we multiply by 1e23 to go from cgs to Jy.
factor = 1.0e23 / (4.0 * np.pi)
I_nu_total = factor * np.asarray(tpl.L_nu_spdust_total)
I_nu_Ad_CNM = factor * np.asarray(tpl.L_nu_spdust_Ad_CNM)
I_nu_Ad_WNM = factor * np.asarray(tpl.L_nu_spdust_Ad_WNM)
I_nu_PAH_CNM = factor * np.asarray(tpl.L_nu_spdust_PAH_CNM)
I_nu_PAH_WNM = factor * np.asarray(tpl.L_nu_spdust_PAH_WNM)

fig, ax = plt.subplots(figsize=(7.0, 5.0))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\nu\ [\mathrm{GHz}]$", fontsize=12)
ax.set_ylabel(
    r"$I_\nu/N_{\rm H}\ [\mathrm{Jy\,cm^2\,sr^{-1}\,H^{-1}}]$",
    fontsize=12,
)
ax.set_xlim(10.0, 100.0)
ax.set_ylim(1.0e-19, 2.0e-18)

nu_ghz = nu_hz * 1.0e-9
ax.plot(nu_ghz, I_nu_Ad_CNM, color="#e41a1c", ls="--", label="Astrodust CNM")
ax.plot(nu_ghz, I_nu_Ad_WNM, color="#e41a1c", ls=":", label="Astrodust WNM")
ax.plot(nu_ghz, I_nu_PAH_CNM, color="#0868ac", ls="--", label="PAHs CNM")
ax.plot(nu_ghz, I_nu_PAH_WNM, color="#0868ac", ls=":", label="PAHs WNM")
ax.plot(nu_ghz, I_nu_total, color="k", lw=1.6, label=r"Total ($f_{\rm CNM}=0.28$)", zorder=0)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title("Spinning dust emission (HDU 9)", fontsize=11)
# Match the notebook's plain-decimal x-tick labels.
ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.NullFormatter())
ax.set_xticks([10, 30, 60, 100])
ax.set_xticklabels(["10", "30", "60", "100"])
plt.show()

