r"""
Astrodust+PAH polarized emission and polarization fraction
==========================================================

Polarized emission and polarization fraction — H&D 2023.

Reproduces a panel showing the polarized emission
:math:`\\lambda P_\\lambda/N_H` from Astrodust (HDU 8) at the
fiducial :math:`\\log_{10} U=0.2`, plus the resulting polarization
fraction :math:`P_\\lambda / I_\\lambda^{\\rm Astrodust}`.

Reference
---------
* Notebook: brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb
"""


# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.components.dust.astrodust_hd23 import load_astrodust_hd23_or_raise

setup_style()


def _find(name):
    """Locate HDF5 file from project root or docs/ (sphinx-gallery) cwd."""
    for p in (
        Path(f"data/{name}"),
        Path(f"../data/{name}"),
        Path(f"../../data/{name}"),
    ):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{name} not found in data/ hierarchy")


HDF5 = _find("astrodust_templates.h5")

tpl = load_astrodust_hd23_or_raise(HDF5)
wave_um = np.asarray(tpl.wavelength_um)
lgU = np.asarray(tpl.lgU)

i = int(np.argmin(np.abs(lgU - 0.2)))
lambda_P_lambda = np.asarray(tpl.lambda_P_lambda_polarized[i])  # erg/s/sr/H

c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
factor = c_cgs / (4.0 * np.pi * lam_cm)
li_astrodust = np.asarray(tpl.L_nu_astrodust[i]) * factor  # erg/s/sr/H

# Polarization fraction is P / I.  Mask near zero intensity.
with np.errstate(divide="ignore", invalid="ignore"):
    p_frac = np.where(li_astrodust > 0, lambda_P_lambda / li_astrodust, np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=11)
ax1.set_ylabel(
    r"$\lambda\,P_\lambda/N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    fontsize=10,
)
# Notebook fig 8 (cell 22) ranges: 10-10000 μm × 1e-32 to 1e-25.
ax1.set_xlim(10.0, 1.0e4)
ax1.set_ylim(1.0e-32, 1.0e-25)
ax1.plot(wave_um, lambda_P_lambda, color="k", lw=1.5)
ax1.set_title(rf"Polarized emission at $\log_{{10}}U={lgU[i]:.1f}$", fontsize=11)

ax2.set_xscale("log")
ax2.set_yscale("linear")
ax2.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=11)
ax2.set_ylabel(r"$P_\lambda / I_\lambda^{\rm Astrodust}$", fontsize=11)
ax2.set_xlim(50.0, 3.0e3)
ax2.set_ylim(0.0, 0.30)
ax2.plot(wave_um, p_frac, color="k", lw=1.5)
ax2.set_title("Astrodust polarization fraction", fontsize=11)
plt.show()

