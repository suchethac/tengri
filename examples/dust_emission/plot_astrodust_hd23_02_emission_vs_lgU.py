r"""
Astrodust+PAH emission vs log U
================================

Plots :math:`\lambda I_\lambda / N_{\rm H} / U` for several
:math:`\log_{10} U` values from the Hensley & Draine 2023 grid.
Dividing by :math:`U` makes the U-dependence of the PAH-vs-FIR
ratio visible: low-U curves stack atop each other in the FIR while
the MIR features rise steeply with :math:`U`.

Reproduces tutorial Figure 8 from
``brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb``.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tengri.components.dust.astrodust_hd23 import load_astrodust_hd23_or_raise


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

tpl = load_astrodust_hd23_or_raise(_PATH)
wave_um = np.asarray(tpl.wavelength_um)
lgU = np.asarray(tpl.lgU)
L_nu_total = np.asarray(tpl.L_nu_total)

c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
lam_I_lam = L_nu_total * c_cgs / (4.0 * np.pi * lam_cm[None, :])

fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\lambda I_\lambda / (N_{\rm H}\,U)\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    fontsize=11,
)
ax.set_xlim(2.0, 1000.0)
ax.set_ylim(1.0e-28, 5.0e-25)

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
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Astrodust+PAH emission per H per U", fontsize=11)

plt.show()
