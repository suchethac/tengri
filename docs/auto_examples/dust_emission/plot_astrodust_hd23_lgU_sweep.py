r"""
Hensley & Draine 2023 Astrodust+PAH: log U sweep
=================================================

Sweep ``log10 U`` over the published ``[-3, +6]`` range of the
Hensley & Draine 2023 Astrodust+PAH grid (91 lgU points, finer
than Draine+2021 PAHspec's 15-point grid).  Shows the FIR peak
shifting blueward and the MIR PAH features rising as the
radiation field intensifies.

Reference
---------
Hensley, B.S. & Draine, B.T. 2023, ApJ 948, 55, arXiv:2208.12365.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

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

tpl = load_astrodust_hd23_or_raise(_PATH)
wave_um = np.asarray(tpl.wavelength_um)
lgU = np.asarray(tpl.lgU)
L_nu_total = np.asarray(tpl.L_nu_total)

c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
li_um = L_nu_total * c_cgs / (4.0 * np.pi * lam_cm[None, :])  # lambda*I_lambda/N_H

fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\lambda I_\lambda / N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    fontsize=11,
)
ax.set_xlim(1.0, 1.0e3)
# Full published range: lgU ∈ [-3, +6] step 1.0 sampled from the
# 91-point underlying grid.  Spectrum scales ~linearly with U over
# 9 decades; the y-axis dynamic range reflects that span.
ax.set_ylim(1.0e-29, 1.0e-18)

cmap = plt.get_cmap("viridis")
targets = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
for k, tg in enumerate(targets):
    i = int(np.argmin(np.abs(lgU - tg)))
    ax.plot(
        wave_um, li_um[i],
        color=cmap(k / max(1, len(targets) - 1)),
        lw=1.4,
        label=rf"$\log_{{10}} U={lgU[i]:+.1f}$",
    )

ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
ax.set_title(
    "Hensley & Draine 2023 Astrodust+PAH — log U sweep",
    fontsize=11,
)

plt.show()
