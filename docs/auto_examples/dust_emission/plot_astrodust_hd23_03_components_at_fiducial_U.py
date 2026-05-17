r"""
Astrodust+PAH per-component decomposition
=========================================

Astrodust vs PAH components at the fiducial U=1.6 (lgU=0.2).

Plots the per-component decomposition (Astrodust continuum + PAHs +
spinning-dust microwave emission) at the H&D 2023 fiducial value
:math:`\\log_{10} U = 0.2`, matching the published tutorial.

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
lgU = np.asarray(tpl.lgU)

i = int(np.argmin(np.abs(lgU - 0.2)))  # fiducial slice

# Convert L_nu (erg/s/Hz/H) -> lambda*I_lambda (erg/s/sr/H).
c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
factor = c_cgs / (4.0 * np.pi * lam_cm)
li_total = np.asarray(tpl.L_nu_total[i]) * factor
li_astro = np.asarray(tpl.L_nu_astrodust[i]) * factor
li_pah = np.asarray(tpl.L_nu_pah[i]) * factor
# Spinning dust at f_CNM=0.28 (the published default).
spd = 0.28 * (
    np.asarray(tpl.L_nu_spdust_Ad_CNM) + np.asarray(tpl.L_nu_spdust_PAH_CNM)
) + 0.72 * (np.asarray(tpl.L_nu_spdust_Ad_WNM) + np.asarray(tpl.L_nu_spdust_PAH_WNM))
li_spd = spd * factor

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\lambda I_\lambda / N_{\rm H}\ "
    r"[\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    fontsize=11,
)
# Spinning-dust λI_λ peaks ~4e-31 at ~1 cm (HD23 §6), 5-6 decades
# below the FIR thermal peak.  Extend y down to 1e-32 so the
# spinning-dust bump is visible without losing the thermal anchor.
ax.set_xlim(5.0, 3.0e4)
ax.set_ylim(1.0e-32, 1.0e-24)

ax.plot(wave_um, li_astro, color="#e41a1c", ls="--", lw=1.5, label="Astrodust")
ax.plot(wave_um, li_pah, color="#0868ac", ls="--", lw=1.5, label="PAHs")
# Spinning dust is identically zero at short λ; mask the leading
# zeros so the log-axis trace doesn't drop to -inf.
mask = li_spd > 1.0e-32
ax.plot(
    wave_um[mask],
    li_spd[mask],
    color="#984ea3",
    ls=":",
    lw=1.5,
    label=r"Spinning dust ($f_{\rm CNM}=0.28$)",
)
ax.plot(wave_um, li_total, color="k", lw=2, label="Total thermal", zorder=0)
ax.legend(loc="lower left", frameon=False, fontsize=10)
ax.set_title(
    rf"H&D 2023 fiducial: $\log_{{10}} U={lgU[i]:.1f}$",
    fontsize=12,
)
plt.show()

