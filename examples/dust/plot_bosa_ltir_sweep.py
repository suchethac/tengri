r"""
BOSA: log L_TIR sweep at fixed log sSFR
==========================================

Sweep ``log10 L_TIR`` over the full published 41-point grid
(8.5 to 12.5 dex in 0.1 dex steps) at fixed
``log10 sSFR = -9.6`` (typical star-forming galaxy).
Increasing L_TIR makes the dust hotter → FIR peak shifts blueward
and PAH features become more prominent relative to the FIR
continuum.

Reference
---------
Boquien M. & Salim S. 2021, A&A 653 A149, arXiv:2106.04595.
Library: https://salims.pages.iu.edu/bosa/.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


def _find_h5():
    for p in (
        Path("data/bosa_templates.h5"),
        Path("../data/bosa_templates.h5"),
        Path("../../data/bosa_templates.h5"),
    ):
        if p.exists():
            return str(p)
    return None


_PATH = _find_h5()
if _PATH is None:
    raise FileNotFoundError(
        "BOSA HDF5 not found. Build with "
        "`python scripts/build_bosa_hdf5.py --download`."
    )

with h5py.File(_PATH, "r") as f:
    wave_aa = np.asarray(f["wavelength_aa"][:])
    log_ltir = np.asarray(f["log_ltir_grid"][:])
    log_ssfr = np.asarray(f["log_ssfr_grid"][:])
    spectra = np.asarray(f["spectra"][:])

wave_um = wave_aa * 1.0e-4
i_ssfr = int(np.argmin(np.abs(log_ssfr - (-9.6))))
c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\nu L_\nu\ [\mathrm{normalised}\ \int L_\nu d\nu = 1]$",
    fontsize=11,
)
ax.set_xlim(3.0, 1.0e3)
ax.set_ylim(1.0e-3, 2.0e0)

cmap = plt.get_cmap("plasma")
# Subsample ~12 of the 41 LTIR points for a clean legend.
idx_show = np.linspace(0, len(log_ltir) - 1, 12).astype(int)
for k, il in enumerate(idx_show):
    L_nu = spectra[il, i_ssfr]
    ax.plot(
        wave_um, nu * L_nu,
        color=cmap(k / max(1, len(idx_show) - 1)),
        lw=1.3,
        label=rf"$\log_{{10}} L_{{\rm TIR}} = {log_ltir[il]:.1f}$",
    )

ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
ax.set_title(
    rf"BOSA at $\log_{{10}} \mathrm{{sSFR}}={log_ssfr[i_ssfr]:.1f}$",
    fontsize=11,
)

plt.show()
