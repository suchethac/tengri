r"""
THEMIS: q_HAC sweep at fixed U_min
====================================

Sweep ``q_HAC`` (mass fraction of small a-C(:H) hydrocarbon grains
< 1.5 nm) across the canonical CIGALE-distributed Jones+2017
THEMIS grid at fixed ``U_min = 1`` and ``alpha = 2``.  The PAH-like
mid-IR features at 3.3, 6.2, 7.7, 8.6, 11.3 μm strengthen with
``q_HAC``; the FIR continuum is essentially unchanged.

Reference
---------
Jones, A.P., Köhler, M., Ysard, N., et al. 2017, A&A 602, A46
(arXiv:1703.00775).  Templates via CIGALE
(https://gitlab.lam.fr/cigale/cigale).
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri.analysis.plotting import setup_style

setup_style()


def _find_h5():
    for p in (
        Path("data/themis_templates.h5"),
        Path("../data/themis_templates.h5"),
        Path("../../data/themis_templates.h5"),
    ):
        if p.exists():
            return str(p)
    return None


_PATH = _find_h5()
if _PATH is None:
    raise FileNotFoundError(
        "THEMIS HDF5 not found. Build with "
        "`python scripts/build_themis_hdf5.py --clone`."
    )

with h5py.File(_PATH, "r") as f:
    wave_aa = np.asarray(f["wavelength_aa"][:])
    qhac_grid = np.asarray(f["qhac_grid"][:])
    umin_grid = np.asarray(f["umin_grid"][:])
    single_u = np.asarray(f["single_u"][:])  # (n_qhac, n_umin, n_wave)

wave_um = wave_aa * 1.0e-4
i_umin = int(np.argmin(np.abs(umin_grid - 1.0)))

fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\nu L_\nu\ [\mathrm{arbitrary,\ normalised}]$",
    fontsize=11,
)
ax.set_xlim(2.0, 1.0e3)

c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

cmap = plt.get_cmap("viridis")
for k, qhac in enumerate(qhac_grid):
    L_nu = single_u[k, i_umin]      # already normalised to int L_nu d nu = 1
    nu_Lnu = nu * L_nu
    ax.plot(
        wave_um, nu_Lnu,
        color=cmap(k / max(1, len(qhac_grid) - 1)),
        lw=1.3,
        label=rf"$q_{{\rm HAC}} = {qhac:.2f}$",
    )

ax.set_ylim(1.0e-26, 1.0e-23)
ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=3)
ax.set_title(
    rf"THEMIS (Jones+2017) at $U_{{\rm min}}={umin_grid[i_umin]:.2f}$, $\alpha=2$",
    fontsize=11,
)

plt.show()
