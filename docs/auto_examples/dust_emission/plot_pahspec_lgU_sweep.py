r"""
Draine+2021 PAHspec: log U sweep at fixed (starlight, ion, size)
================================================================

Sweep the starlight intensity ``log10 U`` over the published
``[0, 7]`` range of the Draine, Li, Hensley et al. 2021 PAHspec
library at a fixed (mMMP starlight, standard ionization, standard
size distribution) configuration.

At low U the spectrum is in the FIR-cooling regime; at high U the
peak shifts blueward into the mid-IR and the PAH features rise as
a fraction of the IR power.

Reference
---------
Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K., Sandstrom, K.,
Smith, J.-D.T. 2021, ApJ 917, 3, arXiv:2011.07046.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri.analysis.plotting import setup_style
from tengri.components.dust.draine2021_pah import (
    load_pahspec_or_raise,
)

setup_style()


def _find_pahspec_h5():
    for p in (
        Path("data/pahspec_draine2021.h5"),
        Path("../data/pahspec_draine2021.h5"),
        Path("../../data/pahspec_draine2021.h5"),
    ):
        if p.exists():
            return str(p)
    return None


_PATH = _find_pahspec_h5()
if _PATH is None:
    raise FileNotFoundError(
        "PAHspec HDF5 not found.  Build with "
        "`python scripts/build_pahspec_hdf5.py --output "
        "data/pahspec_draine2021.h5 --download`"
    )

from tengri.components.dust.draine2021_pah import (
    select_pahspec_axes,
)

tpl = load_pahspec_or_raise(_PATH)
nu_pnu = select_pahspec_axes(
    tpl,
    starlight="mMMP",
    ionization="st",
    size_distribution="std",
    slab=False,
)  # (n_lgU=15, n_wave_um)
wave_um = np.asarray(tpl.wavelength_um)
lgU_grid = np.asarray(tpl.lgU)

# nu*P_nu -> lambda*I_lambda (4*pi sr) for a familiar y-axis scale.
c_cgs = 2.99792458e10
lam_cm = wave_um * 1.0e-4
nu_hz = c_cgs / lam_cm
li_um = np.asarray(nu_pnu) / (4.0 * np.pi)  # erg/s/sr/H

fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\lambda I_\lambda / N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    fontsize=11,
)
ax.set_xlim(1.0, 1.0e3)
# Full published range: lgU ∈ [0, 7] step 0.5 (15 grid points).
# Spectrum amplitude scales ~linearly with U → 7-decade dynamic
# range across the sweep.
ax.set_ylim(1.0e-26, 1.0e-17)

cmap = plt.get_cmap("viridis")
targets = list(np.arange(0.0, 7.5, 0.5))  # full 15-point grid
for k, tg in enumerate(targets):
    i = int(np.argmin(np.abs(lgU_grid - tg)))
    ax.plot(
        wave_um, li_um[i],
        color=cmap(k / max(1, len(targets) - 1)),
        lw=1.4,
        label=rf"$\log_{{10}} U={lgU_grid[i]:+.1f}$",
    )

ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=3)
ax.set_title(
    "Draine+2021 PAHspec — mMMP starlight, std ionization, std size dist",
    fontsize=11,
)

plt.show()
