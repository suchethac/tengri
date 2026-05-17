r"""
Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U
=============================================================

Sweep the categorical ``starlight`` config across the 13 published
PAHspec choices (mMMP, m31bulge, BC03 / BPASS at various ages and
metallicities) at fixed ``log10 U = 1`` and the standard
``(ionization, size_distribution)`` defaults.  This shows that
PAH features scale strongly with starlight hardness — the headline
result of the paper.

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
        "PAHspec HDF5 not found. Build with "
        "`python scripts/build_pahspec_hdf5.py --download`."
    )

from tengri.components.dust.draine2021_pah import (
    select_pahspec_axes,
)

tpl = load_pahspec_or_raise(_PATH)
wave_um = np.asarray(tpl.wavelength_um)
lgU_grid = np.asarray(tpl.lgU)
i_lgU1 = int(np.argmin(np.abs(lgU_grid - 1.0)))

# All 13 starlight spectra published in the PAHspec library:
# 2 non-SSP ambient + 6 BC03 SSPs + 5 BPASS SSPs.
starlights = tuple(tpl.starlight_names)

fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda\ [\mu\mathrm{m}]$", fontsize=12)
ax.set_ylabel(
    r"$\lambda I_\lambda / N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    fontsize=11,
)
ax.set_xlim(2.0, 1.0e3)
# At fixed lgU=1, all 8 starlights span λI_λ ~ 1e-26 to 5e-24:
# FIR peak (~150 μm) at ~5e-24, MIR PAH features at ~1e-25, with
# starlight-spectrum-dependent variation across this range.
ax.set_ylim(1.0e-27, 5.0e-24)

cmap = plt.get_cmap("plasma")
for k, name in enumerate(starlights):
    if name not in tpl.starlight_names:
        continue
    nu_pnu = select_pahspec_axes(
        tpl, starlight=name, ionization="st",
        size_distribution="std", slab=False,
    )
    li = np.asarray(nu_pnu[i_lgU1]) / (4.0 * np.pi)  # erg/s/sr/H
    ax.plot(
        wave_um, li,
        color=cmap(k / max(1, len(starlights) - 1)),
        lw=1.3,
        label=name,
    )

ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=1)
ax.set_title(
    r"Draine+2021 PAHspec — starlight sweep at $\log_{10} U = 1$",
    fontsize=11,
)

plt.show()
