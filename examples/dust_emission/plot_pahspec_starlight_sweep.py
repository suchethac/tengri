r"""
Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U
=============================================================

Sweep across the 13 published PAHspec starlight spectra (mMMP, m31bulge,
BC03/BPASS SSPs) at fixed ionization parameter. Demonstrates strong
dependence of PAH features on starlight hardness.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri import data_path, load_pahspec_draine2021, select_pahspec_axes
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

tpl = load_pahspec_draine2021(data_path("pahspec_draine2021.h5"))
wave_um = np.asarray(tpl.wavelength_um)
lgU_grid = np.asarray(tpl.lgU)
i_lgU1 = int(np.argmin(np.abs(lgU_grid - 1.0)))

starlights = tuple(tpl.starlight_names)

fig, ax = plt.subplots(figsize=(7.5, 5.0))
cmap = plt.get_cmap("viridis")
n = max(1, len(starlights) - 1)
norm = mpl.colors.Normalize(vmin=0, vmax=n)
for k, name in enumerate(starlights):
    if name not in tpl.starlight_names:
        continue
    nu_pnu = select_pahspec_axes(
        tpl,
        starlight=name,
        ionization="st",
        size_distribution="std",
        slab=False,
    )
    li = np.asarray(nu_pnu[i_lgU1]) / (4.0 * np.pi)
    ax.plot(wave_um, li, color=cmap(norm(k)), lw=1.3)
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\lambda I_\lambda / N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$",
    xlim=(2.0, 1.0e3),
    ylim=(1.0e-27, 5.0e-24),
)
# Colorbar with starlight-spectrum names as discrete ticks (13 entries, too many for legend).
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, ticks=range(len(starlights)), pad=0.01)
cbar.ax.set_yticklabels(list(starlights), fontsize=7)
cbar.set_label("Starlight spectrum (softer → harder)", fontsize=9)
fig.tight_layout()
plt.savefig("plot_pahspec_starlight_sweep.png", dpi=150, bbox_inches="tight")
