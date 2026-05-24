r"""
Two PAH libraries respond to log U with the same FIR-peak migration
====================================================================

The mid-infrared ionisation-parameter sensitivity is library-specific, but
the FIR-peak migration with rising log U is a universal prediction. We
overlay the Hensley & Draine 2023 (Astrodust+PAH) and the Draine+2021
PAHspec libraries at the same three log U values to surface where the two
agree (FIR peak position) and where they differ (MIR PAH-feature strength
and the Astrodust silicate plateau near 18 microns).

References:
    Hensley, B.S. & Draine, B.T. 2023, ApJ, 948, 55.
    Draine, B.T. et al. 2021, ApJ, 917, 3.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from tengri import data_path
from tengri.analysis.plotting import setup_style
from tengri.components.dust.astrodust_hd23 import load_astrodust_hd23_or_raise
from tengri.components.dust.draine2021_pah import (
    load_pahspec_or_raise,
    select_pahspec_axes,
)

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_CGS = 2.99792458e10
SHOWN_LGU = (-1.0, 1.0, 3.0)

hd23 = load_astrodust_hd23_or_raise(data_path("astrodust_templates.h5"))
wave_um_hd23 = np.asarray(hd23.wavelength_um)
lgU_hd23 = np.asarray(hd23.lgU)
lam_cm_hd23 = wave_um_hd23 * 1.0e-4
li_um_hd23 = np.asarray(hd23.L_nu_total) * C_CGS / (4.0 * np.pi * lam_cm_hd23[None, :])

pah = load_pahspec_or_raise(data_path("pahspec_draine2021.h5"))
nu_pnu_pah = select_pahspec_axes(
    pah, starlight="mMMP", ionization="st", size_distribution="std", slab=False
)
wave_um_pah = np.asarray(pah.wavelength_um)
lgU_pah = np.asarray(pah.lgU)
li_um_pah = np.asarray(nu_pnu_pah) / (4.0 * np.pi)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
cmap = plt.get_cmap("viridis")

for k, target in enumerate(SHOWN_LGU):
    color = cmap(k / (len(SHOWN_LGU) - 1))
    i_hd = int(np.argmin(np.abs(lgU_hd23 - target)))
    axes[0].plot(
        wave_um_hd23,
        li_um_hd23[i_hd],
        color=color,
        lw=1.4,
        label=rf"$\log_{{10}} U={lgU_hd23[i_hd]:+.1f}$",
    )
    i_pah = int(np.argmin(np.abs(lgU_pah - target)))
    axes[1].plot(
        wave_um_pah,
        li_um_pah[i_pah],
        color=color,
        lw=1.4,
        label=rf"$\log_{{10}} U={lgU_pah[i_pah]:+.1f}$",
    )

for ax in axes:
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
        xlim=(1.0, 1.0e3),
        ylim=(1.0e-26, 1.0e-18),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)

axes[0].set_ylabel(r"$\lambda I_\lambda / N_{\rm H}\ [\mathrm{erg\,s^{-1}\,sr^{-1}\,H^{-1}}]$")
axes[0].text(0.04, 0.95, "Astrodust+PAH (HD23)", transform=axes[0].transAxes, va="top")
axes[1].text(0.04, 0.95, "PAHspec (D21)", transform=axes[1].transAxes, va="top")

fig.tight_layout()
plt.savefig("plot_logu_cross_library.png", dpi=150, bbox_inches="tight")
