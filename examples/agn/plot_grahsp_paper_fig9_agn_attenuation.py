"""
GRAHSP Fig. 9 reproduction: attenuation of the AGN model
=========================================================

Faithful reproduction of Fig. 9 of Buchner et al. (2024, GRAHSP): the AGN
spectrum from intrinsic (blue, top) to strongly attenuated (red, bottom) as
the AGN-only color excess ``agn_grahsp_ebv_agn`` is swept from 0.01 to 1.
GRAHSP attenuates the AGN side with an SMC/Prevot (1984) law (paper §2.1.5),
which rises steeply into the UV — so the UV/optical continuum is suppressed
far more than the near-IR, and the heaviest attenuation eventually bites into
the torus too. The intrinsic torus component is overplotted dashed black.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from tengri.analysis.plotting import setup_style
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

templates = load_grahsp_templates()
wave_nm = jnp.logspace(np.log10(80.0), np.log10(3000.0), 2500)  # 0.08 - 3 um
wave_um = np.asarray(wave_nm) / 1000.0

L5100 = 1.0e45  # bright QSO, to land in the paper's luminosity range
ebv_grid = np.logspace(np.log10(0.01), np.log10(1.0), 11)

# Diverging blue->white->red color map on log E(B-V), matching the paper.
norm = colors.LogNorm(vmin=0.01, vmax=1.0)
cmap = cm.get_cmap("RdBu_r")

fig, ax = plt.subplots(figsize=(6.4, 7.8))


def lambda_Llambda_W(arr_erg_s_nm):
    # lambda * L_lambda in W:  (wave_nm * L_lambda[erg/s/nm]) * 1e-7 [W/erg s^-1]
    return np.asarray(wave_nm) * np.asarray(arr_erg_s_nm) * 1e-7


for ebv in ebv_grid:
    sed = evaluate_grahsp_agn(
        wave_nm,
        GRAHSPParams(l5100=L5100, ebv_agn=ebv, a_feii=5.0, fcov=0.4, si=-1.0),
        templates,
    )
    total = lambda_Llambda_W(sed.bbb_attenuated + sed.torus_attenuated)
    ax.plot(wave_um, total, color=cmap(norm(ebv)), lw=1.4, zorder=3)

# Intrinsic torus (unattenuated), dashed black.
sed0 = evaluate_grahsp_agn(
    wave_nm,
    GRAHSPParams(l5100=L5100, ebv_agn=0.0, ebv=0.0, a_feii=5.0, fcov=0.4, si=-1.0),
    templates,
)
torus_intrinsic = lambda_Llambda_W(sed0.torus + sed0.si)
ax.plot(
    wave_um,
    np.clip(torus_intrinsic, 1e30, None),
    color="k",
    ls="--",
    lw=1.3,
    label="torus",
    zorder=4,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.08, 3.0)
ax.set_ylim(1e36, 1e40)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_\lambda$ [W/$\mu$m]")
ax.legend(loc="lower right", frameon=True, fontsize=10)

# Frequency top axis.
secax = ax.secondary_xaxis(
    "top", functions=(lambda x: 2.99792458e14 / x, lambda nu: 2.99792458e14 / nu)
)
secax.set_xlabel("Frequency [Hz]")

# Color bar for E(B-V)-AGN.
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, location="right")
cbar.set_label("E(B-V)-AGN")
cbar.set_ticks([0.01, 0.1, 1.0])
cbar.set_ticklabels(["0.01", "0.1", "1"])

fig.tight_layout()
plt.show()
