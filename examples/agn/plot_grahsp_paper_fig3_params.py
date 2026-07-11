"""
GRAHSP Fig. 3 reproduction: AGN model parameter map
====================================================

Faithful reproduction of Fig. 3 of Buchner et al. (2024, GRAHSP),
showing how the 15 AGN parameters configure the spectrum in
:math:`L_\\lambda` (arbitrary units). The bending power-law BBB
(blue) is normalized at 5100 Å with optical slope
:math:`\\beta`, bending at :math:`\\lambda_{\\rm bend}` (width
:math:`W_{\\rm bend}`) to the UV slope :math:`\\beta_{\\rm UV}`. Emission
lines (light red) of width :math:`W_{\\rm lines}` and an
FeII forest (dark red) are scaled by
:math:`A_{\\rm lines}` / :math:`A_{\\rm FeII}`. The log-Gaussian torus
(dark yellow) has cool/hot components at
:math:`\\lambda_{\\rm cool}/\\lambda_{\\rm hot}` (widths
:math:`W_{\\rm cool}/W_{\\rm hot}`), peak ratio :math:`f_{\\rm hot}`, 12 µm
normalization :math:`f_{\\rm cov}`, and silicate depth ``Si`` (here −1,
absorption; dotted). Component colors map to the GRAHSP/pcigale modules
``activatepl`` (BBB), ``activategtorus`` (torus), and ``activatelines``
(lines + FeII).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.agn import GRAHSPParams, evaluate_grahsp_agn, load_grahsp_templates
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# --- Illustrative parameter set (matches the shapes in Buchner+24 Fig. 3) ---
LAMBDA_BEND_NM = 100.0
W_BEND = 1.0
BETA = -1.7
COOL_LAM_UM, COOL_WIDTH = 20.0, 0.5
HOT_LAM_UM, HOT_WIDTH = 2.5, 0.5
HOT_FCOV = 1.0
FCOV = 0.7
SI = -1.0

params = GRAHSPParams(
    l5100=1e44,
    plslope=BETA,
    uvslope=0.0,
    plbendloc_nm=LAMBDA_BEND_NM,
    plbendwidth=W_BEND,
    a_lines=1.0,
    a_feii=5.0,
    linewidth_kms=5000.0,
    fcov=FCOV,
    si=SI,
    cool_lam_um=COOL_LAM_UM,
    cool_width=COOL_WIDTH,
    hot_lam_um=HOT_LAM_UM,
    hot_width=HOT_WIDTH,
    hot_fcov=HOT_FCOV,
)

templates = load_grahsp_templates()
wave_nm = jnp.logspace(np.log10(40.0), np.log10(1.0e5), 4000)  # 0.04 - 100 um
wave_um = np.asarray(wave_nm) / 1000.0
sed = evaluate_grahsp_agn(wave_nm, params, templates)

W = 1e-7  # erg/s/nm -> W/nm
bbb = np.asarray(sed.bbb) * W
torus = np.asarray(sed.torus) * W
si = np.asarray(sed.si) * W
lines = np.asarray(sed.broad_lines + sed.narrow_lines) * W
feii = np.asarray(sed.feii) * W

fig, ax = plt.subplots(figsize=(11.0, 5.0))
ax.plot(wave_um, bbb, color="#1f77d4", lw=2.4, label="BBB (bending power-law)", zorder=4)
ax.plot(wave_um, torus, color="#d6a319", lw=2.4, label="Torus", zorder=4)
# Si shown dotted where it perturbs the torus (continuum + Si).
ax.plot(
    wave_um,
    np.clip(torus + si, 1e-30, None),
    color="#d6a319",
    lw=1.6,
    ls=":",
    label="Torus + silicate",
    zorder=5,
)
ax.plot(wave_um, lines, color="#ff6b6b", lw=0.8, label="Emission lines", zorder=3)
ax.plot(wave_um, feii, color="#8b1a1a", lw=0.8, label="FeII forest", zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.04, 100.0)
ax.set_ylim(1e32, 1e36)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\lambda$ [W/nm]")
ax.legend(loc="upper right", frameon=True, fontsize=9)

# ----- Parameter annotations (mirroring the paper layout) -----
i5100 = int(np.argmin(np.abs(wave_um - 0.510)))
l5100 = bbb[i5100]
# L5100 normalization: vertical drop + blue marker at 5100 A.
ax.plot([0.510], [l5100], "o", color="#1f77d4", ms=9, zorder=6)
ax.annotate(
    "",
    xy=(0.510, 1.3e32),
    xytext=(0.510, l5100),
    arrowprops=dict(arrowstyle="-", color="k", lw=1.2),
)
ax.text(0.33, 6e32, r"$L_{5100\,\mathrm{\AA}}^{\mathrm{AGN}}$", fontsize=13, ha="center")
ax.text(0.60, l5100 * 0.5, r"$\beta$", fontsize=13)

# UV slope / bend annotations on the BBB (top-left, decluttered).
ax.text(0.043, 6.0e35, r"$\beta_{\rm UV}$", fontsize=12)
ax.text(0.105, 4.3e35, r"$\lambda_{\rm bend}$", fontsize=12)
ax.annotate(
    "",
    xy=(0.050, 2.4e35),
    xytext=(0.20, 2.4e35),
    arrowprops=dict(arrowstyle="-", color="k", lw=0.9),
)
ax.text(0.072, 2.7e35, r"$W_{\rm bend}$", fontsize=12)

# Line annotations.
ax.text(0.295, 5.2e35, r"$W_{\rm lines}$", fontsize=12)
ax.annotate(
    "",
    xy=(0.32, 1.6e35),
    xytext=(0.32, 4.0e35),
    arrowprops=dict(arrowstyle="-", color="k", lw=0.9),
)
ax.text(0.36, 1.2e35, r"$A_{\rm lines}$", fontsize=12)
ax.text(0.165, 8.5e34, r"$A_{\rm FeII}$", fontsize=12)

# Torus annotations.
ihot = int(np.argmin(np.abs(wave_um - HOT_LAM_UM)))
ax.text(HOT_LAM_UM * 0.82, torus[ihot] * 1.25, r"$\lambda_{\rm hot}$", fontsize=11)
ax.text(1.6, 7e32, r"$f_{\rm hot}$", fontsize=11)
ax.annotate(
    "", xy=(2.0, 1.5e33), xytext=(6.5, 1.5e33), arrowprops=dict(arrowstyle="-", color="k", lw=0.9)
)
ax.text(3.0, 1.7e33, r"$W_{\rm hot}$", fontsize=11)
# fcov bracket: 5100A level across to 12 um.
ax.annotate(
    "",
    xy=(0.510, l5100 * 1.05),
    xytext=(12.0, l5100 * 1.05),
    arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
)
ax.annotate(
    "",
    xy=(12.0, l5100 * 1.05),
    xytext=(12.0, torus[int(np.argmin(np.abs(wave_um - 12.0)))]),
    arrowprops=dict(arrowstyle="-", color="k", lw=0.8),
)
ax.text(13.0, 3e33, r"$f_{\rm cov}$", fontsize=11)
ax.text(8.5, 6e32, "Si", fontsize=11)
icool = int(np.argmin(np.abs(wave_um - COOL_LAM_UM)))
ax.text(COOL_LAM_UM * 0.9, torus[icool] * 1.3, r"$\lambda_{\rm cool}$", fontsize=11)
ax.annotate(
    "", xy=(18.0, 8e32), xytext=(55.0, 8e32), arrowprops=dict(arrowstyle="-", color="k", lw=0.9)
)
ax.text(26.0, 9e32, r"$W_{\rm cool}$", fontsize=11)

fig.tight_layout()
plt.show()
