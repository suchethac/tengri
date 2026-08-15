"""
Hardness ratio across the alpha_OX vs log N_H plane
====================================================

The CIGALE-faithful obscured-AGN spectral model combines two knobs
that classification surveys often confound: ``delta_alpha_ox``
(offset from the empirical alpha_OX-L_2500 relation, controlling the
intrinsic X-ray-to-UV ratio) and ``log N_H`` (line-of-sight column
density, suppressing soft-band flux through ``zphabs × cabs``). We
compute the hardness ratio HR = (H - S) / (H + S) with S = 0.5-2 keV
and H = 2–10 keV across the joint (delta_alpha_ox, log N_H) plane on
a fixed L_2500 anchor (= L_bol = 1e45 erg/s through the Hopkins+2007
bolometric correction).

The contour pattern surfaces the X-ray classification degeneracy:
an X-ray-quiet unobscured AGN and an X-ray-loud obscured AGN sit on
the same HR locus. The Compton-thick boundary at log N_H = 24 is
marked; above it the soft band is dominated by the scattered floor.

References
----------

- Just et al. 2007, ApJ 665, 1004 (alpha_OX-L_2500).
- Ricci et al. 2017, Nature 549, 488 (X-ray spectral model).
- Yang et al. 2020, MNRAS 491, 740 (X-CIGALE corona).

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.xray import alpha_ox_from_l2500, xray_agn_corona

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wavelength = jnp.logspace(np.log10(0.0124), np.log10(124.0), 384)
wave_keV = 12.398 / np.array(wavelength)
soft_mask = (wave_keV >= 0.5) & (wave_keV <= 2.0)
hard_mask = (wave_keV >= 2.0) & (wave_keV <= 10.0)

L_BOL = 1.0e45
L_2500 = L_BOL / (5.15 * 1.199e15)
ALPHA_OX_J07 = float(alpha_ox_from_l2500(L_2500))

delta_grid = np.linspace(-0.4, 0.4, 17)
log_nh_grid = np.linspace(20.0, 25.5, 23)

hr = np.empty((len(delta_grid), len(log_nh_grid)))
for i, delta in enumerate(delta_grid):
    for j, log_nh in enumerate(log_nh_grid):
        sed = np.asarray(
            xray_agn_corona(
                wavelength,
                l_2500_30deg_erg_hz=L_2500,
                delta_alpha_ox=float(delta),
                log_nh=float(log_nh),
            )
        )
        nu = wave_keV * 1.602e-9 / 6.626e-27  # keV → Hz
        S = float(np.trapezoid(sed[soft_mask][np.argsort(nu[soft_mask])], np.sort(nu[soft_mask])))
        H = float(np.trapezoid(sed[hard_mask][np.argsort(nu[hard_mask])], np.sort(nu[hard_mask])))
        # Use band-luminosity ratio HR (Park+2006 convention)
        hr[i, j] = (H - S) / (H + S)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
alpha_eff = ALPHA_OX_J07 + delta_grid
mesh = ax.pcolormesh(
    log_nh_grid, alpha_eff, hr, shading="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0
)
cs = ax.contour(
    log_nh_grid,
    alpha_eff,
    hr,
    levels=[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
    colors="k",
    linewidths=0.6,
)
ax.clabel(cs, fontsize=8, fmt="%.2f")
ax.axvline(24.0, color="0.2", ls=":", lw=0.8)
ax.text(24.05, alpha_eff[-2], "Compton-thick", fontsize=8, color="0.2")
ax.set(
    xlabel=r"$\log N_H$  [cm$^{-2}$]",
    ylabel=r"effective $\alpha_{\rm OX}$  (Just+07 baseline $+\delta$)",
)
cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
cbar.set_label(r"hardness ratio  $(H - S)/(H + S)$")

fig.tight_layout()
plt.savefig("plot_xray_alpha_ox_nh.png", dpi=150, bbox_inches="tight")
