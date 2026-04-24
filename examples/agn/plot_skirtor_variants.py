"""
SKIRTOR Clumpy Torus: Inclination and Optical Depth
=====================================================

Plot the SKIRTOR (Stalevski et al. 2016) clumpy torus model varying
viewing angle (inclination) and optical depth at 9.7 μm (tau_97).
Shows how geometric effects and dust clumping transform the torus SED.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Locate SKIRTOR grid file
_grid_path = None
for p in [
    Path("data/skirtor_grid.h5"),
    Path("../data/skirtor_grid.h5"),
    Path("../../data/skirtor_grid.h5"),
]:
    if p.exists():
        _grid_path = str(p)
        break

if _grid_path is None:
    raise SystemExit(
        "Skipping: SKIRTOR grid not found. Run: python scripts/build_skirtor_grid.py"
    )

from tengri.analysis.plotting import setup_style
from tengri.components.agn import create_skirtor_from_grid

setup_style()

# Load the SKIRTOR interpolator
skirtor_fn = create_skirtor_from_grid(_grid_path)

# Wavelength grid: 0.5 - 500 micron (IR torus dominated)
wavelength = jnp.logspace(np.log10(5e3), np.log10(5e6), 512)
wave_um = np.array(wavelength) / 1e4

# Figure: 2x3 grid showing tau_97 and inclination variations
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Left column: fixed tau_97=25 (optically thick), vary inclination
tau_97_thick = 25.0
for ax, cos_inc, title in [
    (axes[0, 0], 0.95, "Face-on (θ=18°)"),
    (axes[1, 0], 0.50, "Edge-on (θ=60°)"),
]:
    seds = []
    labels = []
    for p_frac in [0.5, 1.0, 2.0]:  # clumping parameter p
        try:
            sed = skirtor_fn(
                wavelength,
                agn_log_lbol=11.0,
                agn_log_tau_97=np.log10(tau_97_thick),
                agn_p_frac=p_frac,
                agn_q_ratio=2.0,
                agn_oa_angle=0.6,
                agn_cos_inc=cos_inc,
            )
            seds.append(np.array(sed))
            labels.append(f"p={p_frac:.1f}")
        except Exception:
            continue

    for sed, lbl in zip(seds, labels):
        ax.loglog(wave_um, sed, lw=1.5, label=lbl)

    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax.set_title(f"{title} (τ_97={tau_97_thick:.0f})")
    ax.legend(fontsize=10, frameon=False)
    ax.set_xlim(0.5, 500)

# Middle column: fixed inclination, vary tau_97
cos_inc_fixed = 0.5
for ax, p_frac, title in [
    (axes[0, 1], 1.0, "p=1.0, vary τ_97"),
    (axes[1, 1], 2.0, "p=2.0, vary τ_97"),
]:
    for tau_97 in [5.0, 15.0, 30.0]:
        try:
            sed = skirtor_fn(
                wavelength,
                agn_log_lbol=11.0,
                agn_log_tau_97=np.log10(tau_97),
                agn_p_frac=p_frac,
                agn_q_ratio=2.0,
                agn_oa_angle=0.6,
                agn_cos_inc=cos_inc_fixed,
            )
            ax.loglog(wave_um, np.array(sed), lw=1.5, label=f"τ_97={tau_97:.0f}")
        except Exception:
            continue

    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax.set_title(title)
    ax.legend(fontsize=10, frameon=False)
    ax.set_xlim(0.5, 500)

# Right column: Total SED landscape (3D heatmap as color)
for row, cos_inc in enumerate([0.9, 0.3]):
    ax = axes[row, 2]

    tau_97_vals = np.linspace(5.0, 40.0, 6)
    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_97_vals)))

    for tau_97, color in zip(tau_97_vals, colors):
        try:
            sed = skirtor_fn(
                wavelength,
                agn_log_lbol=11.0,
                agn_log_tau_97=np.log10(tau_97),
                agn_p_frac=1.5,
                agn_q_ratio=2.0,
                agn_oa_angle=0.6,
                agn_cos_inc=cos_inc,
            )
            ax.loglog(
                wave_um,
                np.array(sed),
                lw=1.2,
                color=color,
                label=f"τ_97={tau_97:.1f}",
            )
        except Exception:
            continue

    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax.set_title(f"Luminosity landscape (θ={np.degrees(np.arccos(cos_inc)):.0f}°)")
    ax.legend(fontsize=10, frameon=False, ncol=2)
    ax.set_xlim(0.5, 500)

fig.suptitle("SKIRTOR Clumpy Torus: Inclination, Optical Depth, and Clumping Effects", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plot_skirtor_variants.png", dpi=100, bbox_inches="tight")
plt.show()
