"""
SSP Grid: Age and Metallicity Evolution
=======================================

Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova
Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

LOG10_ZSUN = -1.848

ssp = tengri.load_ssp()
age_gyr = 10 ** np.array(ssp.ssp_lg_age_gyr)
log_z_abs = np.array(ssp.ssp_lgmet)
log_z = log_z_abs - LOG10_ZSUN  # log10(Z / Z_sun)
wave = np.array(ssp.ssp_wave)
flux = np.array(ssp.ssp_flux)

z_solar = np.argmin(np.abs(log_z - 0.0))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

wave_um = wave / 1e4

ax = axes[0, 0]
for t_target, label, c in [(0.01, "10 Myr", "C0"), (1.0, "1 Gyr", "C1"), (10.0, "10 Gyr", "C2")]:
    i = np.argmin(np.abs(age_gyr - t_target))
    lfl = wave * flux[z_solar, i]
    safe = np.where(lfl > 0, lfl, np.nan)
    ax.loglog(wave_um, safe / np.nanmax(safe), lw=1.4, color=c, label=f"Age = {label}")
ax.set_xlim(0.01, 10)
ax.set_ylim(1e-3, 2)
ax.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda F_\lambda$ (peak-normalized)")
ax.text(
    0.05,
    0.95,
    r"(a) Age sequence (Z = Z$_\odot$)",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3, which="both")

ax = axes[0, 1]
age_mid = np.argmin(np.abs(age_gyr - 1.0))
for logz_target, c in [(-1.5, "C3"), (-0.5, "C4"), (0.3, "C5")]:
    i = np.argmin(np.abs(log_z - logz_target))
    lfl = wave * flux[i, age_mid]
    safe = np.where(lfl > 0, lfl, np.nan)
    ax.loglog(
        wave_um,
        safe / np.nanmax(safe),
        lw=1.4,
        color=c,
        label=rf"Z/Z$_\odot$ = {10 ** log_z[i]:.2f}",
    )
ax.set_xlim(0.01, 10)
ax.set_ylim(1e-3, 2)
ax.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda F_\lambda$ (peak-normalized)")
ax.text(
    0.05,
    0.95,
    f"(b) Z sequence (Age = {age_gyr[age_mid]:.1f} Gyr)",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3, which="both")

ax = axes[1, 0]
for label, wl, c in [
    (r"UV (2500 $\mathrm{\AA}$)", 2500, "C0"),
    (r"Opt (5500 $\mathrm{\AA}$)", 5500, "C1"),
    (r"NIR (10000 $\mathrm{\AA}$)", 10000, "C2"),
]:
    j = np.argmin(np.abs(wave - wl))
    ax.loglog(age_gyr, flux[z_solar, :, j], lw=1.4, marker="o", markersize=4, color=c, label=label)
ax.set_xlim(1e-3, 20)
ax.set_ylim(1e-22, 1e-16)
ax.set_xlabel("Age [Gyr]")
ax.set_ylabel(r"Monochromatic flux [erg s$^{-1}$ Hz$^{-1}$ M$_\odot^{-1}$]")
ax.text(
    0.05,
    0.95,
    r"(c) Flux evolution (Z = Z$_\odot$)",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3, which="both")

ax = axes[1, 1]
age_sample = [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 13.0]
cmap_colors = plt.cm.viridis(np.linspace(0, 1, len(age_sample)))
uv, opt, nir = (np.argmin(np.abs(wave - w)) for w in (2500, 5500, 10000))
for t_target, c in zip(age_sample, cmap_colors):
    i = np.argmin(np.abs(age_gyr - t_target))
    for z_idx in range(len(log_z)):
        s = flux[z_idx, i]
        ax.scatter(
            -2.5 * np.log10(s[uv] / s[opt]),
            -2.5 * np.log10(s[opt] / s[nir]),
            s=30,
            color=c,
            alpha=0.5,
            edgecolors="none",
        )
ax.set_xlabel("UV – Opt color (mag)")
ax.set_ylabel("Opt – NIR color (mag)")
ax.text(
    0.05,
    0.95,
    "(d) Color-color (all Z)",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=mpl.colors.Normalize(vmin=min(age_sample), vmax=max(age_sample))
)
sm.set_array([])
cax = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.05, fraction=0.046)
cax.set_label("Age [Gyr]", fontsize=8)

fig.tight_layout()
plt.savefig("plot_ssp_grid.png", dpi=150, bbox_inches="tight")
