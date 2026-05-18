"""
SSP Grid: Age and Metallicity Evolution
=======================================

Four-panel tour of the DSPS SSP grid: age sequence at fixed Z, metallicity
sequence at fixed age, broadband flux vs age, and a mock UV-optical vs
optical-NIR color-color diagram across the full grid.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_ssp_grid_001.png
   :alt: plot_ssp_grid
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt
import numpy as np

from tengri import load_ssp
from tengri.analysis.plotting import setup_style

setup_style()

ssp = load_ssp()
age_gyr = 10 ** np.array(ssp.ssp_lg_age_gyr)
log_z = np.array(ssp.ssp_lgmet)
wave = np.array(ssp.ssp_wave)
flux = np.array(ssp.ssp_flux)  # (n_z, n_age, n_wave)

z_solar = np.argmin(np.abs(log_z - 0.0))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Panel 1: SSP spectra at three ages, solar metallicity
ax = axes[0, 0]
for t_target, label, c in [(0.01, "10 Myr", "C0"), (1.0, "1 Gyr", "C1"), (10.0, "10 Gyr", "C2")]:
    i = np.argmin(np.abs(age_gyr - t_target))
    ax.loglog(wave / 10.0, flux[z_solar, i], lw=1.8, color=c, label=f"Age={label}")
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [arbitrary]",
    title=f"Age Sequence (Z={10 ** log_z[z_solar]:.2f} Z$_\\odot$)",
    xlim=(0.01, 10),
    ylim=(1e-72, 1e-12),
)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3, which="both")

# Panel 2: metallicity sequence at ~1 Gyr
ax = axes[0, 1]
age_mid = np.argmin(np.abs(age_gyr - 1.0))
for logz_target, c in [(-0.5, "C3"), (0.0, "C4"), (0.3, "C5")]:
    i = np.argmin(np.abs(log_z - logz_target))
    ax.loglog(
        wave / 10.0, flux[i, age_mid], lw=1.8, color=c, label=f"Z={10 ** log_z[i]:.2f} Z$_\\odot$"
    )
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [arbitrary]",
    title=f"Metallicity Sequence (Age={age_gyr[age_mid]:.2f} Gyr)",
    xlim=(0.01, 10),
    ylim=(1e-72, 1e-14),
)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3, which="both")

# Panel 3: monochromatic flux vs age at three bands (solar Z)
ax = axes[1, 0]
for label, wl, c in [
    ("UV (2500Å)", 2500, "C0"),
    ("Optical (5500Å)", 5500, "C1"),
    ("NIR (10000Å)", 10000, "C2"),
]:
    j = np.argmin(np.abs(wave - wl))
    ax.loglog(age_gyr, flux[z_solar, :, j], lw=2.0, marker="o", color=c, label=label)
ax.set(
    xlabel="Age [Gyr]",
    ylabel="Flux [arbitrary]",
    title=f"Broad-Band Flux vs Age (Z={10 ** log_z[z_solar]:.2f} Z$_\\odot$)",
    ylim=(1e-22, 1e-16),
)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3, which="both")

# Panel 4: mock UV-optical / optical-NIR color-color, all metallicities
ax = axes[1, 1]
age_sample = [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 13.0]
cmap = plt.cm.viridis(np.linspace(0, 1, len(age_sample)))
uv, opt, nir = (np.argmin(np.abs(wave - w)) for w in (2500, 5500, 10000))
for t_target, c in zip(age_sample, cmap):
    i = np.argmin(np.abs(age_gyr - t_target))
    for z_idx in range(len(log_z)):
        s = flux[z_idx, i]
        ax.scatter(
            -2.5 * np.log10(s[uv] / s[opt]),
            -2.5 * np.log10(s[opt] / s[nir]),
            s=60,
            color=c,
            alpha=0.6,
            edgecolors="k",
            linewidth=0.5,
        )
ax.set(
    xlabel="UV-Optical Color (mock)",
    ylabel="Optical-NIR Color (mock)",
    title="SSP Color-Color Diagram (all metallicities)",
)
ax.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(age_sample), vmax=max(age_sample))
)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.15, aspect=20).set_label("Age [Gyr]")

fig.suptitle("SSP Grid: Age and Metallicity Evolution", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("plot_ssp_grid.png", dpi=100, bbox_inches="tight")
plt.show()
