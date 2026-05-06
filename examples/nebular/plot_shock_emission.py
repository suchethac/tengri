"""
Nebular Shock Emission (MAPPINGS V)
====================================

Plot shock emission line diagnostics from MAPPINGS V photoionization models.
Shows how shock velocity, density, and magnetic field affect line ratios
and can mimic AGN-like emission in diagnostic diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.nebular import shock_line_ratios

setup_style()

# --- Panel structure ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- Panel 1: BPT diagnostic [O III]/Hβ vs [N II]/Hα ---
ax = axes[0, 0]

# Vary shock velocity (km/s) at fixed density
v_shock_range = np.linspace(200, 500, 15)
colors = plt.cm.viridis(np.linspace(0, 1, len(v_shock_range)))

for v_shock, color in zip(v_shock_range, colors):
    try:
        line_ratios = shock_line_ratios(
            shock_velocity=float(v_shock),
            shock_log_density=2.0,
            shock_b_over_sqrt_n=1.0,
        )
        # Extract [O III]/Hβ and [N II]/Hα
        log_oiii_hb = np.log10(line_ratios["O3_5007A"] / line_ratios["Hb_4861A"])
        log_nii_ha = np.log10(line_ratios["NII_6583A"] / line_ratios["HA_6563A"])
        ax.scatter(log_nii_ha, log_oiii_hb, color=color, s=80, alpha=0.7, edgecolors="k", lw=0.3)
    except Exception:
        pass

ax.set_xlabel(r"log([N II]$\lambda$6584 / H$\alpha$)")
ax.set_ylabel(r"log([O III]$\lambda$5007 / H$\beta$)")
ax.set_title("BPT Diagram: Shock Velocity Sequence")
ax.grid(True, alpha=0.3)

# Add AGN/SF dividing lines (schematic)
x_bpt = np.linspace(-1.5, 0.5, 100)
y_bpt_agn = 0.61 / (x_bpt - 0.05) + 1.3  # Kewley+2001 AGN line
ax.plot(x_bpt, y_bpt_agn, "k--", lw=1.5, alpha=0.4, label="AGN/SF boundary")
ax.legend(fontsize=10, frameon=False)
ax.set_xlim(-1.5, 0.5)
ax.set_ylim(-1.2, 1.5)

# --- Panel 2: Shock density dependence ---
ax = axes[0, 1]

density_range = np.logspace(0, 4, 12)
colors = plt.cm.plasma(np.linspace(0, 1, len(density_range)))

for dens, color in zip(density_range, colors):
    try:
        line_ratios = shock_line_ratios(
            shock_velocity=200.0,
            shock_log_density=float(np.log10(dens)),
            shock_b_over_sqrt_n=1.0,
        )
        log_oiii_hb = np.log10(line_ratios["O3_5007A"] / line_ratios["Hb_4861A"])
        log_nii_ha = np.log10(line_ratios["NII_6583A"] / line_ratios["HA_6563A"])
        ax.scatter(log_nii_ha, log_oiii_hb, color=color, s=80, alpha=0.7, edgecolors="k", lw=0.3)
    except Exception:
        pass

ax.set_xlabel(r"log([N II]$\lambda$6584 / H$\alpha$)")
ax.set_ylabel(r"log([O III]$\lambda$5007 / H$\beta$)")
ax.set_title("BPT Diagram: Density Sequence (v_s=200 km/s)")
ax.grid(True, alpha=0.3)
ax.plot(x_bpt, y_bpt_agn, "k--", lw=1.5, alpha=0.4)
ax.set_xlim(-1.5, 0.5)
ax.set_ylim(-1.2, 1.5)

# --- Panel 3: Line flux ratios vs shock velocity ---
ax = axes[1, 0]

v_shock_dense = np.linspace(200, 500, 20)
line_flux_dict = {
    "[O III]5007/Hβ": [],
    "[O I]6300/Hα": [],
    "[S II]6724/Hα": [],
}

for v_shock in v_shock_dense:
    try:
        line_ratios = shock_line_ratios(
            shock_velocity=float(v_shock),
            shock_log_density=2.0,
            shock_b_over_sqrt_n=1.0,
        )
        line_flux_dict["[O III]5007/Hβ"].append(line_ratios["O3_5007A"] / line_ratios["Hb_4861A"])
        line_flux_dict["[O I]6300/Hα"].append(line_ratios["OI_6300A"] / line_ratios["HA_6563A"])
        line_flux_dict["[S II]6724/Hα"].append(
            (line_ratios["SII_6716A"] + line_ratios["SII_6731A"]) / line_ratios["HA_6563A"]
        )
    except Exception:
        pass

for label, ratios in line_flux_dict.items():
    if ratios:
        ax.semilogy(v_shock_dense[: len(ratios)], ratios, "o-", lw=1.5, label=label, markersize=4)

ax.set_xlabel(r"Shock Velocity [km/s]")
ax.set_ylabel("Line Flux Ratio")
ax.set_title("Shock Emission: Line Ratios vs Velocity")
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3, which="both")

# --- Panel 4: Magnetic field dependence ---
ax = axes[1, 1]

b_param_range = np.logspace(-3, 1, 12)  # μG, 0.001 to 10
colors = plt.cm.cool(np.linspace(0, 1, len(b_param_range)))

for b_param, color in zip(b_param_range, colors):
    try:
        line_ratios = shock_line_ratios(
            shock_velocity=250.0,
            shock_log_density=2.0,
            shock_b_over_sqrt_n=float(max(b_param, 1e-4)),
        )
        log_oiii_hb = np.log10(line_ratios["O3_5007A"] / line_ratios["Hb_4861A"])
        log_nii_ha = np.log10(line_ratios["NII_6583A"] / line_ratios["HA_6563A"])
        ax.scatter(log_nii_ha, log_oiii_hb, color=color, s=80, alpha=0.7, edgecolors="k", lw=0.3)
    except Exception:
        pass

ax.set_xlabel(r"log([N II]$\lambda$6584 / H$\alpha$)")
ax.set_ylabel(r"log([O III]$\lambda$5007 / H$\beta$)")
ax.set_title("BPT Diagram: Magnetic Field Strength (v_s=250 km/s)")
ax.grid(True, alpha=0.3)
ax.plot(x_bpt, y_bpt_agn, "k--", lw=1.5, alpha=0.4)
ax.set_xlim(-1.5, 0.5)
ax.set_ylim(-1.2, 1.5)

fig.suptitle("Shock Emission Diagnostics: MAPPINGS V Models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plot_shock_emission.png", dpi=100, bbox_inches="tight")
plt.show()
