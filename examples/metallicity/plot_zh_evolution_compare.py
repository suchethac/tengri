"""
Chemical Evolution: Model Comparison
=====================================

Compare metallicity evolution Z(t) from closed-box and leaky-box chemical
evolution models. Shows how star formation history and gas outflows shape
the enrichment history. Demonstrates how different feedback mechanisms
alter metal abundance relative to a closed box (maximum enrichment).

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_zh_evolution_compare_001.png
   :alt: plot_zh_evolution_compare
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt
import numpy as np

from tengri.cosmology import age_at_z0
from tengri.plot import setup_style
from tengri.sfh import closed_box_metallicity

setup_style()

# --- Time axis: cosmological age (look-back time in Gyr) ---
age_uni_gyr = float(age_at_z0())
t_gyr = np.linspace(0, min(13.5, age_uni_gyr), 200)
t_yr = t_gyr * 1e9

Z_sun = 10.0 ** (-1.848)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Chemical Evolution: Model Comparison", fontsize=13, y=0.995)

# --- Panel 1: Closed-box with varying timescales ---
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0.0, 0.85, 4))
age_from_start = age_uni_gyr - t_gyr
for tau_gyr, color in zip([1.0, 2.0, 5.0, 10.0], colors):
    sfr = np.exp(-age_from_start / tau_gyr)
    log_z = closed_box_metallicity(t_yr, sfr, yield_y=0.03, eta_outflow=0.0, f_gas_init=0.9)
    z_ratio = 10.0 ** np.array(log_z)
    ax.plot(t_gyr, z_ratio, lw=2.0, color=color, label=f"τ = {tau_gyr:.1f} Gyr")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title("Closed-Box: SFR Timescale Dependence", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 13)
ax.set_ylim(0, 2.0)

# --- Panel 2: Closed-box vs leaky-box at τ = 3 Gyr ---
ax = axes[0, 1]
sfr_ref = np.exp(-age_from_start / 3.0)
eta_values = [0.0, 0.2, 0.5, 0.8]
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(eta_values)))

for eta, color in zip(eta_values, colors):
    log_z = closed_box_metallicity(t_yr, sfr_ref, yield_y=0.03, eta_outflow=eta, f_gas_init=0.9)
    z_ratio = 10.0 ** np.array(log_z)
    ax.plot(t_gyr, z_ratio, lw=2.0, color=color, label=f"η = {eta:.1f}")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title(r"Leaky-Box: Outflow Rate η Dependence (τ = 3 Gyr)", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 13)
ax.set_ylim(0, 2.0)

# --- Panel 3: Constant SFR comparison ---
ax = axes[1, 0]
sfr_const = np.ones_like(t_yr)
models = [
    (0.0, "Closed-box (η=0)", "#1f77b4"),
    (0.3, "Leaky (η=0.3)", "#ff7f0e"),
    (0.5, "Leaky (η=0.5)", "#2ca02c"),
    (0.8, "Leaky (η=0.8)", "#d62728"),
]

for eta, label, color in models:
    log_z = closed_box_metallicity(t_yr, sfr_const, yield_y=0.03, eta_outflow=eta, f_gas_init=0.9)
    z_ratio = 10.0 ** np.array(log_z)
    ax.plot(t_gyr, z_ratio, lw=2.0, color=color, label=label)

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title("Constant SFR: Closed vs Leaky Box", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 13)
ax.set_ylim(0, 3.0)

# --- Panel 4: Age-metallicity relation analogue ---
ax = axes[1, 1]
# Analogy: different lookback times ↔ different metallicities within a single galaxy
# Early-assembled regions have lower [Z/Z_sun] (less enrichment time)
# Recently assembled regions have higher [Z/Z_sun] (continuous enrichment)

t_assembly = np.array([1.0, 3.0, 7.0, 10.0])  # Lookback time of assembly (Gyr)
z_assembly = []

for t_lbt in t_assembly:
    idx = int(np.argmin(np.abs(t_gyr - t_lbt)))
    if 0 <= idx < len(t_gyr):
        sfr_tau3 = np.exp(-age_from_start / 3.0)
        log_z = closed_box_metallicity(
            t_yr[:idx], sfr_tau3[:idx], yield_y=0.03, eta_outflow=0.2, f_gas_init=0.9
        )
        if len(log_z) > 0:
            z_assembly.append(10.0 ** log_z[-1])
        else:
            z_assembly.append(0.0)

ax.scatter(t_assembly, z_assembly, s=100, color="#9467bd", alpha=0.7, edgecolors="black", lw=1.5)
ax.plot(t_assembly, z_assembly, lw=2.0, color="#9467bd", alpha=0.5, label="Assembly sequence")

ax.set_xlabel("Assembly Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity at Assembly (Z / Z$_\odot$)")
ax.set_title("Age-Metallicity Relation Analogue", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 11)
ax.set_ylim(0, 1.0)

fig.tight_layout()
plt.savefig("plot_zh_evolution_compare.png", dpi=150, bbox_inches="tight")
plt.show()
