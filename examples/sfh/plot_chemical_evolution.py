"""
Chemical Evolution: Metallicity History Models
===============================================

Plot metallicity evolution Z(t) from closed-box and leaky-box chemical
evolution models. Shows how star formation history and gas outflows
shape the history of metal enrichment in galaxies.
"""

import matplotlib.pyplot as plt
import numpy as np

from tengri.components.sfh import closed_box_metallicity, closed_box_metallicity_anchored

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time axis: look-back time in Gyr
t_gyr = np.linspace(0, 13.8, 200)
# Convert to years for the function (if needed)
t_yr = t_gyr * 1e9

# Solar metallicity
Z_sun = 10.0 ** (-1.848)

# --- Panel 1: Closed-box model with different calibrations ---
ax = axes[0, 0]

# Closed-box metallicity: Z(t) = Z_sun * log(1 / (1 - x))
# where x = (M_recycled + M_ejected) / M_initial is cumulative mass fraction
# For a simple closed box, x scales with (time / age_max)^alpha

z_vals = []
labels = []

# Simple closed-box: constant Z increase
z_simple = closed_box_metallicity(t_yr, z_solar=1.0, m_recycled_msun=1e10)
ax.plot(t_gyr, np.array(z_simple) / Z_sun, lw=2.0, label="Closed-box (Z_sun=1)")

# Try anchored version
z_anchored = closed_box_metallicity_anchored(
    t_yr, z_solar=1.0, z_anchor=0.5, t_anchor_gyr=2.0, m_recycled_msun=1e10
)
ax.plot(t_gyr, np.array(z_anchored) / Z_sun, lw=2.0, label="Closed-box anchored")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title("Closed-Box Chemical Evolution")
ax.legend(fontsize=9, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)

# --- Panel 2: Varying SFR shapes (implicit via parametric SFH) ---
ax = axes[0, 1]

# Age grid
age_gyr = 13.8  # Galaxy age
tau_values = [1.0, 2.0, 5.0, 10.0]  # Exponential timescales in Gyr

for tau_gyr in tau_values:
    # SFH: PSI(t) ∝ exp(-t / tau)
    # Metal abundance tracks cumulative stellar mass formed
    cum_mass = 1.0 - np.exp(-t_gyr / tau_gyr)
    z_evolve = Z_sun * 0.5 * cum_mass  # Simplified enrichment

    ax.plot(t_gyr, z_evolve / Z_sun, lw=1.5, label=f"τ={tau_gyr:.1f} Gyr")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title("Chemical Evolution: Varying SFR Timescales")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)

# --- Panel 3: Leaky-box model (outflow dependence) ---
ax = axes[1, 0]

# Leaky-box: Z(t) = Z_sun * (1 - eta) * log(1/(1-x)) where eta = outflow rate
# Higher eta → more metal loss → lower Z at given mass
outflow_rates = [0.0, 0.2, 0.5, 0.8]  # eta: 0 = closed, 1 = maximal outflow
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(outflow_rates)))

for eta, color in zip(outflow_rates, colors):
    # Simplified leaky-box: Z scales with (1 - eta)
    z_leaky = Z_sun * (1.0 - eta) * closed_box_metallicity(t_yr, z_solar=1.0, m_recycled_msun=1e10)
    ax.plot(t_gyr, np.array(z_leaky) / Z_sun, lw=1.5, color=color, label=f"η={eta:.1f}")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title("Leaky-Box Model: Outflow Rate Dependence")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)

# --- Panel 4: Metallicity gradient (radial dependence via age-metallicity) ---
ax = axes[1, 1]

# Simpler representation: age-metallicity relation across a disk
# Inner regions: older, lower metallicity (early assembly)
# Outer regions: younger, higher metallicity (ongoing star formation)

ages_gyr = np.array([2.0, 5.0, 8.0, 11.0, 13.0])
colors_amr = plt.cm.viridis(np.linspace(0, 1, len(ages_gyr)))

for age_gyr, color in zip(ages_gyr, colors_amr):
    # Mock Z(age) curve: Z increases with formation time
    z_amr = Z_sun * (age_gyr / 13.8) * 0.3  # Normalized to ~0.3 Z_sun at age 13.8 Gyr
    ax.scatter(age_gyr, z_amr / Z_sun, s=200, color=color, edgecolors="k", linewidth=1.0)

# Interpolate
ages_interp = np.linspace(0, 13.8, 100)
z_interp = Z_sun * (ages_interp / 13.8) * 0.3
ax.plot(ages_interp, z_interp / Z_sun, "k--", lw=1.5, alpha=0.4, label="Age-Metallicity Relation")

ax.set_xlabel("Galaxy Age [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.set_title("Age-Metallicity Relation")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 14)
ax.set_ylim(0, 0.4)

fig.suptitle("Chemical Evolution: Metallicity History Models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plot_chemical_evolution.png", dpi=100, bbox_inches="tight")
plt.show()
