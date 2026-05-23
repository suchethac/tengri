"""
Chemical evolution: How SFH and outflows shape metal enrichment history
=======================================================================

Four perspectives on chemical evolution: (1) closed-box model with varying SFR
timescales; (2) cumulative metallicity from different exponential SFHs; (3)
leaky-box model showing how outflow rates suppress Z; and (4) age-metallicity
relation across galactic radii. Together they show how star formation and
galactic winds control the Z(t) history.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time axis: look-back time in Gyr (cosmology-dependent age)
age_uni_gyr = float(tengri.age_at_z0())
t_gyr = np.linspace(0, age_uni_gyr, 200)
t_yr = t_gyr * 1e9

# Solar metallicity
Z_sun = 10.0 ** (-1.848)

# --- Panel 1: Closed-box model with different SFR timescales ---
ax = axes[0, 0]
age_from_start = age_uni_gyr - t_gyr
for tau_gyr, y_label in [(2.0, "τ=2 Gyr"), (5.0, "τ=5 Gyr"), (10.0, "τ=10 Gyr")]:
    sfr = np.exp(-age_from_start / tau_gyr)
    log_z = tengri.closed_box_metallicity(t_yr, sfr, yield_y=0.03, eta_outflow=0.0, f_gas_init=0.9)
    ax.plot(t_gyr, 10.0 ** np.array(log_z), lw=2.0, label=y_label)

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)

# --- Panel 2: Varying SFR shapes ---
ax = axes[0, 1]
tau_values = [1.0, 2.0, 5.0, 10.0]
for tau_gyr in tau_values:
    cum_mass = 1.0 - np.exp(-t_gyr / tau_gyr)
    z_evolve = Z_sun * 0.5 * cum_mass
    ax.plot(t_gyr, z_evolve / Z_sun, lw=1.5, label=f"τ={tau_gyr:.1f} Gyr")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)

# --- Panel 3: Leaky-box model (outflow dependence) ---
ax = axes[1, 0]
outflow_rates = [0.0, 0.2, 0.5, 0.8]
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(outflow_rates)))

sfr_const = np.ones_like(t_yr)
for eta, color in zip(outflow_rates, colors):
    log_z = tengri.closed_box_metallicity(
        t_yr, sfr_const, yield_y=0.03, eta_outflow=eta, f_gas_init=0.9
    )
    ax.plot(t_gyr, 10.0 ** np.array(log_z), lw=1.5, color=color, label=f"η={eta:.1f}")

ax.set_xlabel("Look-back Time [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)

# --- Panel 4: Age-metallicity relation ---
ax = axes[1, 1]
ages_gyr = np.array([2.0, 5.0, 8.0, 11.0, 13.0])
colors_amr = plt.cm.viridis(np.linspace(0, 1, len(ages_gyr)))

for age_gyr, color in zip(ages_gyr, colors_amr):
    z_amr = Z_sun * (age_gyr / age_uni_gyr) * 0.3
    ax.scatter(age_gyr, z_amr / Z_sun, s=200, color=color, edgecolors="k", linewidth=1.0)

ages_interp = np.linspace(0, age_uni_gyr, 100)
z_interp = Z_sun * (ages_interp / age_uni_gyr) * 0.3
ax.plot(ages_interp, z_interp / Z_sun, "k--", lw=1.5, alpha=0.4, label="Age-Metallicity Relation")

ax.set_xlabel("Galaxy Age [Gyr]")
ax.set_ylabel(r"Metallicity (Z / Z$_\odot$)")
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 14)
ax.set_ylim(-2.5, 0.5)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("plot_chemical_evolution.png", dpi=150, bbox_inches="tight")
