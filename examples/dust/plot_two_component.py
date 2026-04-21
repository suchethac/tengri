"""
Two-Component Dust SEDModel
=========================

Visualize the Charlot & Fall (2000) two-component dust model: birth
cloud attenuation affects only young stars (age < ~10 Myr), while
diffuse ISM attenuation affects all stars. The smooth sigmoid
transition between components is shown as a function of stellar age.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import two_component_dust

# --- Setup ---
wavelength = jnp.linspace(1000.0, 10000.0, 500)
log_ages = jnp.linspace(5.0, 10.2, 200)
age_grid = 10.0 ** log_ages

# --- Compute transmission for different dust parameters ---
configs = [
    {"tau_v1": 1.5, "tau_v2": 0.3, "label": r"$\tau_{\rm bc}=1.5$, $\tau_{\rm diff}=0.3$"},
    {"tau_v1": 0.5, "tau_v2": 0.5, "label": r"$\tau_{\rm bc}=0.5$, $\tau_{\rm diff}=0.5$"},
    {"tau_v1": 0.0, "tau_v2": 0.8, "label": r"$\tau_{\rm bc}=0.0$, $\tau_{\rm diff}=0.8$"},
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Panel A: Transmission at V-band (5500 A) vs stellar age ---
ax = axes[0]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for cfg, color in zip(configs, colors):
    trans = two_component_dust(
        wavelength, age_grid, cfg["tau_v1"], cfg["tau_v2"],
        law_bc="power_law", law_diff="power_law", n_slope=-0.7,
    )
    # Extract transmission at V-band (~5500 A)
    i_v = int(jnp.argmin(jnp.abs(wavelength - 5500.0)))
    trans_v = np.array(trans[:, i_v])
    ax.plot(np.array(log_ages), trans_v, color=color, lw=1.5, label=cfg["label"])

ax.axvline(7.0, ls=":", color="grey", lw=0.8, alpha=0.6)
ax.annotate("10 Myr", xy=(7.05, 0.05), xycoords=("data", "axes fraction"),
            fontsize=8, color="grey")
ax.set_xlabel(r"log$_{10}$(stellar age / yr)")
ax.set_ylabel("Transmission at V-band")
ax.set_title("Age-Dependent Dust Transmission")
ax.legend(fontsize=8, frameon=False)

# --- Panel B: Transmission spectrum for young vs old stars ---
ax = axes[1]
trans = two_component_dust(
    wavelength, age_grid, tau_v1=1.5, tau_v2=0.3,
    law_bc="power_law", law_diff="power_law", n_slope=-0.7,
)
wave_um = np.array(wavelength) / 1e4

# Young star (1 Myr) and old star (1 Gyr)
i_young = int(jnp.argmin(jnp.abs(log_ages - 6.0)))
i_old = int(jnp.argmin(jnp.abs(log_ages - 9.0)))
ax.plot(wave_um, np.array(trans[i_young, :]), color="#d62728", lw=1.5,
        label="Young star (1 Myr)")
ax.plot(wave_um, np.array(trans[i_old, :]), color="#1f77b4", lw=1.5,
        label="Old star (1 Gyr)")

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel("Transmission")
ax.set_title(r"Birth Cloud + Diffuse ISM ($\tau_{\rm bc}=1.5$, $\tau_{\rm diff}=0.3$)")
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
plt.savefig("plot_two_component.png", dpi=150, bbox_inches="tight")
plt.show()
