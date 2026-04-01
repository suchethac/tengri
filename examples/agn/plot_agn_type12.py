"""
Type 1 vs Type 2 AGN: Geometric Unification
============================================

The unified model of AGN activity: the same physical system appears as
Type 1 (broad-line, blue disc continuum visible) or Type 2 (narrow-line only,
torus blocks the accretion disc) depending purely on viewing angle.
"""

import matplotlib.pyplot as plt
import numpy as np

from tengri import setup_style
from tengri.models.agn import unified_nlr_blr

setup_style()

# --- Wavelength grid: 1e-3 -- 100 um ---
wave_aa = np.logspace(np.log10(100.0), np.log10(1e7), 2000)
wave_um = wave_aa * 1e-4

# --- Three inclination angles ---
configs = [
    {"cos_inc": 0.95, "label": "Type 1 (face-on)",    "color": "#1f77b4", "lw": 2.2},
    {"cos_inc": 0.50, "label": "Intermediate",          "color": "#ff7f0e", "lw": 1.8},
    {"cos_inc": 0.10, "label": "Type 2 (edge-on)",      "color": "#d62728", "lw": 2.0},
]

fig, ax = plt.subplots(figsize=(8, 5))

import jax.numpy as jnp

wave_jnp = jnp.array(wave_aa)
for cfg in configs:
    lnu = unified_nlr_blr(
        wave_jnp,
        agn_log_lbol=44.0,
        agn_cos_inc=cfg["cos_inc"],
        agn_theta_torus=30.0,
        agn_log_mbh=8.0,
        agn_frac=1.0,
    )
    nulnu = np.array(lnu) * 3e18 / wave_aa  # νLν
    mask = nulnu > 0
    ax.loglog(wave_um[mask], nulnu[mask],
              color=cfg["color"], lw=cfg["lw"], label=cfg["label"])

# Mark key wavelengths
for wl, lbl in [(0.1216, r"Ly$\alpha$"), (0.6563, r"H$\alpha$"), (9.7, "Silicate")]:
    ax.axvline(wl, color="grey", ls=":", lw=0.7, alpha=0.5)
    ax.text(wl * 1.05, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e35,
            lbl, fontsize=7, color="grey", va="bottom", rotation=90)

ax.set_xlim(1e-3, 100)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\nu L_\nu$ [arb.]")
ax.set_title("Type 1 vs Type 2 AGN: Same Physics, Different Viewing Angle")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
plt.savefig("plot_agn_type12.png", dpi=150, bbox_inches="tight")
plt.show()
