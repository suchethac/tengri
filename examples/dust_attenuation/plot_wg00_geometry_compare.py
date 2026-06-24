"""
Witt & Gordon 2000: geometry and grain type at fixed optical depth
===================================================================

The WG00 radiative-transfer grid (FSPS ``dust_type=3``) spans three large-scale
star-dust geometries — *shell* (a foreground screen), *cloudy* (a homogeneous
star-dust mix), and *dusty* (a clumpy two-phase medium) — crossed with two grain
populations (Milky-Way and SMC). At a fixed ``tau_V`` these choices set the
*shape* of the transmission ``exp(-A(lambda))``: the foreground screen is the
reddest (steepest UV), while the mixed and clumpy geometries are progressively
grayer because short-wavelength photons escape through low-opacity sightlines.

This figure fixes ``tau_V = 3`` and overlays the three geometries for the MW
(solid) and SMC (dashed) grain curves, using the public
``tengri.dust.wg00_attenuation`` accessor. The SMC curves bite harder in the
far-UV (steeper grains); the geometry ordering (shell reddest, cloudy grayest)
holds for both. See :doc:`plot_wg00_tau_v_sweep` for how these shapes themselves
evolve with ``tau_V``.

References
----------
Witt, A. N. & Gordon, K. D. 2000, ApJ, 528, 799
("Multiple Scattering in Clumpy Media. II. Galactic Environments").
Tables as distributed by FSPS (Conroy & Gunn 2010, ``dust_type=3``).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.dust import wg00_attenuation

setup_style()

wave = jnp.geomspace(1000.0, 30000.0, 500)
wave_um = np.asarray(wave) / 1e4
tau_v = 3.0

GEOM = [
    ("shell", "Shell (screen)", "#1f77b4"),
    ("dusty", "Dusty (clumpy)", "#ff7f0e"),
    ("cloudy", "Cloudy (mixed)", "#2ca02c"),
]

fig, ax = plt.subplots(figsize=(6.8, 4.4))
for geom, label, color in GEOM:
    for curve, ls, tag in (("mw", "-", "MW"), ("smc", "--", "SMC")):
        a = np.asarray(wg00_attenuation(wave, tau_v, dust_curve=curve, geometry=geom))
        ax.plot(
            wave_um,
            np.exp(-a),
            color=color,
            ls=ls,
            lw=1.5,
            label=f"{label} — {tag}",
        )

ax.axvline(0.55, color="0.7", ls=":", lw=0.9)
ax.text(0.57, 0.05, "V band", color="0.45", fontsize=8)
ax.set(
    xscale="log",
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mu$m]",
    ylabel=r"Transmission $\exp[-A(\lambda;\,\tau_V)]$",
    title=r"WG00 geometry $\times$ grain type at $\tau_V = 3$",
    xlim=(0.1, 3.0),
    ylim=(0, 1.0),
)
ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower right")

plt.tight_layout()
plt.show()
