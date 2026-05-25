"""
Inoue+2014 vs Madau+1995: IGM transmission models diverge at Lyman limit
=========================================================================

Two commonly-used IGM absorption models differ significantly in the
Lyman-continuum regime. The Inoue et al. (2014) model captures Lyman-series
absorption and Lyman-continuum opacity in detail; the Madau (1995) model
provides a simpler analytical approximation. At fixed z=4, we show both
models on the same axis to reveal their differences blueward of 912 Å
rest-frame, where the discontinuity matters most for photometric SED fitting.

Reference: Inoue et al. 2014, MNRAS, 442, 1805 and Madau 1995, ApJ, 441, 18.
"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.igm import igm_transmission, igm_transmission_madau

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Observed-frame wavelength grid covering Lyman break to optical
wave_obs = jnp.linspace(500.0, 15000.0, 2000)

z_fixed = 4.0

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Inoue et al. 2014 model
trans_inoue = np.array(igm_transmission(wave_obs, z_fixed))
ax.plot(np.array(wave_obs), trans_inoue, lw=1.4, color="C0", label="Inoue+2014")

# Madau 1995 model
trans_madau = np.array(igm_transmission_madau(wave_obs, z_fixed))
ax.plot(np.array(wave_obs), trans_madau, lw=1.4, color="C1", label="Madau+1995")

# Mark Lyman break at 912 Å rest-frame
lyman_break_obs = 912.0 * (1 + z_fixed)
ax.axvline(lyman_break_obs, color="0.5", lw=0.8, ls="--", alpha=0.4)

ax.set_xscale("log")
ax.set_xlim(500, 15000)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel("IGM transmission")
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
plt.savefig("plot_igm_model_comparison.png", dpi=150, bbox_inches="tight")
