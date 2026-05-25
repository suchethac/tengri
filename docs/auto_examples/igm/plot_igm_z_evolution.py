"""
Lyman forest deepens with redshift: high-z IGM opacity suppresses UV flux
==========================================================================

The intergalactic medium (IGM) opacity increases dramatically with redshift
due to the expanding neutral hydrogen fraction. We sweep redshift z ∈ {2, 3, 4, 5, 6, 7, 8}
on Inoue et al. (2014) IGM transmission curves, showing how the Lyman alpha
forest deepens and the Lyman break (912 Å rest-frame) shifts to longer
observed wavelengths at higher z, suppressing flux blueward of the break.

Reference: Inoue et al. 2014, MNRAS, 442, 1805 (IGM transmission model).
"""

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.igm import igm_transmission

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Observed-frame wavelength grid covering UV to optical
wave_obs = jnp.linspace(800.0, 30000.0, 3000)

redshifts = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
norm = mpl.colors.Normalize(vmin=redshifts.min(), vmax=redshifts.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for z in redshifts:
    trans = igm_transmission(wave_obs, z)
    ax.plot(np.array(wave_obs), np.array(trans), color=cmap(norm(z)), lw=1.4)

ax.set_xlim(800, 30000)
ax.set_xscale("log")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel("IGM transmission")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Redshift $z$")

fig.tight_layout()
plt.savefig("plot_igm_z_evolution.png", dpi=150, bbox_inches="tight")
