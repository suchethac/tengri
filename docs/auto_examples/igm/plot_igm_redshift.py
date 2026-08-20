"""
IGM transmission curves evolve sharply with redshift as Lyman forest deepens
============================================================================

The intergalactic medium (IGM) imprints wavelength-dependent opacity on
observed galaxy SEDs via Lyman-series and Lyman-continuum absorption.
The Lyman break at 912 Å rest-frame shifts to longer observed wavelengths
at higher z, enabling photometric redshift estimation via the dropout
technique. We vary redshift z ∈ {0.5, 1, 2, 3, 4, 6, 8} across the Inoue
et al. (2014) transmission model to show how IGM opacity increases with z.

Reference: Inoue et al. 2014, MNRAS, 442, 1805 (IGM transmission model).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.igm import igm_transmission
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Observed-frame wavelength grid covering UV to optical
wave_obs = jnp.linspace(500.0, 30000.0, 3000)

redshifts = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
norm = mpl.colors.Normalize(vmin=redshifts.min(), vmax=redshifts.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for z in redshifts:
    trans = igm_transmission(wave_obs, z)
    ax.plot(np.array(wave_obs), np.array(trans), color=cmap(norm(z)), lw=1.4)

# Mark Lyman break at 912 Å rest-frame for reference
ax.axvline(912.0 * 3.0, color="0.5", lw=0.8, ls="--", alpha=0.4)
ax.axvline(912.0 * 6.0, color="0.5", lw=0.8, ls="--", alpha=0.4)

ax.set_xlim(500, 30000)
ax.set_xscale("log")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel("IGM transmission")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Redshift $z$")

fig.tight_layout()
plt.savefig("plot_igm_redshift.png", dpi=150, bbox_inches="tight")
