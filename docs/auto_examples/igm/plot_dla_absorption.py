"""
DLA column density sculpts the Lyman alpha forest at z=3
========================================================

Damped Lyman-alpha (DLA) systems imprint strong absorption features
blueward of the Lyman-alpha line (1216 Å rest-frame). We sweep
column density log(N_H) ∈ {19.0, 19.5, 20.0, 20.3, 20.8} cm^{-2}
at fixed redshift z=3, showing how higher column density systems
deepen the Lyman forest and suppress flux in the UV-to-optical SED.

Reference: Wolfe et al. 2005, ARA&A, 43, 861 (DLA review and cross-sections).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.igm import dla_transmission_obs
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Wavelength grid: UV to optical in observed frame
wavelength_obs = jnp.logspace(np.log10(500), np.log10(5000), 1024)

# Fixed redshift; sweep column density
z_dla = 3.0
log_nh_values = np.array([19.0, 19.5, 20.0, 20.3, 20.8])
norm = mpl.colors.Normalize(vmin=log_nh_values.min(), vmax=log_nh_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for log_nh in log_nh_values:
    tau_dla = dla_transmission_obs(wavelength_obs, z_dla=z_dla, log_n_hi=log_nh)
    transmission = jnp.exp(-tau_dla)
    ax.plot(np.array(wavelength_obs), np.array(transmission), color=cmap(norm(log_nh)), lw=1.4)

# Mark Lyman-alpha at rest-frame 1216 Å → observed frame at z=3
lya_obs = 1216.0 * (1 + z_dla)
ax.axvline(lya_obs, color="0.5", lw=0.8, ls="--", alpha=0.4)

# Zoom to the damping wing region around Lyman-alpha trough
# ±300 Å provides good visibility of the wing structure and column-density dependence
ax.set_xlim(lya_obs - 300, lya_obs + 300)
ax.set_xscale("log")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel("DLA transmission")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Column density $\log N_{\mathrm{H}}$ [cm$^{-2}$]")

fig.tight_layout()
plt.savefig("plot_dla_absorption.png", dpi=150, bbox_inches="tight")
