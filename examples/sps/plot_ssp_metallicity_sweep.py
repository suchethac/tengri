"""
Stellar Metallicity Effects on SED
==================================

Metallicity reddens the optical continuum and shifts iron-peak absorption
features in the near-IR. We show five metallicity points spanning the SSP
grid at fixed age (1 Gyr). Peak-normalized λF_λ makes spectral shape
variations visible without large luminosity differences obscuring them.

Reference: DSPS SSP grid (Conroy et al. 2009).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
age_gyr = 10 ** np.array(ssp.ssp_lg_age_gyr)
log_z = np.array(ssp.ssp_lgmet)
wave = np.array(ssp.ssp_wave)
flux = np.array(ssp.ssp_flux)

age_1gyr = np.argmin(np.abs(age_gyr - 1.0))

LOG10_ZSUN = -1.848
targets_zsol = [-1.5, -1.0, -0.3, 0.0, 0.3]
met_idx = [np.argmin(np.abs(log_z - (t + LOG10_ZSUN))) for t in targets_zsol]
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(met_idx)))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i, c in zip(met_idx, colors):
    lfl = wave * flux[i, age_1gyr]
    safe = np.where(lfl > 0, lfl, np.nan)
    label = rf"$\log Z/Z_\odot$ = {log_z[i] - LOG10_ZSUN:+.2f}"
    ax.loglog(wave / 1e4, safe / np.nanmax(safe), lw=1.4, color=c, label=label)

ax.set_xlim(0.05, 5.0)
ax.set_ylim(1e-3, 2.0)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
ax.set_ylabel(r"$\lambda F_\lambda$ / $\lambda F_\lambda^{\rm max}$ (peak-normalized)")

ax.legend(fontsize=8, frameon=False, loc="lower right")

fig.tight_layout()
plt.savefig("plot_ssp_metallicity_sweep.png", dpi=150, bbox_inches="tight")
