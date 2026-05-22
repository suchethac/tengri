"""
Stellar Population Aging: SSP at Solar Metallicity
==================================================

Five representative ages from the DSPS SSP grid at solar metallicity.
A single stellar population transitions from UV-dominated (young, hot)
to NIR-dominated (old, red); peak-normalized λF_λ on log-log axes
makes the temperature inversion visible.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_ssp_age_sweep_001.png
   :alt: plot_ssp_age_sweep
   :class: sphx-glr-single-img

"""

import matplotlib.pyplot as plt
import numpy as np

from tengri import load_ssp
from tengri.plot import setup_style

setup_style()

ssp = load_ssp()
age_gyr = 10 ** np.array(ssp.ssp_lg_age_gyr)
log_z = np.array(ssp.ssp_lgmet)
wave = np.array(ssp.ssp_wave)
flux = np.array(ssp.ssp_flux)  # (n_z, n_age, n_wave)

# Solar metallicity slice, 5 representative ages
z_solar = np.argmin(np.abs(log_z - 0.0))
target_ages = [1e-3, 0.01, 0.1, 1.0, 10.0]  # Gyr
age_idx = [np.argmin(np.abs(age_gyr - t)) for t in target_ages]
labels = ["1 Myr", "10 Myr", "100 Myr", "1 Gyr", "10 Gyr"]
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(target_ages)))

fig, ax = plt.subplots(figsize=(10, 6))
for i, label, color in zip(age_idx, labels, colors):
    # Peak-normalize so SED *shape* is comparable across 30+ decades of flux.
    lfl = wave * flux[z_solar, i, :]
    safe = np.where(lfl > 0, lfl, np.nan)
    ax.loglog(wave / 1e4, safe / np.nanmax(safe), lw=2.2, color=color, label=label)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\lambda F_\lambda$ / $\lambda F_\lambda^{\rm max}$ (peak-normalized)",
    title="Stellar Population Aging at Solar Metallicity",
    xlim=(0.05, 5.0),
    ylim=(1e-3, 2.0),
)
ax.legend(fontsize=11, frameon=False, loc="lower right")
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
plt.savefig("plot_ssp_age_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
