"""
IMF Comparison: Mass-to-Light Ratio
====================================

Different Initial Mass Functions produce different M/L ratios at fixed age
and metallicity. We rescale a Chabrier SSP by literature M/L values for
Chabrier, Kroupa, and Salpeter at 1 Gyr, solar metallicity. The NIR
(where massive stars dominate the mass budget) is most diagnostic of IMF choice.

Reference: Conroy 2012, ApJ, 747, 69; Conroy, Gunn & White 2009.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
age_gyr = 10 ** np.array(ssp.ssp_lg_age_gyr)
log_z = np.array(ssp.ssp_lgmet)
wave = np.array(ssp.ssp_wave)
flux = np.array(ssp.ssp_flux)

z_solar = np.argmin(np.abs(log_z - 0.0))
age_1gyr = np.argmin(np.abs(age_gyr - 1.0))

imfs = [
    ("Chabrier", 1.00, "#0173B2"),
    ("Kroupa", 1.15, "#029E73"),
    ("Salpeter", 1.55, "#D55E00"),
]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, ml, c in imfs:
    lfl = (wave * flux[z_solar, age_1gyr]) / ml
    safe = np.where(lfl > 0, lfl, np.nan)
    ax.loglog(
        wave / 1e4, safe / np.nanmax(safe), lw=1.4, color=c, label=f"{name} (M/L = {ml:.2f})"
    )

ax.set_xlim(0.05, 5.0)
ax.set_ylim(1e-3, 2.0)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
ax.set_ylabel(r"$\lambda F_\lambda$ / $\lambda F_\lambda^{\rm max}$ (peak-normalized)")

ax.legend(fontsize=8, frameon=False, loc="lower right")

fig.tight_layout()
plt.savefig("plot_ssp_imf_compare.png", dpi=150, bbox_inches="tight")
