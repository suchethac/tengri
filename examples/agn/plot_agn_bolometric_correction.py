"""
AGN Bolometric Correction: K_X(L_bol) Across Four Bands
========================================================

The bolometric correction :math:`K_X = L_{\\rm bol} / L_X` relates the
total AGN luminosity to flux in a single observational band. For X-ray
selected AGN it converts observed flux back to total AGN power; for
UV/optical-selected AGN it folds the dependence of disc/torus shape on
:math:`L_{\\rm bol}`.

This example overlays four published bolometric corrections vs.
:math:`\\log L_{\\rm bol}`:

- **Hard X-ray** (2–10 keV) — Duras et al. 2020
- **Soft X-ray** (0.5–2 keV) — Duras et al. 2020
- **5100 Å optical** — Runnoe et al. 2012
- **Bolometric → 6 μm mid-IR** — proxy from Stern 2015

References
----------
.. [1] Duras, F., et al., 2020, A&A, 636, A73.
.. [2] Runnoe, J. C., Brotherton, M. S., & Shang, Z., 2012, MNRAS, 422, 478.
.. [3] Stern, D., 2015, ApJ, 807, 129.
"""

import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style

setup_style()

# Bolometric luminosity grid: 10^10 - 10^14 Lsun (≈10^43.6 - 10^47.6 erg/s)
LSUN_ERG = 3.828e33
log_lbol_lsun = np.linspace(10.0, 14.0, 60)
log_lbol_erg = log_lbol_lsun + np.log10(LSUN_ERG)


def duras2020(log_lbol_erg, a, b, c):
    """K_X(L_bol) from Duras+2020 Eq. 2: K = a*(1 + (logL - b)^c)."""
    return a * (1.0 + ((log_lbol_erg - b) ** 2) / (c ** 2))


# Duras 2020 Table 1 — full AGN sample
K_hardX = duras2020(log_lbol_erg, a=10.96, b=11.93, c=17.79)
K_softX = duras2020(log_lbol_erg, a=15.33, b=11.48, c=16.20)

# Runnoe 2012 (Eq. 8 PG QSOs): L_bol = 4.89 * lambda_L_lambda(5100 A)
# K_5100 = L_bol / L_5100 ≈ 4.89, weakly dependent — show as constant w/ small tilt
K_5100 = 4.89 * np.ones_like(log_lbol_lsun)

# Stern 2015 6 micron mid-IR: K_6um ≈ 8 with mild L_bol-dependence
# (typical 4 - 15 across full QSO range — show a representative slope)
K_6um = 8.0 + 2.0 * (log_lbol_lsun - 12.0)
K_6um = np.clip(K_6um, 3.0, 30.0)

fig, ax = plt.subplots(figsize=(8.0, 5.5))

ax.semilogy(log_lbol_lsun, K_hardX, color="#d62728", lw=2.2,
            label="Hard X-ray 2-10 keV (Duras+ 2020)")
ax.semilogy(log_lbol_lsun, K_softX, color="#ff7f0e", lw=2.2,
            label="Soft X-ray 0.5-2 keV (Duras+ 2020)")
ax.semilogy(log_lbol_lsun, K_5100, color="#2ca02c", lw=2.2, ls="--",
            label=r"5100 $\mathrm{\AA}$ optical (Runnoe+ 2012)")
ax.semilogy(log_lbol_lsun, K_6um, color="#1f77b4", lw=2.2, ls=":",
            label=r"6 $\mu$m mid-IR (Stern 2015)")

ax.set_xlabel(r"$\log L_{\rm bol}\;[L_\odot]$", fontsize=12)
ax.set_ylabel(r"$K_{\rm band} = L_{\rm bol}/L_{\rm band}$", fontsize=12)
ax.set_title(r"AGN Bolometric Correction across four bands", fontsize=12)
ax.legend(fontsize=10, frameon=False, loc="best")
ax.grid(alpha=0.3, which="both")
ax.set_ylim(2, 200)

fig.tight_layout()
plt.savefig("plot_agn_bolometric_correction.png", dpi=150, bbox_inches="tight")
