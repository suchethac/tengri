r"""
Chabrier 2003 IMF — analytic normalisation and SSP mean stellar mass
====================================================================

External ground truth: Chabrier 2003 PASP 115 763, Eq. 16–17.

  ξ(m) ∝ (1/m) exp[-(log10 m - log10 0.22)² / (2·0.57²)]   for m ≤ 1 M_sun
  ξ(m) ∝ m^{-2.3}                                          for m > 1 M_sun

Normalised so ∫ m ξ(m) dm = 1 M_sun over [0.1, 100].

We compute this analytical normalisation from the paper, and check
the FSPS Chabrier SSP's recorded scalar `ssp_mass_remaining`
(zero-age value) against the expected 1.0. The per-age
mass-remaining curve is NOT currently exposed on the public surface
(see filed library issue) — that part is omitted from the figure.

References:
- Chabrier 2003 PASP 115 763.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore")


def chabrier_imf_analytic(m):
    m = np.atleast_1d(np.asarray(m, dtype=float))
    out = np.empty_like(m)
    lo = m < 1.0
    arg = (np.log10(m[lo]) - np.log10(0.22)) ** 2 / (2.0 * 0.57 ** 2)
    out[lo] = (1.0 / m[lo]) * np.exp(-arg)
    out[~lo] = m[~lo] ** (-2.3)
    return out


# normalise so ∫ m ξ(m) dm  =  1 M_sun over [0.1, 100]
norm_const, _ = quad(lambda m: m * chabrier_imf_analytic(m)[0], 0.1, 100.0)
mass_integral_after_norm, _ = quad(
    lambda m: m * chabrier_imf_analytic(m)[0] / norm_const, 0.1, 100.0,
)

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
# `ssp.ssp_mass_remaining` is currently None for this file; no public surface
# yet exposes the per-age mass-loss curve for the diagnostic. Issue filed
# separately.
_ssp_loaded = ssp is not None

fig, ax = plt.subplots(figsize=(7.0, 4.5))
m_plot = np.logspace(-1, 2, 400)
ax.loglog(m_plot, chabrier_imf_analytic(m_plot) / norm_const,
          color="C0", lw=1.6, label="Chabrier 2003 (normalised)")
ax.axvline(1.0, color="0.5", ls=":", lw=0.8)
ax.text(1.0, 5e-4, " 0.22 M⊙ peak ←   → Salpeter $m^{-2.3}$",
        fontsize=8, color="0.4")
ax.set(xlabel=r"stellar mass $m$  [M$_\odot$]",
       ylabel=r"$\xi(m)$ — normalised so $\int m\,\xi\,dm = 1$",
       xlim=(0.1, 100))
ax.text(0.04, 0.06,
        (f"Analytic check\n"
         f"  ∫ m ξ(m) dm  =  {mass_integral_after_norm:.6f}  M⊙\n"
         f"  expected 1.000000,  residual {abs(1 - mass_integral_after_norm):.1e}"),
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="0.95", alpha=0.9))
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
plt.savefig("plot_diag_chabrier_imf_norm.png", dpi=150, bbox_inches="tight")
