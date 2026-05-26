"""
M_BH–M_* scaling relation: Kormendy & Ho 2013 and Reines & Volonteri 2015
=========================================================================

The black hole mass (M_BH) and stellar bulge mass (M_*) of galaxies follow a
tight empirical scaling relation. This example generates synthetic galaxies
distributed around the published Kormendy & Ho (2013) and Reines & Volonteri
(2015) relations, showing their consistency with observed scatter (~0.3 dex).

Each synthetic sample spans log M_* ∈ [8.5, 12.5] M_☉ with scatter σ_M* ≈ 0.15
dex, and derives M_BH from each relation with intrinsic scatter as reported in
the original papers:

- **K&H 2013**: Applied to elliptical galaxies; scatter σ ≈ 0.28 dex
- **R&V 2015**: Extended to dwarf galaxies & low-mass systems; scatter σ ≈ 0.55 dex

**References:**

.. [1] Kormendy, J., & Ho, L. C. (2013). Coevolution of supermassive black holes
       and galaxies. Annual Review of Astronomy and Astrophysics, 51, 511–653.
       https://doi.org/10.1146/annurev-astro-082812-141024

.. [2] Reines, A. E., & Volonteri, M. (2015). Relations between central black hole
       mass and total galaxy stellar mass in the local universe. The Astrophysical
       Journal, 813(2), 82. https://doi.org/10.1088/0004-637X/813/2/82
"""

import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style

setup_style()


def mbh_from_mstar_kormendy2013(log_mstar: float | np.ndarray) -> float | np.ndarray:
    """
    Kormendy & Ho (2013) relation: log(M_BH/M_☉) = α + β·log(M_*/M_☉).

    Fit to elliptical galaxies; α=8.39, β=1.16.
    Valid for log(M_*) ~ 9.5–12.

    Parameters
    ----------
    log_mstar : float or ndarray
        Log stellar bulge mass [M_☉]

    Returns
    -------
    log_mbh : float or ndarray
        Log black hole mass [M_☉]
    """
    alpha = 8.39
    beta = 1.16
    return alpha + beta * log_mstar


def mbh_from_mstar_reines2015(log_mstar: float | np.ndarray) -> float | np.ndarray:
    """
    Reines & Volonteri (2015) relation: log(M_BH/M_☉) = α + β·log(M_*/M_☉).

    Extended to low-mass regime (IMBHs in dwarf galaxies); α=8.0, β=1.1.
    Valid for log(M_*) ~ 6–12.

    Parameters
    ----------
    log_mstar : float or ndarray
        Log total stellar mass [M_☉]

    Returns
    -------
    log_mbh : float or ndarray
        Log black hole mass [M_☉]
    """
    alpha = 8.0
    beta = 1.1
    return alpha + beta * log_mstar


# ============================================================================
# Generate synthetic samples
# ============================================================================

# Random seed for reproducibility
rng = np.random.default_rng(42)

# Generate synthetic stellar masses: uniform in log space over [8.5, 12.5]
n_samples = 200
log_mstar_samples = rng.uniform(8.5, 12.5, n_samples)

# Add small scatter to stellar mass
scatter_mstar = 0.15  # dex
log_mstar_kh = log_mstar_samples + rng.normal(0, scatter_mstar, n_samples)
log_mstar_rv = log_mstar_samples + rng.normal(0, scatter_mstar, n_samples)

# Apply K&H 2013 relation with intrinsic scatter
log_mbh_kh_relation = mbh_from_mstar_kormendy2013(log_mstar_kh)
scatter_kh = 0.28  # dex (reported in K&H 2013)
log_mbh_kh = log_mbh_kh_relation + rng.normal(0, scatter_kh, n_samples)

# Apply R&V 2015 relation with intrinsic scatter
log_mbh_rv_relation = mbh_from_mstar_reines2015(log_mstar_rv)
scatter_rv = 0.55  # dex (reported in R&V 2015)
log_mbh_rv = log_mbh_rv_relation + rng.normal(0, scatter_rv, n_samples)

# ============================================================================
# Plotting
# ============================================================================

fig, ax = plt.subplots(figsize=(8.0, 6.5))

# K&H 2013: scatter
ax.scatter(
    log_mstar_kh,
    log_mbh_kh,
    s=30,
    alpha=0.4,
    color="C0",
    edgecolors="none",
    label="Kormendy & Ho (2013) samples",
    zorder=2,
)

# R&V 2015: scatter
ax.scatter(
    log_mstar_rv,
    log_mbh_rv,
    s=30,
    alpha=0.4,
    color="C1",
    edgecolors="none",
    label="Reines & Volonteri (2015) samples",
    zorder=2,
)

# Overlay the deterministic relation lines
mstar_line = np.linspace(8.5, 12.5, 100)
mbh_kh_line = mbh_from_mstar_kormendy2013(mstar_line)
mbh_rv_line = mbh_from_mstar_reines2015(mstar_line)

ax.plot(
    mstar_line,
    mbh_kh_line,
    "-",
    color="C0",
    lw=2.5,
    alpha=0.8,
    label=r"K&H 2013: $\log M_{\mathrm{BH}} = 8.39 + 1.16 \log M_*$",
    zorder=3,
)
ax.plot(
    mstar_line,
    mbh_rv_line,
    "-",
    color="C1",
    lw=2.5,
    alpha=0.8,
    label=r"R&V 2015: $\log M_{\mathrm{BH}} = 8.0 + 1.1 \log M_*$",
    zorder=3,
)

# Axis labels and limits
ax.set_xlabel(r"Log Stellar Mass $\log M_* / M_\odot$", fontsize=11)
ax.set_ylabel(r"Log Black Hole Mass $\log M_{\mathrm{BH}} / M_\odot$", fontsize=11)
ax.set_xlim(8.5, 12.5)
ax.set_ylim(6.0, 11.0)

# Legend
ax.legend(
    loc="upper left",
    fontsize=9,
    frameon=True,
    fancybox=True,
    shadow=False,
    framealpha=0.95,
)

# Grid
ax.grid(True, alpha=0.25, which="both", linestyle="-", linewidth=0.4)

fig.tight_layout()
plt.savefig("plot_mbh_mstar_relation.png", dpi=150, bbox_inches="tight")
