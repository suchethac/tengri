"""
AGN Bolometric Correction: K_X(L_bol) Across Four Bands
========================================================

The bolometric correction :math:`K_X = L_{\\rm bol} / L_X` relates the
total AGN luminosity to the flux in a single observational band. For X-ray
selected AGN, this is essential for converting observed X-ray fluxes back to
total AGN power.

This gallery uses the **X-ray corona model** (:func:`tengri.xray.xray_agn_corona`)
to demonstrate how hard X-ray emission scales with AGN bolometric luminosity.
We integrate the SED across four representative bands and compute bolometric
corrections, following **Hopkins et al. (2007)** and **Duras et al. (2020)**:
AGN with higher accretion rates have softer X-ray spectra and (per unit bolometric
luminosity) relatively weaker hard X-rays, so :math:`K_{\\rm hard}` grows
monotonically with :math:`L_{\\rm bol}`.

Bands sampled:

- **Hard X-ray**: 2–10 keV (:math:`\\lambda = 1.24–6.2 \\,\\mathrm{\\AA}`)
- **Soft X-ray**: 0.5–2 keV (:math:`\\lambda = 6.2–24.8 \\,\\mathrm{\\AA}`)
- **Optical-UV**: 1000–7000 Å
- **Mid-IR**: 5–30 μm

The figure shows :math:`K_X` vs :math:`\\log L_{\\rm bol}` (log bolometric
luminosity in solar units).

References
----------
.. [1] Hopkins, A. M., et al., 2007, ApJ, 654, 731.
.. [2] Duras, F., et al., 2020, A&A, 642, A204.

"""

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.diagnostics import energy_balance
from tengri.analysis.plotting import setup_style
from tengri.xray import xray_agn_corona

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# ─────────────────────────────────────────────────────────────────────────────
# Build a simple composite SED: X-ray corona (public API)
# ─────────────────────────────────────────────────────────────────────────────
#
# For bolometric corrections, we use the X-ray corona function directly.
# This models the hard X-ray emission from the inner accretion disc corona
# and allows us to sweep the spectral properties (gamma, E_cut, alpha_ox).

# Fixed AGN parameters
X_RAY_GAMMA = 1.8  # Photon index (spectral hardness)
X_RAY_E_CUT = 300.0  # High-energy cutoff [keV]
X_RAY_ALPHA_OX = -1.4  # UV-to-X-ray slope (sets relative strength)

# ─────────────────────────────────────────────────────────────────────────────
# Bolometric correction band definitions (rest-frame, in Å and μm)
# ─────────────────────────────────────────────────────────────────────────────

# Convert keV to Å: E[keV] = 12.398 / λ[Å]
bands = {
    "hard X-ray (2–10 keV)": (1.24, 6.2),  # 2-10 keV → λ in Å
    "soft X-ray (0.5–2 keV)": (6.2, 24.8),  # 0.5-2 keV → λ in Å
    "optical-UV (1000–7000 Å)": (1000, 7000),  # rest-frame optical-UV
    "mid-IR (5–30 μm)": (5e4, 3e5),  # 5-30 μm → Å
}

band_names = list(bands.keys())
band_edges = np.array([bands[name] for name in band_names])

# Colors for each band.
band_colors = {
    "hard X-ray (2–10 keV)": "#d62728",
    "soft X-ray (0.5–2 keV)": "#ff7f0e",
    "optical-UV (1000–7000 Å)": "#2ca02c",
    "mid-IR (5–30 μm)": "#1f77b4",
}

# ─────────────────────────────────────────────────────────────────────────────
# Sweep log_lbol from 10 to 13.5 L_sun
# ─────────────────────────────────────────────────────────────────────────────

# Wavelength grid: fine enough to resolve both soft and hard X-rays.
# E[keV] = 12.398 / λ[Å], so 0.1 keV ↔ 124 Å and 100 keV ↔ 0.124 Å.
wavelength = jnp.logspace(np.log10(0.1), np.log10(1e6), 2000)  # 0.1 Å to 1 Mm

log_lbol_values = np.linspace(10.0, 13.5, 16)
k_corr_by_band = {band: [] for band in band_names}

for log_lbol in log_lbol_values:
    # Convert log_lbol [L_sun] to erg/s.
    L_sun_erg = 3.828e33  # erg/s
    L_bol_erg = 10.0**log_lbol * L_sun_erg

    # Convert L_bol → L_2500 via Hopkins+2007 BC=5.15 and call the
    # CIGALE-faithful new corona signature (post-#329).
    L_2500 = L_bol_erg / (5.15 * 1.199e15)
    sed = xray_agn_corona(
        wavelength,
        l_2500_30deg_erg_hz=L_2500,
        gamma=X_RAY_GAMMA,
        E_cut=X_RAY_E_CUT,
    )
    sed = np.asarray(sed)

    # Total bolometric luminosity (integral over full SED).
    l_bol = energy_balance.integrate_lnu_over_band(
        wavelength, sed, wavelength.min(), wavelength.max()
    )

    # Compute flux in each band and bolometric correction.
    for band_name in band_names:
        lambda_lo, lambda_hi = bands[band_name]
        # Integrate L_nu over the band to get total luminosity in that band.
        l_band = energy_balance.integrate_lnu_over_band(wavelength, sed, lambda_lo, lambda_hi)
        # Bolometric correction: K_X = L_bol / L_band.
        # Avoid division by zero.
        k_corr = float(np.where(l_band > 0, l_bol / l_band, np.nan))
        k_corr_by_band[band_name].append(k_corr)

# ─────────────────────────────────────────────────────────────────────────────
# Plot bolometric correction vs log_lbol
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.5, 5.0))

for band_name in band_names:
    k_vals = np.array(k_corr_by_band[band_name])
    ax.loglog(
        log_lbol_values,
        k_vals,
        marker="o",
        markersize=5,
        linewidth=2.0,
        label=band_name,
        color=band_colors[band_name],
    )

ax.set_xlabel(r"Log Bolometric Luminosity $\log L_{\mathrm{bol}} \, [L_\odot]$", fontsize=12)
ax.set_ylabel(r"Bolometric Correction $K_X = L_{\mathrm{bol}} / L_{\rm band}$", fontsize=12)
ax.set_title("AGN Bolometric Correction: Hard X-ray Dominance at High Luminosity", fontsize=13)
ax.legend(fontsize=10, frameon=False, loc="best")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_agn_bolometric_correction.png", dpi=150, bbox_inches="tight")
