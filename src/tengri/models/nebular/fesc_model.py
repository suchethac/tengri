"""Lyman continuum escape fraction model (Chisholm+2022).

This module provides an empirical model for the ionizing photon escape fraction
(f_esc) based on UV slope, specific star formation rate, and stellar mass.
Calibrated on the Low-z Lyman Continuum Survey (Chisholm et al. 2022).

The escape fraction is critical for reionization studies and galaxy ionizing
photon budgets. This parametric model enables physically motivated priors
in spectral energy distribution fitting.

References
----------
Chisholm et al. 2022, ApJ 931, 37 (LzLCS)
Naidu et al. 2020, ApJ 892, 109
"""

import jax.numpy as jnp


def compute_uv_slope(
    wavelength_aa: jnp.ndarray,
    l_nu: jnp.ndarray,
    lam_lo: float = 1500.0,
    lam_hi: float = 2500.0,
) -> float:
    """Fit UV slope (β) from L_ν in the UV window via power law.

    Converts monochromatic luminosity L_ν to L_λ and fits a power law
    f_λ ∝ λ^β in the specified UV window via linear regression in log-log space.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelengths in Angstrom (rest frame).
    l_nu : array, shape (n_wave,)
        Monochromatic luminosity in erg/s/Hz (can be any units, slope is unitless).
    lam_lo : float, optional
        Lower wavelength limit (Angstrom) for fitting window. Default 1500.
    lam_hi : float, optional
        Upper wavelength limit (Angstrom) for fitting window. Default 2500.

    Returns
    -------
    float
        UV slope β such that f_λ ∝ λ^β.
        Typical range: −3 (blue) to 0 (red).

    Notes
    -----
    Uses the conversion L_λ = L_ν × c / λ², where c = 2.99792458e18 Å/s.
    Linear regression is computed via least-squares on the log-transformed data.
    """
    c_aa_per_s = 2.99792458e18  # speed of light in Angstrom/s

    # Mask for UV window (JIT-compatible: use jnp.where, not boolean indexing)
    w = jnp.where((wavelength_aa >= lam_lo) & (wavelength_aa <= lam_hi), 1.0, 0.0)

    # Convert L_nu to L_lambda: L_lambda = L_nu * c / lambda^2
    l_lambda = l_nu * c_aa_per_s / (wavelength_aa**2)

    # Log-space regression with masking
    log_wave = jnp.log10(jnp.where(w > 0, wavelength_aa, 1.0))
    log_l = jnp.log10(jnp.where(w > 0, jnp.maximum(l_lambda, 1e-50), 1.0))

    n = jnp.sum(w)
    sum_x = jnp.sum(w * log_wave)
    sum_y = jnp.sum(w * log_l)
    sum_xx = jnp.sum(w * log_wave**2)
    sum_xy = jnp.sum(w * log_wave * log_l)

    denom = n * sum_xx - sum_x**2
    beta = jnp.where(denom > 0.0, (n * sum_xy - sum_x * sum_y) / denom, -2.0)
    return beta


def fesc_chisholm2022(
    beta_uv: float,
    log_ssfr: float = -9.0,
    fesc_0: float = 0.15,
    a1: float = -1.22,
    a2: float = 0.0,
    ssfr_0: float = -9.0,
) -> float:
    """Empirical Lyman continuum escape fraction from UV slope and sSFR.

    Parametric model from Chisholm et al. 2022 (LzLCS) relating escape fraction
    to the UV slope (β) and specific star formation rate (sSFR).

    Parameters
    ----------
    beta_uv : float
        UV slope (dimensionless), typically in range −3 to 0.
        Bluer (more negative) values indicate higher escape fraction.
    log_ssfr : float, optional
        log10(sSFR / (Msun/yr/Msun)) = log10(SFR) − log10(M_star).
        Default −9.0 (typical quiescent galaxy).
    fesc_0 : float, optional
        Normalization (f_esc at β = −2, log sSFR = ssfr_0).
        Default 0.15 (Chisholm+2022 calibration).
    a1 : float, optional
        Slope coefficient for UV dependence.
        Default −1.22 (Chisholm+2022 calibration).
    a2 : float, optional
        Slope coefficient for sSFR dependence.
        Default 0.0 (β dependence only by default).
    ssfr_0 : float, optional
        Reference sSFR for normalization (Msun/yr/Msun in log).
        Default −9.0.

    Returns
    -------
    float
        Ionizing photon escape fraction f_esc, clipped to [0, 1].

    Notes
    -----
    Formula::

        f_esc = fesc_0 × 10^{a1 × (β + 2)} × 10^{a2 × (log sSFR − ssfr_0)}

    The (β + 2) pivot reflects the calibration choice (zero dependence at β = −2).
    A2 term enables sSFR dependence when non-zero; by default (a2=0) this is omitted.

    The clipping to [0, 1] ensures physical interpretation as a photon fraction.

    References
    ----------
    Chisholm et al. 2022, ApJ 931, 37, Section 4.2, Equation (4)
    """
    log_fesc_unnormalized = jnp.log10(fesc_0) + a1 * (beta_uv + 2.0) + a2 * (log_ssfr - ssfr_0)
    fesc = 10.0**log_fesc_unnormalized
    return jnp.clip(fesc, 0.0, 1.0)
