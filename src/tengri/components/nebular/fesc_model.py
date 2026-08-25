# SPDX-License-Identifier: BSD-3-Clause
"""Empirical Lyman continuum escape fraction models.

This module provides parametric and empirical models for the ionizing photon
escape fraction f_esc, critical for reionization studies and ionizing photon
budgets in galaxies. Primary model calibration: Chisholm et al. 2022 (LzLCS),
relating f_esc to UV spectral slope β and specific SFR.

Escape fraction encodes the fraction of hydrogen-ionizing photons (λ < 911.76 Å)
that escape the ISM/CGM without photoionizing gas. Values range [0, 1], with
f_esc ≈ 0.1–0.3 in low-z starburst galaxies and potentially f_esc > 0.5 at
high-z (z > 6) during reionization epoch.
"""

import jax.numpy as jnp

from tengri.utils.physics_constants import C_AA


def compute_uv_slope(
    wavelength_aa: jnp.ndarray,
    l_nu: jnp.ndarray,
    lam_lo: float = 1500.0,
    lam_hi: float = 2500.0,
) -> float:
    r"""Measure UV spectral slope β via power-law fit in FUV.

    Fits the continuum spectrum to a power law λ^β in the 1500–2500 Å window
    (rest-frame FUV), a standard practice in high-z galaxy characterization.
    The UV slope correlates with stellar age, dust attenuation, and ionizing
    photon production; it is a key predictor in the f_esc escape-fraction model.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame, increasing).
    l_nu : array, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz] (units cancel in slope fit)
    lam_lo : float, optional
        Lower edge of fitting window. Default: 1500 Å. [Å]
    lam_hi : float, optional
        Upper edge of fitting window. Default: 2500 Å. [Å]

    Returns
    -------
    float
        Dimensionless UV slope β where f_λ ∝ λ^β. Typical range [−3, 0];
        more negative = bluer (younger, less attenuated).

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` operations and masking (no Python
    conditionals on traced values).

    **Power law conversion** (Buat et al. 1989, Meurer et al. 1999):
        Input spectrum is L_ν [erg/s/Hz]. Convert to L_λ via the wavelength
        Jacobian:

        .. math::

            L_\lambda = L_\nu \, \left|\frac{\mathrm{d}\nu}{\mathrm{d}\lambda}\right|
                = L_\nu \, \frac{c}{\lambda^2}

        Then fit log(L_λ) = log(L_0) + β log(λ) to get the slope β.

    **Least-squares regression**:
        Minimizes χ² = Σ [log(L_λ) − log(L_0) − β log(λ)]² in log-log space.
        Equivalent to fitting L_λ ∝ λ^β linearly in log space.

    **Masking**:
        Only wavelengths in [lam_lo, lam_hi] contribute to the fit. Pixels
        outside are masked to zero weight (JIT-safe). If insufficient in-window
        pixels, the fit returns a default β = −2.0.

    References
    ----------
    .. [1] G. R. Meurer, T. M. Heckman, and D. Calzetti, "Dust Absorption and
       the Ultraviolet Luminosity Density at z ~ 3 as Calibrated by Local
       Starburst Galaxies," ApJ, 521, 64 (1999). arXiv:astro-ph/9903054.
       https://doi.org/10.1086/307523
    .. [2] V. Buat et al., "Far-Infrared Observations of Extremely Luminous
       Infrared Galaxies," A&A, 223, 42 (1989).

    """
    # Mask for UV window (JIT-compatible: use jnp.where, not boolean indexing)
    w = jnp.where((wavelength_aa >= lam_lo) & (wavelength_aa <= lam_hi), 1.0, 0.0)

    # Convert L_nu to L_lambda: L_lambda = L_nu * c / lambda^2
    l_lambda = l_nu * C_AA / (wavelength_aa**2)

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
    r"""Predict Lyman continuum escape fraction from UV slope and sSFR.

    Empirical parametric model calibrated on the Low-z Lyman Continuum Survey
    (Chisholm et al. 2022). Encodes the dependence of ionizing photon escape
    on UV spectral slope (proxy for stellar age and dust), and optionally on
    specific star formation rate (ISM conditions).

    Parameters
    ----------
    beta_uv : float
        UV spectral slope. [dimensionless]
        Typical range [−3, 0]: −3 = very blue (young, metal-poor);
        0 = red (old, dust-attenuation-dominated).
    log_ssfr : float, optional
        Specific star formation rate. [log10(yr^{-1})]
        Defined as log10(SFR / M_star). Typical range [−11, −8].
        Default: −9.0 (quiescent galaxies).
    fesc_0 : float, optional
        Normalization: f_esc at pivot point (β = −2, log sSFR = ssfr_0).
        Default: 0.15 (Chisholm+ calibration on local LyC survey).
    a1 : float, optional
        UV slope coefficient. Default: −1.22 (Chisholm+).
        Negative a1 means bluer UV → higher f_esc (physically expected).
    a2 : float, optional
        sSFR coefficient. Default: 0.0 (no sSFR dependence by default).
        Set to non-zero value to enable sSFR-dependent f_esc.
    ssfr_0 : float, optional
        Reference sSFR for normalization. [log10(yr^{-1})]
        Default: −9.0 (matches log_ssfr default).

    Returns
    -------
    float
        Ionizing photon escape fraction. [dimensionless, ∈ [0, 1]]
        Clipped to physical bounds [0, 1].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    **Parametric form** (Chisholm et al. 2022, §4.2, Eq. 4):

        .. math::

            f_{\mathrm{esc}} = f_{\mathrm{esc}, 0} \,
                10^{a_1 (\beta - (-2))} \,
                10^{a_2 (\log \dot{\mathrm{SFR}}_\ast - \mathrm{sSFR}_0)}

        where β is UV slope, sSFR is specific star formation rate
        [yr^{-1}], and the pivot point (β = −2) is chosen by calibration.

    **Physical interpretation**:

        - Bluer UV (more negative β) → higher f_esc (younger stars, less dust)
        - Higher sSFR (when a2 > 0) → higher f_esc (more ionizing photons,
          less dust shielding)
        - sSFR dependence is weak and often set a2=0 (default) in practice

    **Calibration source**: Low-z Lyman Continuum Survey (Chisholm et al. 2022)
        Local (z ≈ 0.03) sample of 8 star-forming galaxies with direct LyC
        measurements. Escape fractions span 0.01 < f_esc < 0.30. Caution: small
        sample; extrapolation to higher redshifts or different galaxy types may
        incur larger uncertainties.

    References
    ----------
    .. [1] J. Chisholm et al., "The far-ultraviolet continuum slope as a Lyman
       Continuum escape estimator at high redshift," MNRAS, 517, 5104 (2022).
    .. [2] R. P. Naidu et al., "Rapid Reionization by the Oligarchs: The Case
       for Massive, UV-bright, Star-forming Galaxies with High Escape Fractions,"
       ApJ, 892, 109 (2020). arXiv:1907.13130.
       https://doi.org/10.3847/1538-4357/ab7cc9

    """
    log_fesc_unnormalized = jnp.log10(fesc_0) + a1 * (beta_uv + 2.0) + a2 * (log_ssfr - ssfr_0)
    fesc = 10.0**log_fesc_unnormalized
    return jnp.clip(fesc, 0.0, 1.0)
