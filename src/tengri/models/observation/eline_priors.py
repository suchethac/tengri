"""CLOUDY-based emission line priors for spectral fitting.

Provides informative Gaussian priors on emission line amplitudes based on
pre-computed CLOUDY photoionization model line ratios. These replace the
flat (uninformative) priors in the default emission line marginalization,
improving fits for lines that are blended or have low S/N.

The priors are parameterized by metallicity and ionization parameter,
with a configurable scatter (default 0.3 dex) to accommodate non-solar
abundance patterns, AGN contamination, and other systematics.

All functions are pure JAX and JIT-compatible.

References
----------
- Byler+2017: CLOUDY+MESA Isochrones nebular emission predictions.
- Johnson+2021: Prospector emission line marginalization.
- Ferland+2017: CLOUDY photoionization code.
"""

import jax.numpy as jnp

# -------------------------------------------------------------------------
# CLOUDY reference line ratios (relative to Hbeta = 1.0)
# -------------------------------------------------------------------------

# Rest-frame vacuum wavelengths (Angstrom) for the reference lines.
# These must be in the same order as the ratio arrays below.
CLOUDY_LINE_WAVELENGTHS = jnp.array(
    [
        3727.0,  # [OII] 3726+3729 doublet
        4101.73,  # H-delta
        4340.46,  # H-gamma
        4861.33,  # H-beta (reference)
        4959.0,  # [OIII] 4959
        5007.0,  # [OIII] 5007
        6548.0,  # [NII] 6548
        6563.0,  # H-alpha
        6583.0,  # [NII] 6583
        6716.0,  # [SII] 6716
        6731.0,  # [SII] 6731
    ]
)

CLOUDY_LINE_NAMES = (
    "[OII]3727",
    "H-delta",
    "H-gamma",
    "H-beta",
    "[OIII]4959",
    "[OIII]5007",
    "[NII]6548",
    "H-alpha",
    "[NII]6583",
    "[SII]6716",
    "[SII]6731",
)

# Line ratios relative to Hbeta at solar metallicity, logU = -3.
# Source: standard CLOUDY HII region models (Byler+2017, Levesque+2010).
_CLOUDY_SOLAR_LOGU3 = jnp.array(
    [
        2.50,  # [OII] 3727 / Hbeta
        0.26,  # H-delta / Hbeta (Case B)
        0.47,  # H-gamma / Hbeta (Case B)
        1.00,  # H-beta (reference)
        0.45,  # [OIII] 4959 / Hbeta
        1.34,  # [OIII] 5007 / Hbeta
        0.13,  # [NII] 6548 / Hbeta
        2.86,  # H-alpha / Hbeta (Case B)
        0.39,  # [NII] 6583 / Hbeta
        0.22,  # [SII] 6716 / Hbeta
        0.16,  # [SII] 6731 / Hbeta
    ]
)

# Line ratios at sub-solar metallicity (0.2 Zsun), logU = -3.
# Metal lines are weaker, Balmer ratios unchanged (Case B).
_CLOUDY_SUBSOLAR_LOGU3 = jnp.array(
    [
        1.20,  # [OII] — lower at low Z
        0.26,  # H-delta (Case B, Z-independent)
        0.47,  # H-gamma (Case B, Z-independent)
        1.00,  # H-beta
        0.80,  # [OIII] 4959 — higher at low Z (less cooling)
        2.40,  # [OIII] 5007 — higher at low Z
        0.03,  # [NII] 6548 — much weaker at low Z
        2.86,  # H-alpha (Case B)
        0.09,  # [NII] 6583 — much weaker at low Z
        0.10,  # [SII] 6716 — weaker at low Z
        0.07,  # [SII] 6731 — weaker at low Z
    ]
)

# Line ratios at solar metallicity, logU = -2 (higher ionization).
# [OIII] is stronger, [NII]/[SII] weaker relative to logU=-3.
_CLOUDY_SOLAR_LOGU2 = jnp.array(
    [
        1.50,  # [OII] — lower at high U
        0.26,  # H-delta (Case B)
        0.47,  # H-gamma (Case B)
        1.00,  # H-beta
        1.10,  # [OIII] 4959 — much stronger at high U
        3.30,  # [OIII] 5007 — much stronger at high U
        0.08,  # [NII] 6548 — weaker at high U
        2.86,  # H-alpha (Case B)
        0.24,  # [NII] 6583 — weaker at high U
        0.12,  # [SII] 6716 — weaker at high U
        0.09,  # [SII] 6731 — weaker at high U
    ]
)

# Solar metallicity reference: log10(Z/Zsun) = 0
_LOG_ZSOL = 0.0
# Sub-solar reference: log10(Z/Zsun) ~ -0.7
_LOG_Z_SUBSOLAR = -0.7


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def cloudy_line_priors(
    log_z: float = 0.0,
    neb_logU: float = -3.0,
    line_wavelengths: jnp.ndarray | None = None,
    prior_width_dex: float = 0.3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return CLOUDY-based Gaussian priors on emission line ratios.

    Uses pre-computed line luminosity ratios from CLOUDY grids as
    informative priors for emission line marginalization. Interpolates
    between metallicity and ionization parameter grid points.

    Parameters
    ----------
    log_z : float
        Gas-phase metallicity log10(Z/Zsun). Default 0.0 (solar).
    neb_logU : float
        Ionization parameter log10(U). Default -3.0.
    line_wavelengths : array (n_lines,) or None
        Rest-frame wavelengths to return priors for. If None, returns
        priors for all CLOUDY reference lines.
    prior_width_dex : float
        Gaussian prior width in dex (applied in log-space as fractional
        scatter). Default 0.3 dex to accommodate non-solar abundances,
        AGN contamination, etc.

    Returns
    -------
    prior_means : array (n_lines,)
        Expected line luminosities relative to Hbeta.
    prior_sigmas : array (n_lines,)
        Prior standard deviations (in same units as means, i.e. the
        linear-space scatter corresponding to ``prior_width_dex``).
    """
    # Interpolate between metallicity grid points
    # Linear interp in log_z between sub-solar and solar
    z_frac = jnp.clip((log_z - _LOG_Z_SUBSOLAR) / (_LOG_ZSOL - _LOG_Z_SUBSOLAR), 0.0, 1.0)
    ratios_logU3 = (1.0 - z_frac) * _CLOUDY_SUBSOLAR_LOGU3 + z_frac * _CLOUDY_SOLAR_LOGU3

    # Interpolate between logU grid points (-3 and -2)
    u_frac = jnp.clip((neb_logU - (-3.0)) / ((-2.0) - (-3.0)), 0.0, 1.0)
    ratios_solar_u = (1.0 - u_frac) * _CLOUDY_SOLAR_LOGU3 + u_frac * _CLOUDY_SOLAR_LOGU2

    # Combine: Z interpolation at the appropriate U
    # For simplicity, use logU=-3 Z-interpolated ratios and blend with logU=-2
    prior_means = (1.0 - u_frac) * ratios_logU3 + u_frac * ratios_solar_u

    # Convert dex scatter to linear-space sigma: sigma = mean * (10^width - 1)
    # For small width, this approximates mean * width * ln(10)
    linear_scatter = prior_means * (10.0**prior_width_dex - 1.0)
    prior_sigmas = jnp.maximum(linear_scatter, 1e-10)

    # If specific wavelengths requested, match to nearest reference line
    if line_wavelengths is not None:
        # For each requested wavelength, find nearest CLOUDY reference line
        diffs = jnp.abs(line_wavelengths[:, None] - CLOUDY_LINE_WAVELENGTHS[None, :])
        nearest_idx = jnp.argmin(diffs, axis=1)
        prior_means = prior_means[nearest_idx]
        prior_sigmas = prior_sigmas[nearest_idx]

    return prior_means, prior_sigmas


def marginalize_emission_lines_cloudy(
    residual: jnp.ndarray,
    noise: jnp.ndarray,
    design_matrix: jnp.ndarray,
    log_z: float = 0.0,
    neb_logU: float = -3.0,
    line_wavelengths: jnp.ndarray | None = None,
    prior_width_dex: float = 0.3,
    l_hbeta: float = 1.0,
) -> tuple:
    """Emission line marginalization with CLOUDY-based priors.

    Wraps :func:`~tengri.models.observation.eline_marginalization.marginalize_emission_lines`
    with informative CLOUDY-based Gaussian priors on line amplitudes instead
    of flat priors.

    Parameters
    ----------
    residual : array (n_pix,)
        Data minus continuum model: ``d - m``.
    noise : array (n_pix,)
        Per-pixel noise standard deviation.
    design_matrix : array (n_pix, n_lines)
        Gaussian line profile design matrix.
    log_z : float
        Gas-phase metallicity log10(Z/Zsun).
    neb_logU : float
        Ionization parameter log10(U).
    line_wavelengths : array (n_lines,) or None
        Rest-frame wavelengths of the lines in the design matrix.
        Must match column order of ``design_matrix``.
    prior_width_dex : float
        Prior scatter in dex. Default 0.3.
    l_hbeta : float
        Estimated Hbeta luminosity (or flux) to scale the CLOUDY
        ratios to absolute amplitudes. Default 1.0 (ratios only).

    Returns
    -------
    ln_L_marg : scalar
        Marginalized log-likelihood.
    a_hat : array (n_lines,)
        Posterior-mean line amplitudes.
    a_cov : array (n_lines, n_lines)
        Posterior covariance of line amplitudes.
    """
    from tengri.models.observation.eline_marginalization import marginalize_emission_lines

    prior_means, prior_sigmas = cloudy_line_priors(
        log_z=log_z,
        neb_logU=neb_logU,
        line_wavelengths=line_wavelengths,
        prior_width_dex=prior_width_dex,
    )

    # Scale from Hbeta-relative ratios to absolute amplitudes
    scaled_means = prior_means * l_hbeta
    scaled_sigmas = prior_sigmas * l_hbeta

    # Convert to prior variance for the marginalization
    prior_variance = scaled_sigmas**2

    # Shift residual by prior mean (the marginalization assumes zero-mean prior
    # by default; we pre-subtract the prior mean from the residual and add it back)
    design_matrix.shape[1]
    residual_shifted = residual - design_matrix @ scaled_means

    ln_l_marg, a_hat_shifted, a_cov = marginalize_emission_lines(
        residual_shifted,
        noise,
        design_matrix,
        prior_variance=prior_variance,
    )

    # Restore the prior mean offset
    a_hat = a_hat_shifted + scaled_means

    return ln_l_marg, a_hat, a_cov
