# SPDX-License-Identifier: BSD-3-Clause
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
import numpy as np

from tengri.observation.eline_catalog import (
    CLOUDY_LINE_NAMES,  # noqa: F401, re-exported for convenience
    CLOUDY_LINE_WAVELENGTHS,
)

# ── CLOUDY reference line ratios (relative to Hbeta = 1.0) ────────

# Line ratios relative to Hbeta at solar metallicity, logU = -3.
# Source: standard CLOUDY HII region models (Byler+2017, Levesque+2010).
# [OII] 3726+3729 total ~2.50; split with 3729/3726 ~ 1.3 (n_e ~ 100 cm^-3).
_CLOUDY_SOLAR_LOGU3 = jnp.array(
    [
        1.09,  # [OII] 3726 / Hbeta  (2.50 / 2.30)
        1.41,  # [OII] 3729 / Hbeta  (2.50 * 1.3 / 2.30)
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
        0.52,  # [OII] 3726, lower at low Z  (1.20 / 2.30)
        0.68,  # [OII] 3729, lower at low Z  (1.20 * 1.3 / 2.30)
        0.26,  # H-delta (Case B, Z-independent)
        0.47,  # H-gamma (Case B, Z-independent)
        1.00,  # H-beta
        0.80,  # [OIII] 4959, higher at low Z (less cooling)
        2.40,  # [OIII] 5007, higher at low Z
        0.03,  # [NII] 6548, much weaker at low Z
        2.86,  # H-alpha (Case B)
        0.09,  # [NII] 6583, much weaker at low Z
        0.10,  # [SII] 6716, weaker at low Z
        0.07,  # [SII] 6731, weaker at low Z
    ]
)

# Line ratios at solar metallicity, logU = -2 (higher ionization).
# [OIII] is stronger, [NII]/[SII] weaker relative to logU=-3.
_CLOUDY_SOLAR_LOGU2 = jnp.array(
    [
        0.65,  # [OII] 3726, lower at high U  (1.50 / 2.30)
        0.85,  # [OII] 3729, lower at high U  (1.50 * 1.3 / 2.30)
        0.26,  # H-delta (Case B)
        0.47,  # H-gamma (Case B)
        1.00,  # H-beta
        1.10,  # [OIII] 4959, much stronger at high U
        3.30,  # [OIII] 5007, much stronger at high U
        0.08,  # [NII] 6548, weaker at high U
        2.86,  # H-alpha (Case B)
        0.24,  # [NII] 6583, weaker at high U
        0.12,  # [SII] 6716, weaker at high U
        0.09,  # [SII] 6731, weaker at high U
    ]
)

# Line ratios at sub-solar metallicity (0.2 Zsun), logU = -2.
# 4th grid corner required for proper bilinear interpolation.
# At low Z + high U: less metal cooling → hotter HII region → stronger [OIII];
# [NII]/[SII] very weak; [OII] suppressed by both high U (ionized to [OIII])
# and low Z. Values derived from Byler+2017 CLOUDY trends.
_CLOUDY_SUBSOLAR_LOGU2 = jnp.array(
    [
        0.25,  # [OII] 3726, strongly suppressed (low Z + high U)
        0.32,  # [OII] 3729, strongly suppressed (low Z + high U)
        0.26,  # H-delta (Case B, Z-independent)
        0.47,  # H-gamma (Case B, Z-independent)
        1.00,  # H-beta
        1.80,  # [OIII] 4959, enhanced (low Z + high U)
        5.40,  # [OIII] 5007, strongly enhanced (low Z + high U)
        0.02,  # [NII] 6548, very weak (low Z + high U)
        2.86,  # H-alpha (Case B)
        0.05,  # [NII] 6583, very weak (low Z + high U)
        0.05,  # [SII] 6716, weak at low Z
        0.04,  # [SII] 6731, weak at low Z
    ]
)

# Solar metallicity reference: log10(Z/Zsun) = 0
_LOG_ZSOL = 0.0
# Sub-solar reference: log10(Z/Zsun) ~ -0.7
_LOG_Z_SUBSOLAR = -0.7


# ── Public API ────────────────────────────────────────────────────


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
    prior_means : array, shape (n_lines,)
        Expected line luminosities relative to Hbeta [dimensionless].
    prior_sigmas : array, shape (n_lines,)
        Prior standard deviations [dimensionless], linear-space scatter
        corresponding to ``prior_width_dex``.

    Notes
    -----
    **JIT-compatible**: yes, uses only jnp primitives.
    Performs bilinear interpolation over a 2×2 grid in (Z, logU) space.
    For richer priors using a full CLOUDY grid, use ``cloudy_grid_line_priors()``.

    """
    # Bilinear interpolation over all 4 (Z, logU) grid corners.
    # Grid: Z ∈ {sub-solar, solar}, logU ∈ {-3, -2}
    # Without all 4 corners, metallicity becomes a no-op at the logU=-2 boundary.
    z_frac = jnp.clip((log_z - _LOG_Z_SUBSOLAR) / (_LOG_ZSOL - _LOG_Z_SUBSOLAR), 0.0, 1.0)
    u_frac = jnp.clip((neb_logU - (-3.0)) / ((-2.0) - (-3.0)), 0.0, 1.0)

    prior_means = (
        (1.0 - z_frac) * (1.0 - u_frac) * _CLOUDY_SUBSOLAR_LOGU3
        + z_frac * (1.0 - u_frac) * _CLOUDY_SOLAR_LOGU3
        + (1.0 - z_frac) * u_frac * _CLOUDY_SUBSOLAR_LOGU2
        + z_frac * u_frac * _CLOUDY_SOLAR_LOGU2
    )

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

    Wraps :func:`~tengri.observation.eline_marginalization.marginalize_emission_lines`
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
        Marginalized log-likelihood [dimensionless].
    a_hat : array, shape (n_lines,)
        Posterior-mean line amplitudes (same units as ``residual``).
    a_cov : array, shape (n_lines, n_lines)
        Posterior covariance of line amplitudes (same units^2 as ``a_hat``).

    Notes
    -----
    **JIT-compatible**: no (calls external function).
    For richer priors using a full CLOUDY grid, use ``cloudy_grid_line_priors()``.

    """
    from tengri.observation.eline_marginalization import marginalize_emission_lines

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


def cloudy_grid_line_priors(
    grid_data,
    log_z: float,
    neb_logU: float,
    log_age_yr: float = 7.0,
    prior_width_dex: float = 0.3,
    target_wavelengths: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """CLOUDY-grid-interpolated emission line priors.

    Uses the full CLOUDY HDF5 grid (typically shape n_met × n_age × n_logU ×
    n_lines in log10 space) to compute line ratio priors via trilinear
    interpolation, replacing the hardcoded 2×2 grid in ``cloudy_line_priors()``.

    Parameters
    ----------
    grid_data : CloudyGridData
        Loaded CLOUDY grid (from ``load_cloudy_grid(path)`` in
        ``tengri.components.nebular``). Must have attributes:
        ``line_luminosity`` (n_met, n_age, n_logU, n_lines),
        ``line_log_met``, ``line_log_age``, ``line_log_U``,
        ``line_wavelengths``.
    log_z : float
        Gas-phase metallicity log10(Z) (absolute, not Z/Zsun).
    neb_logU : float
        Ionization parameter log10(U).
    log_age_yr : float
        log10(age / yr) for the dominant stellar population.
        Default 7.0 (10 Myr, typical for HII regions).
    prior_width_dex : float
        Gaussian prior width in dex. Default 0.3.
    target_wavelengths : array (n_target,) or None
        If provided, return priors only for these rest-frame wavelengths
        by matching each to the nearest grid line. If None, returns priors
        for all grid lines.

    Returns
    -------
    prior_means : array, shape (n_lines_out,)
        Line luminosities relative to Hbeta [dimensionless].
    prior_sigmas : array, shape (n_lines_out,)
        Prior standard deviations [dimensionless].

    Notes
    -----
    **JIT-compatible**: no (uses NumPy for interpolation).
    The grid stores ``line_luminosity`` in log10 space. The function:

    1. Clamps (log_z, neb_logU, log_age_yr) to grid bounds
    2. Performs trilinear interpolation across the 3 axes in log10 space
    3. Exponentiates: ``10**interpolated`` to get linear luminosities
    4. Normalizes relative to Hbeta (detected by wavelength ~4861 Å)
    5. Applies dex scatter: ``sigma = mean * (10**width - 1)``

    This function is preferred over ``cloudy_line_priors()`` when a full
    CLOUDY grid is available, as it provides richer metallicity/age/ionization
    coverage.

    """
    # Clamp to grid bounds
    log_z_c = float(jnp.clip(log_z, grid_data.line_log_met.min(), grid_data.line_log_met.max()))
    log_u_c = float(jnp.clip(neb_logU, grid_data.line_log_U.min(), grid_data.line_log_U.max()))
    log_a_c = float(
        jnp.clip(log_age_yr, grid_data.line_log_age.min(), grid_data.line_log_age.max())
    )

    # Helper: 1D linear interp weight
    def _frac(arr, val):
        """Compute interpolation index and fractional weight for 1D linear interpolation."""
        idx = np.searchsorted(arr, val) - 1
        idx = np.clip(idx, 0, len(arr) - 2)
        t = (val - arr[idx]) / (arr[idx + 1] - arr[idx] + 1e-30)
        return int(idx), float(np.clip(t, 0.0, 1.0))

    iz, tz = _frac(np.array(grid_data.line_log_met), log_z_c)
    ia, ta = _frac(np.array(grid_data.line_log_age), log_a_c)
    iu, tu = _frac(np.array(grid_data.line_log_U), log_u_c)

    # Trilinear: interpolate over 8 corners of the (z, a, u) cube
    lum = grid_data.line_luminosity  # (n_met, n_age, n_logU, n_lines)

    def _get(dz, da, du):
        """Retrieve a slice of the grid at the given offset from the interpolation corner."""
        return np.array(lum[iz + dz, ia + da, iu + du, :])

    log_lum = (
        (1 - tz) * (1 - ta) * (1 - tu) * _get(0, 0, 0)
        + tz * (1 - ta) * (1 - tu) * _get(1, 0, 0)
        + (1 - tz) * ta * (1 - tu) * _get(0, 1, 0)
        + (1 - tz) * (1 - ta) * tu * _get(0, 0, 1)
        + tz * ta * (1 - tu) * _get(1, 1, 0)
        + tz * (1 - ta) * tu * _get(1, 0, 1)
        + (1 - tz) * ta * tu * _get(0, 1, 1)
        + tz * ta * tu * _get(1, 1, 1)
    )

    # Convert from log10 to linear
    lin_lum = 10.0**log_lum

    # Normalize to Hbeta = 1.0 (find Hbeta by nearest wavelength to 4861 Å)
    grid_wavs = np.array(grid_data.line_wavelengths)
    hbeta_idx = int(np.argmin(np.abs(grid_wavs - 4862.68)))  # vacuum Hβ
    hbeta_lum = lin_lum[hbeta_idx]
    if hbeta_lum < 1e-30:
        hbeta_lum = 1.0  # safety fallback
    prior_means_all = lin_lum / hbeta_lum

    # Apply dex scatter
    prior_sigmas_all = prior_means_all * (10.0**prior_width_dex - 1.0)
    prior_sigmas_all = np.maximum(prior_sigmas_all, 1e-10)

    # Wrap as jnp arrays
    prior_means_all = jnp.array(prior_means_all)
    prior_sigmas_all = jnp.array(prior_sigmas_all)

    if target_wavelengths is None:
        return prior_means_all, prior_sigmas_all

    # Match target_wavelengths to nearest grid line
    target_wavs = np.array(target_wavelengths)
    diffs = np.abs(target_wavs[:, None] - grid_wavs[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    return prior_means_all[nearest_idx], prior_sigmas_all[nearest_idx]


def balmer_decrement_prior(
    dust_tau_diff: float,
    R_V: float = 4.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predicted Balmer line ratios given nebular dust attenuation.

    Uses the Calzetti (2000) nebular attenuation law to predict the
    observed Balmer decrement. Couples the Balmer line priors to the
    dust parameter being fit, providing a self-consistent constraint.

    Intrinsic Case B ratios (T=10^4 K, n_e=100 cm^-3,
    Osterbrock & Ferland 2006):

    - Hα/Hβ = 2.86
    - Hγ/Hβ = 0.468
    - Hδ/Hβ = 0.259
    - Hε/Hβ = 0.159

    Parameters
    ----------
    dust_tau_diff : float
        Diffuse ISM optical depth at V-band (the ``dust_tau_diff``
        physical parameter in tengri).
    R_V : float
        Total-to-selective extinction ratio. Default 4.05 (Calzetti+2000).

    Returns
    -------
    wavelengths : array, shape (4,)
        Balmer line rest-frame vacuum wavelengths [Hα, Hβ, Hγ, Hδ]
        [Angstrom].
    predicted_ratios : array, shape (4,)
        Predicted observed Balmer ratios relative to Hβ = 1.0 [dimensionless],
        after applying nebular dust attenuation.

    Notes
    -----
    **JIT-compatible**: yes, uses only jnp primitives.
    The Calzetti nebular E(B-V) is related to the diffuse dust optical
    depth by:

        E(B-V)_neb = dust_tau_diff / (R_V * 0.44)

    where the factor 0.44 comes from the Calzetti (2000) prescription
    that nebular emission is more attenuated than stellar continuum:

        E(B-V)_neb = E(B-V)_star / 0.44.

    Balmer attenuation uses the Calzetti law k(λ) evaluated at each
    Balmer wavelength. Relative to Hβ:

        ratio_obs(λ) = ratio_intrinsic(λ) × 10^{-0.4 × [A(λ) - A(Hβ)]}

    References
    ----------
    Calzetti, D. 2000, ApJ, 533, 682, nebular attenuation law.
    Osterbrock, D. E., & Ferland, G. J. 2006, Case B ratios.

    """
    # Intrinsic Case B ratios (Hα, Hβ, Hγ, Hδ) relative to Hβ=1
    intrinsic = jnp.array([2.86, 1.0, 0.468, 0.259])
    wavelengths = jnp.array([6564.61, 4862.68, 4341.68, 4102.89])  # vacuum

    # Calzetti (2000) k(λ) piecewise fit. λ_um = λ_Angstrom / 1e4
    # Red branch [6300, 22000 Å]: k = 2.659(-1.857 + 1.040/λ_um) + R_V
    # Blue branch [1200, 6300 Å]: k = 2.659(-2.156 + 1.509/λ_um - 0.198/λ_um^2
    #                                       + 0.011/λ_um^3) + R_V
    # Hα (6564.61 Å) is above the 6300 Å boundary → red branch.
    lam_um = wavelengths / 1e4
    k_red = 2.659 * (-1.857 + 1.040 / lam_um) + R_V
    k_blue = 2.659 * (-2.156 + 1.509 / lam_um - 0.198 / lam_um**2 + 0.011 / lam_um**3) + R_V
    k_lam = jnp.where(wavelengths > 6300.0, k_red, k_blue)
    k_lam = jnp.maximum(k_lam, 0.0)

    # E(B-V)_neb from dust_tau_diff
    # tau_V = R_V * E(B-V)_star, and tau_diff ≈ tau_V,
    # so E(B-V)_star = tau_diff / R_V
    # E(B-V)_neb = E(B-V)_star / 0.44
    ebv_neb = dust_tau_diff / (R_V * 0.44)

    # Attenuation A(λ) = k(λ) * E(B-V)_neb
    a_lam = k_lam * ebv_neb

    # Hβ index is 1
    a_hbeta = a_lam[1]

    # Relative attenuation factor: 10^{-0.4 * (A(λ) - A(Hβ))}
    rel_atten = 10.0 ** (-0.4 * (a_lam - a_hbeta))

    # Predicted observed ratios
    predicted_ratios = intrinsic * rel_atten

    return wavelengths, predicted_ratios
