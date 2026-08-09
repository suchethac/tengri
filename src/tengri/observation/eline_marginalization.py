# SPDX-License-Identifier: BSD-3-Clause
"""Analytical emission-line marginalization for spectral fitting.

When fitting spectra, individual emission-line amplitudes are nuisance
parameters.  Rather than sampling them, we analytically marginalize them
out of the likelihood under a Gaussian prior, following the approach in
Prospector (Johnson et al. 2021).

Given observed data *d*, a continuum model *m*, noise *sigma*, and a
design matrix *G* whose columns are Gaussian line profiles, the model
with free line amplitudes is::

    model = m + G @ a

The marginalized log-likelihood integrates out *a* analytically:

    ln L_marg = -0.5 * (chi2_marg + prior_penalty - log_det_correction)

where chi2_marg accounts for the improvement from fitting lines, and the
prior penalty and determinant correction come from the Gaussian prior on
line amplitudes.

All functions are pure JAX, JIT-compatible, and differentiable w.r.t.
the continuum model (enabling gradient-based inference of physical
parameters).
"""

import jax
import jax.numpy as jnp

from tengri.observation.eline_catalog import (
    DEFAULT_LINE_NAMES,  # noqa: F401 — re-exported for convenience
    DEFAULT_LINE_WAVELENGTHS,  # noqa: F401 — re-exported for convenience
)
from tengri.utils.scale import whiten

# ── Public API ────────────────────────────────────────────────────


@jax.jit
def _build_eline_design_matrix_jitted(
    wave_obs: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    spectral_resolution: float,
    redshift: float,
    eline_sigma_kms: float,
    eline_delta_v_kms: float,
) -> jnp.ndarray:
    """Internal JIT-compiled implementation of build_eline_design_matrix."""
    c_kms = 299792.458

    def _single_column(lam_rest):
        """Build a normalized Gaussian profile for a single emission line."""
        lam_obs = lam_rest * (1.0 + redshift)
        # Velocity offset: convert km/s to wavelength shift
        delta_lam = lam_obs * eline_delta_v_kms / c_kms
        lam_center = lam_obs + delta_lam
        # Instrument resolution
        sigma_inst = lam_obs / (2.355 * spectral_resolution)
        # Intrinsic velocity broadening
        sigma_vel = lam_obs * eline_sigma_kms / c_kms
        # Combined sigma
        sigma = jnp.sqrt(sigma_inst**2 + sigma_vel**2)
        profile = jnp.exp(-0.5 * ((wave_obs - lam_center) / sigma) ** 2) / (
            jnp.sqrt(2.0 * jnp.pi) * sigma
        )
        return profile

    # vmap over lines -> (n_lines, n_pix), then transpose -> (n_pix, n_lines)
    columns = jax.vmap(_single_column)(line_wavelengths)
    return columns.T


def build_eline_design_matrix(
    wave_obs: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    spectral_resolution: float,
    redshift: float,
    eline_sigma_kms: float = 0.0,
    eline_delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Build the (n_pix, n_lines) Gaussian design matrix.

    Each column *j* is a normalized Gaussian profile centered at the
    redshifted line wavelength::

        G_j(lambda) = exp(-0.5 * ((lambda - lam_j*(1+z)) / sig_j)^2)
                      / (sqrt(2*pi) * sig_j)

    where ``sig_j = sqrt(sig_inst^2 + sig_vel^2)`` combines instrument
    resolution and intrinsic velocity broadening. The profile can also
    be offset by a velocity shift (eline_delta_v_kms).

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    line_wavelengths : array, shape (n_lines,)
        Rest-frame vacuum wavelengths of the lines [Angstrom].
    spectral_resolution : float
        Instrument spectral resolution R = lambda / delta_lambda
        [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    eline_sigma_kms : float
        Intrinsic velocity broadening [km/s]. Default 0.
    eline_delta_v_kms : float
        Velocity offset from systemic [km/s]. Default 0.

    Returns
    -------
    array, shape (n_pix, n_lines)
        Design matrix *G* [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — delegates to ``_build_eline_design_matrix_jitted``
    which is JIT-decorated. Differentiable w.r.t. all parameters.

    Each profile is normalized to unit integral (``int G_j dlam = 1``) to
    enable proper line amplitude marginalization in
    :func:`marginalize_emission_lines`.
    """
    return _build_eline_design_matrix_jitted(
        wave_obs,
        line_wavelengths,
        spectral_resolution,
        redshift,
        eline_sigma_kms,
        eline_delta_v_kms,
    )


def build_broad_design_matrix(
    wave_obs: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    spectral_resolution: float,
    redshift: float,
    broad_sigma_kms: float,
    eline_delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Build design matrix for broad emission line component.

    Same as build_eline_design_matrix but uses broad_sigma_kms as the
    intrinsic velocity dispersion (typically 500-5000 km/s for AGN BLR).
    Instrument resolution is still added in quadrature.

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    line_wavelengths : array, shape (n_lines,)
        Rest-frame vacuum wavelengths of broad-line candidates [Angstrom].
    spectral_resolution : float
        Instrument spectral resolution R = lambda/delta_lambda
        [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    broad_sigma_kms : float
        Broad component velocity dispersion [km/s]. Typical: 500–5000.
    eline_delta_v_kms : float
        Velocity offset of broad component from systemic [km/s]. Default 0.

    Returns
    -------
    array, shape (n_pix, n_lines)
        Broad-component design matrix G_broad [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — delegates to :func:`build_eline_design_matrix`
    and is itself JIT-decorated. Differentiable w.r.t. all parameters.

    Used for modeling broad AGN emission lines (e.g., broad H-alpha, H-beta)
    in conjunction with narrow lines from :func:`build_eline_design_matrix`.
    """
    # Delegate to build_eline_design_matrix with broad_sigma_kms as eline_sigma_kms
    return build_eline_design_matrix(
        wave_obs,
        line_wavelengths,
        spectral_resolution,
        redshift,
        eline_sigma_kms=broad_sigma_kms,
        eline_delta_v_kms=eline_delta_v_kms,
    )


# Apply JIT
build_broad_design_matrix = jax.jit(build_broad_design_matrix)


def apply_doublet_constraints(
    design_matrix: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Apply doublet ratio constraints to the design matrix.

    Reduces the design matrix from shape ``(n_pix, n_lines)`` to
    ``(n_pix, n_independent)`` by enforcing fixed flux ratios between
    doublet pairs (e.g., [OIII] 5007/4959 = 2.98).

    Parameters
    ----------
    design_matrix : array, shape (n_pix, n_lines)
        Full design matrix with one column per emission line.
    constraint_matrix : array, shape (n_lines, n_independent)
        Constraint matrix from ``LineList.build_constraint_matrix()``.
        Encodes doublet ratios as a linear transformation.

    Returns
    -------
    array, shape (n_pix, n_independent)
        Constrained design matrix. Marginalizing over ``n_independent``
        amplitudes automatically enforces all doublet ratio constraints.

    Notes
    -----
    **JIT-compatible**: yes — pure matrix multiplication via ``jnp.dot``.

    Examples
    --------
    ::

        cat = LineList.default_optical()
        C = cat.build_constraint_matrix()
        G = build_eline_design_matrix(wave_obs, cat.wavelengths, R, z)
        G_eff = apply_doublet_constraints(G, C)  # (n_pix, n_independent)

    """
    return design_matrix @ constraint_matrix


def expand_constrained_amplitudes(
    a_hat: jnp.ndarray,
    a_cov: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Expand independent amplitudes back to full line set.

    After marginalization with a constrained design matrix, the posterior
    amplitudes have shape ``(n_independent,)``. This function expands them
    back to ``(n_lines,)`` by applying the constraint matrix, giving the
    flux for every emission line including constrained doublet secondaries.

    Parameters
    ----------
    a_hat : array, shape (n_independent,)
        Posterior-mean amplitudes for the independent parameters.
    a_cov : array, shape (n_independent, n_independent)
        Posterior covariance for the independent parameters.
    constraint_matrix : array, shape (n_lines, n_independent)
        Constraint matrix from ``LineList.build_constraint_matrix()``.

    Returns
    -------
    a_hat_full : ndarray, shape (n_lines,)
        Amplitudes for all lines including constrained doublet secondaries
        [erg/s] or [erg/s/Angstrom] depending on line fitting context.
    a_cov_full : ndarray, shape (n_lines, n_lines)
        Full covariance, propagated through the constraint matrix:
        ``C @ a_cov @ C.T``.

    Notes
    -----
    **JIT-compatible**: yes — matrix multiplications via ``jnp.dot``.

    Examples
    --------
    ::

        G_eff = apply_doublet_constraints(G, C)
        ln_l, a_hat_ind, a_cov_ind = marginalize_emission_lines(resid, noise, G_eff)
        a_hat_full, a_cov_full = expand_constrained_amplitudes(a_hat_ind, a_cov_ind, C)

    """
    a_hat_full = constraint_matrix @ a_hat
    a_cov_full = constraint_matrix @ a_cov @ constraint_matrix.T
    return a_hat_full, a_cov_full


def marginalize_emission_lines(
    residual: jnp.ndarray,
    noise: jnp.ndarray,
    design_matrix: jnp.ndarray,
    prior_variance: jnp.ndarray | None = None,
) -> tuple:
    r"""Analytically marginalize emission-line amplitudes.

    Computes the marginalized log-likelihood by integrating out the
    linear emission-line amplitude vector under a Gaussian prior.

    Parameters
    ----------
    residual : array, shape (n_pix,)
        Data minus continuum model: ``d - m``.
    noise : array, shape (n_pix,)
        Per-pixel noise standard deviation (sigma).
    design_matrix : array, shape (n_pix, n_lines)
        Gaussian design matrix from :func:`build_eline_design_matrix`.
    prior_variance : scalar or array, shape (n_lines,)
        Prior variance on line amplitudes.  Large values (default 1e10)
        give an uninformative prior.

    Returns
    -------
    ln_L_marg : scalar
        Marginalized log-likelihood (dimensionless).
    a_hat : ndarray, shape (n_lines,)
        Posterior-mean (optimal) line amplitudes [same units as residual].
    a_cov : ndarray, shape (n_lines, n_lines)
        Posterior covariance of line amplitudes.

    Notes
    -----
    **JIT-compatible**: yes — all operations via ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere w.r.t. ``residual``,
    ``noise``, and ``design_matrix``, in float32 as well as float64.

    **Numerical stability (#1206)**: the normal equations are assembled from
    the *whitened* design matrix :math:`\tilde G = G/\sigma` rather than from
    :math:`N^{-1} = 1/\sigma^2`,

    .. math::

        G^\mathsf{T} N^{-1} G = \tilde G^\mathsf{T} \tilde G,
        \qquad
        G^\mathsf{T} N^{-1} r = \tilde G^\mathsf{T} \tilde r,
        \qquad
        \chi^2_{\rm cont} = \tilde r^\mathsf{T} \tilde r

    with :math:`\tilde r = r/\sigma`. At a real spectroscopic
    :math:`\sigma \sim 3\times10^{-30}` the quantity :math:`1/\sigma^2` is
    ~1e59, outside the float32 ceiling of 3.4e38, so the previous spelling
    made every one of ``ln_L_marg``, ``a_hat`` and ``a_cov`` — and the
    gradient — ``NaN``, contradicting the promise above. Identical in float64.

    """
    n_lines = design_matrix.shape[1]

    g_whitened = whiten(design_matrix, noise[:, None])
    r_whitened = whiten(residual, noise)
    gt_ninv_g = g_whitened.T @ g_whitened
    gt_ninv_r = g_whitened.T @ r_whitened

    # Flat prior (~uninformative) when prior_variance is omitted.
    _pv = prior_variance if prior_variance is not None else jnp.full((n_lines,), 1e10)
    prior_variance = jnp.broadcast_to(jnp.atleast_1d(_pv), (n_lines,))
    lambda_inv = jnp.diag(1.0 / prior_variance)

    # Posterior covariance: Sigma_a = (G^T N^{-1} G + Lambda^{-1})^{-1}
    a_cov = jnp.linalg.inv(gt_ninv_g + lambda_inv)

    # Posterior mean (optimal amplitudes)
    a_hat = a_cov @ gt_ninv_r

    chi2_continuum = jnp.sum(r_whitened**2)
    chi2_marg = chi2_continuum - a_hat @ gt_ninv_g @ a_hat
    prior_penalty = jnp.sum(a_hat**2 / prior_variance)
    _sign, logdet_sigma = jnp.linalg.slogdet(a_cov)
    log_det_correction = logdet_sigma - jnp.sum(jnp.log(prior_variance))
    ln_l_marg = -0.5 * chi2_marg - 0.5 * prior_penalty + 0.5 * log_det_correction

    return ln_l_marg, a_hat, a_cov


# JIT-compiled version (prior_variance must be traced, not static)
marginalize_emission_lines = jax.jit(marginalize_emission_lines)


@jax.jit
def predict_with_marginalized_lines(
    model_continuum: jnp.ndarray,
    design_matrix: jnp.ndarray,
    a_hat: jnp.ndarray,
) -> jnp.ndarray:
    """Return the full model including optimized emission lines.

    Parameters
    ----------
    model_continuum : array, shape (n_pix,)
        Continuum-only model spectrum.
    design_matrix : array, shape (n_pix, n_lines)
        Gaussian design matrix.
    a_hat : array, shape (n_lines,)
        Optimal line amplitudes from :func:`marginalize_emission_lines`.

    Returns
    -------
    ndarray, shape (n_pix,)
        Full model spectrum including continuum and optimized emission lines
        [erg/s/Hz] or [erg/s/Angstrom] depending on wavelength unit.

    Notes
    -----
    **JIT-compatible**: yes — matrix multiplication via ``jnp.dot``.

    **Gradient-safe**: yes — fully differentiable.

    """
    return model_continuum + design_matrix @ a_hat


def build_line_design_matrix(
    wave_obs: jnp.ndarray,
    narrow_wavelengths: jnp.ndarray,
    broad_wavelengths: jnp.ndarray | None = None,
    spectral_resolution: float = 2000.0,
    redshift: float = 0.0,
    narrow_sigma_kms: float = 0.0,
    broad_sigma_kms: float = 5000.0,
    delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Unified emission line design matrix for narrow and/or broad lines.

    Returns a ``(n_pix, n_narrow + n_broad)`` design matrix suitable for
    analytical marginalization via ``marginalize_emission_lines()``.

    Narrow line columns come first, broad line columns follow. If no
    ``broad_wavelengths`` are provided, returns only the narrow columns.

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid [Angstrom].
    narrow_wavelengths : array, shape (n_narrow,)
        Rest-frame narrow line wavelengths [Angstrom].
    broad_wavelengths : array or None, shape (n_broad,)
        Rest-frame broad line wavelengths [Angstrom].
        If ``None``, only narrow columns are returned.
    spectral_resolution : float
        Spectral resolution R = lambda/delta_lambda. Default 2000.
    redshift : float
        Redshift for shifting lines to observed frame. Default 0.
    narrow_sigma_kms : float
        Intrinsic narrow line width [km/s]. Default 0 (instrument-limited).
    broad_sigma_kms : float
        Intrinsic broad line width [km/s]. Default 5000.
    delta_v_kms : float
        Systematic velocity offset [km/s]. Default 0.

    Returns
    -------
    ndarray, shape (n_pix, n_narrow [+ n_broad])
        Design matrix. Each column is a normalized Gaussian profile
        [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — delegates to :func:`build_eline_design_matrix` and
    :func:`build_broad_design_matrix`, which are JIT-decorated.

    If ``broad_wavelengths`` is ``None``, returns only the narrow-line portion.

    Examples
    --------
    >>> A = build_line_design_matrix(wave_obs, DEFAULT_LINE_WAVELENGTHS, redshift=0.1)

    """
    A_narrow = build_eline_design_matrix(
        wave_obs,
        narrow_wavelengths,
        spectral_resolution=spectral_resolution,
        redshift=redshift,
        eline_sigma_kms=narrow_sigma_kms,
        eline_delta_v_kms=delta_v_kms,
    )

    if broad_wavelengths is None:
        return A_narrow

    A_broad = build_broad_design_matrix(
        wave_obs,
        broad_wavelengths,
        spectral_resolution=spectral_resolution,
        redshift=redshift,
        broad_sigma_kms=broad_sigma_kms,
        eline_delta_v_kms=delta_v_kms,
    )

    return jnp.concatenate([A_narrow, A_broad], axis=1)
