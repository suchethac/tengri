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

# -------------------------------------------------------------------------
# Default emission-line list (rest-frame vacuum wavelengths in Angstrom)
# -------------------------------------------------------------------------

DEFAULT_LINE_NAMES = (
    "Ly-alpha",
    "H-delta",
    "H-gamma",
    "H-beta",
    "[OIII]4959",
    "[OIII]5007",
    "H-alpha",
    "[NII]6548",
    "[NII]6583",
    "[OII]3726",
    "[OII]3729",
    "[SII]6717",
    "[SII]6731",
)

DEFAULT_LINE_WAVELENGTHS = jnp.array(
    [
        1215.67,  # Ly-alpha
        4101.73,  # H-delta
        4340.46,  # H-gamma
        4861.33,  # H-beta
        4958.91,  # [OIII]4959
        5006.84,  # [OIII]5007
        6562.80,  # H-alpha
        6548.05,  # [NII]6548
        6583.45,  # [NII]6583
        3726.03,  # [OII]3726
        3728.82,  # [OII]3729
        6716.44,  # [SII]6717
        6730.81,  # [SII]6731
    ]
)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


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

    Each column *j* is a normalized Gaussian profile centred at the
    redshifted line wavelength::

        G_j(lambda) = exp(-0.5 * ((lambda - lam_j*(1+z)) / sig_j)^2)
                      / (sqrt(2*pi) * sig_j)

    where ``sig_j = sqrt(sig_inst^2 + sig_vel^2)`` combines instrument
    resolution and intrinsic velocity broadening. The profile can also
    be offset by a velocity shift (eline_delta_v_kms).

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid (Angstrom).
    line_wavelengths : array, shape (n_lines,)
        Rest-frame vacuum wavelengths of the lines (Angstrom).
    spectral_resolution : float
        Instrument spectral resolution R = lambda / delta_lambda.
    redshift : float
        Source redshift.
    eline_sigma_kms : float
        Intrinsic velocity broadening (km/s). Default 0.
    eline_delta_v_kms : float
        Velocity offset from systemic (km/s). Default 0.

    Returns
    -------
    array, shape (n_pix, n_lines)
        Design matrix *G*.
    """
    return _build_eline_design_matrix_jitted(
        wave_obs,
        line_wavelengths,
        spectral_resolution,
        redshift,
        eline_sigma_kms,
        eline_delta_v_kms,
    )


@jax.jit
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
        Observed-frame wavelength grid (Angstrom).
    line_wavelengths : array, shape (n_lines,)
        Rest-frame vacuum wavelengths of broad-line candidates (Angstrom).
    spectral_resolution : float
        Instrument spectral resolution R = lambda/delta_lambda.
    redshift : float
        Source redshift.
    broad_sigma_kms : float
        Broad component velocity dispersion (km/s), typically 500-5000.
    eline_delta_v_kms : float
        Velocity offset of broad component from systemic (km/s). Default 0.

    Returns
    -------
    array, shape (n_pix, n_lines)
        Broad-component design matrix G_broad.
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
        Constraint matrix from ``LineCatalog.build_constraint_matrix()``.
        Encodes doublet ratios as a linear transformation.

    Returns
    -------
    array, shape (n_pix, n_independent)
        Constrained design matrix. Marginalizing over ``n_independent``
        amplitudes automatically enforces all doublet ratio constraints.

    Examples
    --------
    ::

        cat = LineCatalog.default_optical()
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
        Constraint matrix from ``LineCatalog.build_constraint_matrix()``.

    Returns
    -------
    a_hat_full : array, shape (n_lines,)
        Amplitudes for all lines including constrained doublet secondaries.
    a_cov_full : array, shape (n_lines, n_lines)
        Full covariance, propagated through the constraint matrix:
        ``C @ a_cov @ C.T``.

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
    prior_variance: jnp.ndarray = None,
) -> tuple:
    """Analytically marginalize emission-line amplitudes.

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
        Marginalized log-likelihood.
    a_hat : array, shape (n_lines,)
        Posterior-mean (optimal) line amplitudes.
    a_cov : array, shape (n_lines, n_lines)
        Posterior covariance of line amplitudes.
    """
    if prior_variance is None:
        prior_variance = jnp.array(1e10)
    n_lines = design_matrix.shape[1]

    # Inverse noise variance, shape (n_pix,)
    n_inv = 1.0 / noise**2

    # G^T N^{-1} G  — (n_lines, n_lines)
    g_weighted = design_matrix * n_inv[:, None]  # (n_pix, n_lines)
    gt_ninv_g = g_weighted.T @ design_matrix

    # G^T N^{-1} r  — (n_lines,)
    gt_ninv_r = g_weighted.T @ residual

    # Prior precision: Lambda^{-1}
    prior_variance = jnp.broadcast_to(jnp.atleast_1d(prior_variance), (n_lines,))
    lambda_inv = jnp.diag(1.0 / prior_variance)

    # Posterior covariance: Sigma_a = (G^T N^{-1} G + Lambda^{-1})^{-1}
    a_cov = jnp.linalg.inv(gt_ninv_g + lambda_inv)

    # Posterior mean (optimal amplitudes)
    a_hat = a_cov @ gt_ninv_r

    # --- Marginalized log-likelihood ---
    # chi2 of residual alone (no lines)
    chi2_continuum = jnp.sum(residual**2 * n_inv)

    # Improvement from fitting lines
    chi2_marg = chi2_continuum - a_hat @ gt_ninv_g @ a_hat

    # Prior penalty
    prior_penalty = jnp.sum(a_hat**2 / prior_variance)

    # Log-determinant correction:
    # 0.5 * (ln|Sigma_a| - ln|Lambda|)
    # = 0.5 * (slogdet(Sigma_a) - sum(ln(prior_variance)))
    _, logdet_sigma = jnp.linalg.slogdet(a_cov)
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
    array, shape (n_pix,)
        Full model: ``model_continuum + G @ a_hat``.
    """
    return model_continuum + design_matrix @ a_hat
