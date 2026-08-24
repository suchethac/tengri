# SPDX-License-Identifier: BSD-3-Clause
"""Spectrophotometric calibration polynomials.

When fitting spectra, the observed spectrum has wavelength-dependent
calibration errors from flux calibration, slit losses, and telluric
residuals.  A low-order Chebyshev polynomial corrects this multiplicatively:

    spec_obs(lambda) = C(lambda) * spec_physical(lambda)

where C(lambda) = 1 + sum_{n=1}^{order} a_n * T_n(x) and
x = 2*(lambda - lambda_min)/(lambda_max - lambda_min) - 1 maps wavelengths
to [-1, 1].  The constant term is unity by convention (overall normalization
is handled elsewhere); the fitted coefficients represent *deviations* from
a flat calibration.

Coefficients a_n have a Gaussian(0, sigma) prior that regularizes the
polynomial toward unity, preventing overfitting of broad spectral features.

Parameters integration example
------------------------------
To add calibration coefficients as free parameters::

    from tengri.parameters.priors import Gaussian

    spec = Parameters(
        ...,
        # 3rd-order calibration polynomial (3 free coefficients)
        cal_c1=Gaussian(0.0, 0.1),
        cal_c2=Gaussian(0.0, 0.1),
        cal_c3=Gaussian(0.0, 0.05),
    )

    # In the forward model, pack coefficients and apply:
    coeffs = jnp.array([params["cal_c1"], params["cal_c2"], params["cal_c3"]])
    model_spec = apply_calibration(physical_spec, wave_obs, coeffs, wave_min, wave_max)

References
----------
Johnson et al. (2021), Prospector calibration model.

"""

from functools import partial

import jax
import jax.numpy as jnp

from tengri.utils.scale import representable_floor as _representable_floor, whiten as _whiten

#: Guard against ``obs_err == 0`` only. Expressed in **sigma**, not variance:
#: the previous ``maximum(obs_err**2, 1e-30)`` was a floor on the variance and
#: so bound for every sigma below 1e-15, i.e. every real spectrum (#1588).
#: ``representable_floor`` raises it to the working dtype's smallest normal, so
#: it is never itself a silent zero in float32.
_ERR_FLOOR = 1e-300


@partial(jax.jit, static_argnums=(1,), static_argnames=("order",))
def chebyshev_basis(
    wavelength: jnp.ndarray,
    order: int,
    wave_min: float,
    wave_max: float,
) -> jnp.ndarray:
    """Evaluate Chebyshev polynomial basis at given wavelengths.

    Uses the three-term recurrence relation for numerical stability:
    T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x).

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom].
    order: int
        Maximum polynomial order (returns order+1 basis functions,
        from T_0 through T_order).
    wave_min: float
        Minimum wavelength for normalization to [-1, 1] [Angstrom].
    wave_max: float
        Maximum wavelength for normalization to [-1, 1] [Angstrom].

    Returns
    -------
    ndarray, shape (order+1, n_wave)
        Chebyshev basis functions T_0(x), T_1(x), ..., T_order(x)
        (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes, `order` is a static argument.
    **Gradient-safe**: yes, differentiable w.r.t. wavelength.

    """
    x = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    # Build basis via scan over the recurrence
    def _step(carry, _k):
        """Execute one step of the Chebyshev recurrence relation."""
        t_prev, t_curr = carry
        t_next = 2.0 * x * t_curr - t_prev
        return (t_curr, t_next), t_next

    t0 = jnp.ones_like(x)
    t1 = x

    if order == 0:
        return t0[jnp.newaxis, :]

    # First two basis functions
    init = (t0, t1)
    _, higher = jax.lax.scan(_step, init, jnp.arange(2, order + 1))

    # Stack: T_0, T_1, T_2, ..., T_order
    return jnp.concatenate([t0[jnp.newaxis], t1[jnp.newaxis], higher], axis=0)


@jax.jit
def calibration_polynomial(
    wavelength: jnp.ndarray,
    coeffs: jnp.ndarray,
    wave_min: float,
    wave_max: float,
) -> jnp.ndarray:
    r"""Multiplicative calibration polynomial C(lambda).

    C(lambda) = 1 + sum_{n=1}^{order} a_n * T_n(x)

    The constant term is fixed to 1 (flat calibration); ``coeffs``
    supplies a_1 through a_order representing deviations from unity.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom].
    coeffs: array, shape (order,)
        Chebyshev coefficients a_1, ..., a_order. Empty array gives
        C(lambda) = 1 everywhere.
    wave_min, wave_max: float
        Wavelength range for normalization to [-1, 1].

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative calibration factor (dimensionless).

    Notes
    -----
    JIT-compatible and gradient-safe in both ``coeffs`` and ``wavelength``.

    Evaluated by Clenshaw's backward recurrence on
    :math:`S = c_0 + \sum_{n \ge 1} c_n T_n(x)` with implicit :math:`c_0 = 1`:
    :math:`b_k = c_k + 2x\,b_{k+1} - b_{k+2}` scanned from :math:`k = N`
    down to :math:`k = 1`, then :math:`S = 1 + x\,b_1 - b_2`.

    """
    x = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    def _clenshaw_step(carry, c_k):
        b_kp1, b_kp2 = carry
        return (c_k + 2.0 * x * b_kp1 - b_kp2, b_kp1), None

    init = (jnp.zeros_like(x), jnp.zeros_like(x))
    (b1, b2), _ = jax.lax.scan(_clenshaw_step, init, coeffs[::-1])
    return 1.0 + x * b1 - b2


@partial(jax.jit, static_argnames=("n_poly",))
def marginalize_calibration(
    model_flux: jnp.ndarray,
    obs_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    wavelength: jnp.ndarray,
    n_poly: int = 3,
    prior_sigma: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Analytically marginalize over calibration polynomial coefficients.

    Given a model SED m(lambda) and observed spectrum d(lambda) with noise
    sigma(lambda), the calibrated model is C(lambda)*m(lambda) where C is
    a Chebyshev polynomial. With a Gaussian prior on the polynomial
    coefficients c ~ N(0, prior_sigma^2 I), the optimal coefficients and
    marginalized log-likelihood are computed in closed form.

    This follows the Prospector approach: the calibration polynomial is
    treated as a nuisance and integrated out analytically at each likelihood
    evaluation, reducing the dimensionality of the sampling problem.

    Parameters
    ----------
    model_flux: array, shape (n_wave,)
        Physical model spectrum (any flux units, matching obs_flux).
    obs_flux: array, shape (n_wave,)
        Observed spectrum (same units as model_flux).
    obs_err: array, shape (n_wave,)
        1-sigma uncertainties on the observed spectrum (same units).
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom].
    n_poly: int, optional
        Number of Chebyshev polynomial coefficients (order 1 through
        n_poly). The constant term (T_0 = 1) is implicit and fixed.
        Default 3.
    prior_sigma: float, optional
        Standard deviation of the Gaussian prior on each coefficient.
        Default 1.0.

    Returns
    -------
    log_likelihood_marginal: scalar
        Marginalized log-likelihood with calibration polynomial
        integrated out.
    c_hat: ndarray, shape (n_poly,)
        MAP calibration coefficients (a_1 ... a_n_poly).
    c_hat_err: ndarray, shape (n_poly,)
        Posterior standard deviations of the coefficients.

    Notes
    -----
    JIT-compatible: yes, `n_poly` is a static argument.
    Gradient-safe: no, uses matrix inversion (not safe for gradient).

    The marginalized log-likelihood integrates the Gaussian likelihood and
    Gaussian prior on the polynomial coefficients in closed form via the
    matrix determinant lemma.

    References
    ----------
    Johnson et al. 2021, ApJS, 254, 22 (Prospector).

    """
    wave_min = wavelength[0]
    wave_max = wavelength[-1]

    # Normalized coordinate x in [-1, 1]
    x = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    # Build Chebyshev basis T_1(x) ... T_{n_poly}(x) via recurrence
    # T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}
    def _cheb_step(carry, _k):
        """Execute one step of the Chebyshev recurrence to build basis functions."""
        t_prev, t_curr = carry
        t_next = 2.0 * x * t_curr - t_prev
        return (t_curr, t_next), t_next

    t0 = jnp.ones_like(x)
    t1 = x
    init = (t0, t1)

    # We need T_1 through T_{n_poly}. T_1 = x is already computed.
    # Scan generates T_2 ... T_{n_poly}.
    _, higher = jax.lax.scan(_cheb_step, init, jnp.arange(2, n_poly + 1))
    # basis shape: (n_poly, n_wave), rows are T_1, T_2, ..., T_{n_poly}
    basis = jnp.concatenate([t1[jnp.newaxis], higher], axis=0)

    # Whitened weights. NEVER form 1/sigma**2 (#1588): it is ~1e59 at a real
    # flux uncertainty, and the variance-domain floor this replaced
    # (``maximum(obs_err**2, 1e-30)``) bound on *every pixel of every realistic
    # spectrum*, sigma < 1e-15 trips it, pinning inv_var to exactly 1e30 in
    # **float64**. The data term then lost to the prior and the recovered
    # polynomial collapsed toward zero: c_hat[0] 5.0e-02 -> 7.5e-04 at
    # F_lambda ~1e-17, and -> 7.6e-26 at F_nu ~1e-28. Guard only against
    # sigma == 0, in the sigma domain, at the working dtype's smallest normal.
    err_safe = jnp.maximum(obs_err, _representable_floor(_ERR_FLOOR))

    # Design matrix: D_jk = sum_i [T_j(x_i) * m_i * T_k(x_i) * m_i / sigma_i^2]
    # = sum_i [(T_j * m/sigma) * (T_k * m/sigma)]
    weighted_model = _whiten(model_flux, err_safe)  # m / sigma
    # B_{j,i} = T_j(x_i) * m(x_i) / sigma(x_i)
    b_matrix = basis * weighted_model[jnp.newaxis, :]  # (n_poly, n_wave)

    # A = B @ B^T + (1/prior_sigma^2) * I
    a_matrix = b_matrix @ b_matrix.T  # (n_poly, n_poly)
    prior_precision = 1.0 / (prior_sigma**2)
    a_matrix = a_matrix + prior_precision * jnp.eye(a_matrix.shape[0])

    # Residual vector: r_j = sum_i [T_j(x_i) * m(x_i) * (d_i - m_i) / sigma_i^2]
    residual = obs_flux - model_flux
    # r_j = sum_i [T_j * (m/sigma) * ((d-m)/sigma)], algebraically the same as
    # T_j * m * (d-m) / sigma^2, without ever forming 1/sigma^2.
    rhs = jnp.sum(
        basis * weighted_model[jnp.newaxis, :] * _whiten(residual, err_safe)[jnp.newaxis, :],
        axis=1,
    )  # (n_poly,)

    # Solve for MAP coefficients: A c_hat = rhs
    c_hat = jax.numpy.linalg.solve(a_matrix, rhs)

    # Posterior covariance Sigma_post = A^{-1}
    # For c_hat_err we need diag(A^{-1}), computed via solve against identity
    sigma_post = jax.numpy.linalg.solve(a_matrix, jnp.eye(a_matrix.shape[0]))
    c_hat_err = jnp.sqrt(jnp.maximum(jnp.diag(sigma_post), 0.0))

    # --- Marginalized log-likelihood ---

    # Calibrated model at c_hat
    cal_poly = 1.0 + jnp.dot(c_hat, basis)  # (n_wave,)
    model_cal = cal_poly * model_flux
    chi2 = jnp.sum(_whiten(obs_flux - model_cal, err_safe) ** 2)

    # Prior penalty: c_hat^T Lambda^{-1} c_hat
    prior_penalty = prior_precision * jnp.dot(c_hat, c_hat)

    # Log-determinant terms
    # ln|Sigma_post| = -ln|A|, ln|Lambda| = n_poly * ln(prior_sigma^2)
    _sign_a, logdet_a = jnp.linalg.slogdet(a_matrix)
    log_det_sigma_post = -logdet_a  # ln|A^{-1}|
    log_det_lambda = n_poly * jnp.log(prior_sigma**2)

    # Constant normalization: -N/2 * ln(2pi) - sum(ln sigma_i)
    n_wave = wavelength.shape[0]
    # ``err_safe``, not ``obs_err``: the same zero the whitening guards would
    # otherwise make this term -inf while chi2 stayed finite.
    log_norm = -0.5 * n_wave * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(err_safe))

    log_likelihood_marginal = log_norm - 0.5 * (
        chi2 + prior_penalty - log_det_sigma_post + log_det_lambda
    )

    return log_likelihood_marginal, c_hat, c_hat_err


@jax.jit
def apply_calibration(
    spectrum: jnp.ndarray,
    wavelength: jnp.ndarray,
    coeffs: jnp.ndarray,
    wave_min: float,
    wave_max: float,
) -> jnp.ndarray:
    """Apply calibration polynomial to a spectrum.

    Returns spectrum * C(lambda), where C(lambda) is the Chebyshev
    calibration polynomial evaluated at the given wavelengths.

    Parameters
    ----------
    spectrum: array, shape (n_wave,)
        Physical model spectrum (any flux units).
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom].
    coeffs: array, shape (order,)
        Chebyshev coefficients a_1, ..., a_order.
    wave_min, wave_max: float
        Wavelength range for normalization to [-1, 1].

    Returns
    -------
    ndarray, shape (n_wave,)
        Calibrated spectrum (same units as input spectrum).

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes, differentiable w.r.t.
    spectrum and coefficients.

    """
    cal = calibration_polynomial(wavelength, coeffs, wave_min, wave_max)
    return spectrum * cal


@jax.jit
def double_calibration_polynomial(
    wavelength: jnp.ndarray,
    coeffs_blue: jnp.ndarray,
    coeffs_red: jnp.ndarray,
    wave_split: float,
) -> jnp.ndarray:
    """Piecewise calibration: independent Chebyshev polynomials for blue and red halves.

    For two-arm spectrographs (e.g. X-SHOOTER, DEIMOS) where a detector
    gap or dichroic split causes a calibration discontinuity, fitting a
    single polynomial across the full range can bias the result. This
    function applies separate polynomials to each half, matched to their
    local wavelength range.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom], must be sorted.
    coeffs_blue: array, shape (order_blue,)
        Chebyshev coefficients for the blue arm (wavelength < wave_split).
    coeffs_red: array, shape (order_red,)
        Chebyshev coefficients for the red arm (wavelength >= wave_split).
    wave_split: float
        Wavelength boundary between blue and red arms [Angstrom].

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative calibration factor (dimensionless).

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes, differentiable w.r.t.
    coefficients and wavelengths.

    """
    wave_min = wavelength[0]
    wave_max = wavelength[-1]

    cal_blue = calibration_polynomial(wavelength, coeffs_blue, wave_min, wave_split)
    cal_red = calibration_polynomial(wavelength, coeffs_red, wave_split, wave_max)

    is_blue = wavelength < wave_split
    return jnp.where(is_blue, cal_blue, cal_red)


@jax.jit
def apply_double_calibration(
    spectrum: jnp.ndarray,
    wavelength: jnp.ndarray,
    coeffs_blue: jnp.ndarray,
    coeffs_red: jnp.ndarray,
    wave_split: float,
) -> jnp.ndarray:
    """Apply piecewise calibration polynomial to a spectrum.

    Parameters
    ----------
    spectrum: array, shape (n_wave,)
        Physical model spectrum (any flux units).
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom], must be sorted.
    coeffs_blue: array, shape (order_blue,)
        Chebyshev coefficients for the blue arm (wavelength < wave_split).
    coeffs_red: array, shape (order_red,)
        Chebyshev coefficients for the red arm (wavelength >= wave_split).
    wave_split: float
        Wavelength boundary [Angstrom].

    Returns
    -------
    ndarray, shape (n_wave,)
        Calibrated spectrum (same units as input spectrum).

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes, differentiable w.r.t.
    spectrum and coefficients.

    """
    cal = double_calibration_polynomial(wavelength, coeffs_blue, coeffs_red, wave_split)
    return spectrum * cal
