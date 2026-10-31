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

from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.utils.scale import representable_floor as _representable_floor

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
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom].
    order : int
        Maximum polynomial order (returns order+1 basis functions,
        from T_0 through T_order).
    wave_min : float
        Minimum wavelength for normalization to [-1, 1] [Angstrom].
    wave_max : float
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
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom].
    coeffs : array, shape (order,)
        Chebyshev coefficients a_1, ..., a_order. Empty array gives
        C(lambda) = 1 everywhere.
    wave_min, wave_max : float
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
def calibration_basis(
    wave: jnp.ndarray,
    wave_min: float,
    wave_max: float,
    n_poly: int,
) -> jnp.ndarray:
    r"""Chebyshev basis T_1 through T_{n_poly} on a calibration wavelength grid.

    Column :math:`k` of the result is :math:`T_k(x)` with
    :math:`x = (2\lambda - \lambda_\min - \lambda_\max)/(\lambda_\max - \lambda_\min)`,
    so that :math:`1 + B\,c` equals :func:`calibration_polynomial` for coefficients
    :math:`c = (a_1, \ldots, a_{n})`.

    Parameters
    ----------
    wave : array_like, shape (n,)
        Wavelength grid [Angstrom].
    wave_min : float
        Lower bound of the normalization interval [Angstrom].
    wave_max : float
        Upper bound of the normalization interval [Angstrom].
    n_poly : int
        Highest polynomial order, at least 1. Static under JIT.

    Returns
    -------
    ndarray, shape (n, n_poly)
        Basis matrix :math:`[T_1(x), \ldots, T_{n_\mathrm{poly}}(x)]` (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes. ``n_poly`` is static and the recurrence is a Python loop
    over it, so no ``lax.scan`` is traced.

    **Gradient-safe**: yes, differentiable with respect to ``wave``, ``wave_min`` and
    ``wave_max``.

    The recurrence is :math:`T_0 = 1`, :math:`T_1 = x`,
    :math:`T_{k+1} = 2x\,T_k - T_{k-1}`.

    """
    x = 2.0 * (wave - wave_min) / (wave_max - wave_min) - 1.0
    columns = [x]
    t_prev, t_curr = jnp.ones_like(x), x
    for _ in range(n_poly - 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
        columns.append(t_curr)
    return jnp.stack(columns, axis=-1)


@partial(jax.jit, static_argnames=("n_poly",))
def marginalize_calibration(
    model_flux: jnp.ndarray,
    obs_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    wavelength: jnp.ndarray,
    n_poly: int = 3,
    prior_sigma: float = 1.0,
    *,
    pix_mask: jnp.ndarray | None = None,
    wave_range: tuple[float, float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Analytically marginalize over calibration polynomial coefficients.

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
    model_flux : array, shape (n_wave,)
        Physical model spectrum (any flux units, matching obs_flux).
    obs_flux : array, shape (n_wave,)
        Observed spectrum (same units as model_flux).
    obs_err : array, shape (n_wave,)
        1-sigma uncertainties on the observed spectrum (same units).
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom].
    n_poly : int, optional
        Number of Chebyshev polynomial coefficients (order 1 through
        n_poly). The constant term (T_0 = 1) is implicit and fixed.
        Default 3.
    prior_sigma : float, optional
        Standard deviation of the Gaussian prior on each coefficient.
        Default 1.0.
    pix_mask : array_like, shape (n_wave,), optional, keyword-only
        Pixel mask in {0, 1}. Masked pixels are absent from every term. Default: all
        pixels used.
    wave_range : tuple of float, optional, keyword-only
        ``(wave_min, wave_max)`` [Angstrom] that normalize the Chebyshev coordinate.
        Default: ``(wavelength.min(), wavelength.max())``.

    Returns
    -------
    log_likelihood_marginal : scalar
        Marginalized log-likelihood with calibration polynomial
        integrated out.
    c_hat : ndarray, shape (n_poly,)
        MAP calibration coefficients (a_1 ... a_n_poly).
    c_hat_err : ndarray, shape (n_poly,)
        Posterior standard deviations of the coefficients.

    Notes
    -----
    JIT-compatible: yes, `n_poly` is a static argument.
    Gradient-safe: yes, inherited from
    :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`.

    The calibration polynomial is the nuisance design
    :math:`X = m(\lambda)\,T(x)` with :math:`T` from :func:`calibration_basis`, and
    the marginal is :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`
    with prior precision :math:`I/\sigma_{\rm prior}^2` and prior mean zero. The
    sigma floor, :math:`\sigma \ge` ``representable_floor(1e-300)``, is applied here,
    before whitening; the primitive itself applies no floor.

    ``wave_range=None`` uses the grid extremes. On a monotonic grid these are the
    first and last wavelengths.

    References
    ----------
    Johnson et al. 2021, ApJS, 254, 22 (Prospector).

    """
    if wave_range is None:
        wave_min, wave_max = wavelength.min(), wavelength.max()
    else:
        wave_min, wave_max = wave_range

    basis = calibration_basis(wavelength, wave_min, wave_max, n_poly)

    # Guard only against sigma == 0, in the sigma domain, at the working dtype's
    # smallest normal. Never a variance floor (#1588).
    err_safe = jnp.maximum(obs_err, _representable_floor(_ERR_FLOOR))

    result = marginalize_linear_nuisance(
        obs_flux,
        model_flux,
        model_flux[:, jnp.newaxis] * basis,
        sigma=err_safe,
        pix_mask=pix_mask,
        prior_precision=jnp.eye(n_poly) / prior_sigma**2,
    )
    c_hat_err = jnp.sqrt(jnp.diag(result.coef_cov))
    return result.log_like, result.coef_hat, c_hat_err


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
    spectrum : array, shape (n_wave,)
        Physical model spectrum (any flux units).
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom].
    coeffs : array, shape (order,)
        Chebyshev coefficients a_1, ..., a_order.
    wave_min, wave_max : float
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
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom], must be sorted.
    coeffs_blue : array, shape (order_blue,)
        Chebyshev coefficients for the blue arm (wavelength < wave_split).
    coeffs_red : array, shape (order_red,)
        Chebyshev coefficients for the red arm (wavelength >= wave_split).
    wave_split : float
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
    spectrum : array, shape (n_wave,)
        Physical model spectrum (any flux units).
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom], must be sorted.
    coeffs_blue : array, shape (order_blue,)
        Chebyshev coefficients for the blue arm (wavelength < wave_split).
    coeffs_red : array, shape (order_red,)
        Chebyshev coefficients for the red arm (wavelength >= wave_split).
    wave_split : float
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
