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
import numpy as np
from jax.scipy.linalg import block_diag

from tengri.observation.banded import BandedMatrix
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


@partial(jax.jit, static_argnames=("n_poly", "n_segments"))
def calibration_basis(
    wave: jnp.ndarray,
    wave_min: float,
    wave_max: float,
    n_poly: int,
    segments: jnp.ndarray | None = None,
    n_segments: int = 0,
) -> jnp.ndarray:
    r"""Chebyshev basis T_1 through T_{n_poly}, optionally prefixed by segment indicators.

    Column :math:`k` of the Chebyshev block is :math:`T_k(x)` with
    :math:`x = (2\lambda - \lambda_\min - \lambda_\max)/(\lambda_\max - \lambda_\min)`,
    so that :math:`1 + B\,c` equals :func:`calibration_polynomial` for coefficients
    :math:`c = (a_1, \ldots, a_{n})`.

    With ``segments`` given, the result is :math:`[\mathbb{1}_{s=1}, \ldots,
    \mathbb{1}_{s=n_\mathrm{seg}}, T_1(x), \ldots, T_{n_\mathrm{poly}}(x)]`: one
    indicator column per camera segment, so that the calibration multiplier
    :math:`\sum_s c_s \mathbb{1}_{s} + \sum_j a_j T_j` carries its constant in the
    segment scales and no implicit unit term.

    Parameters
    ----------
    wave : array_like, shape (n,)
        Wavelength grid [Angstrom].
    wave_min : float
        Lower bound of the normalization interval [Angstrom].
    wave_max : float
        Upper bound of the normalization interval [Angstrom].
    n_poly : int
        Highest Chebyshev order. Zero leaves only the segment indicators when
        ``segments`` is given. Static under JIT.
    segments : array_like of int, shape (n,), optional
        Camera segment id of every pixel, in ``range(n_segments)``. Default: no
        segment columns.
    n_segments : int, optional
        Number of segment columns. Static under JIT. Used only with ``segments``.
        Default 0.

    Returns
    -------
    ndarray, shape (n, n_poly) or (n, n_segments + n_poly)
        Basis matrix (dimensionless). Without ``segments`` the columns are
        :math:`[T_1(x), \ldots, T_{n_\mathrm{poly}}(x)]`; with ``segments`` the
        indicator columns come first.

    Notes
    -----
    **JIT-compatible**: yes. ``n_poly`` and ``n_segments`` are static; the recurrence
    is a Python loop over ``n_poly``, so no ``lax.scan`` is traced.

    **Gradient-safe**: yes, differentiable with respect to ``wave``, ``wave_min`` and
    ``wave_max``. The indicator columns carry no gradient.

    The recurrence is :math:`T_0 = 1`, :math:`T_1 = x`,
    :math:`T_{k+1} = 2x\,T_k - T_{k-1}`.

    """
    x = 2.0 * (wave - wave_min) / (wave_max - wave_min) - 1.0
    columns = []
    if n_poly >= 1:
        columns.append(x)
        t_prev, t_curr = jnp.ones_like(x), x
        for _ in range(n_poly - 1):
            t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
            columns.append(t_curr)
    chebyshev = jnp.stack(columns, axis=-1) if columns else jnp.zeros((x.shape[0], 0), x.dtype)
    if segments is None:
        return chebyshev
    indicator = (segments[:, None] == jnp.arange(n_segments)).astype(x.dtype)
    return jnp.concatenate([indicator, chebyshev], axis=-1)


@partial(jax.jit, static_argnames=("n_poly", "chol_offsets"))
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
    chol_data: jnp.ndarray | None = None,
    chol_offsets: tuple[int, ...] | None = None,
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
    chol_data : array_like, shape (p + 1, n_wave), optional, keyword-only
        Bands of the lower Cholesky factor of the pixel covariance, from
        :func:`~tengri.observation.banded.banded_cholesky`. With a factor the pixels are
        whitened by it and ``obs_err`` is not read. Default: None (diagonal ``obs_err``).
    chol_offsets : tuple of int, optional, keyword-only
        Diagonal offsets ``(0, -1, ..., -p)`` of ``chol_data``. Static. Required with
        ``chol_data``.

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

    if chol_data is None:
        # Guard only against sigma == 0, in the sigma domain, at the working dtype's
        # smallest normal. Never a variance floor (#1588).
        noise = {"sigma": jnp.maximum(obs_err, _representable_floor(_ERR_FLOOR))}
    else:
        noise = {"chol_lower": BandedMatrix(np.asarray(chol_offsets, dtype=int), chol_data)}

    result = marginalize_linear_nuisance(
        obs_flux,
        model_flux,
        model_flux[:, jnp.newaxis] * basis,
        **noise,
        pix_mask=pix_mask,
        prior_precision=jnp.eye(n_poly) / prior_sigma**2,
    )
    c_hat_err = jnp.sqrt(jnp.diag(result.coef_cov))
    return result.log_like, result.coef_hat, c_hat_err


def mean_multiplier_vector(
    basis: jnp.ndarray,
    weight: jnp.ndarray,
    window: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Return the vector :math:`v` with :math:`q - 1 = v^\mathsf{T}\beta` for the window mean.

    The calibration multiplier is :math:`M = 1 + B\beta` with :math:`B` from
    :func:`calibration_basis` and the segment coefficients written as deviations
    :math:`u_s = s_s - 1`. Its mean over the window :math:`W` is
    :math:`q = \sum_i w_i W_i M_i / \sum_i w_i W_i = 1 + v^\mathsf{T}\beta`.

    Parameters
    ----------
    basis : array_like, shape (n, k)
        Calibration basis from :func:`calibration_basis` (dimensionless).
    weight : array_like, shape (n,)
        Pixel mask :math:`w_i \in \{0, 1\}`.
    window : array_like, shape (n,), optional
        Window :math:`W_i \in \{0, 1\}`. Default: every pixel.

    Returns
    -------
    ndarray, shape (k,)
        :math:`v_k = \sum_i w_i W_i B_{ik} / \sum_i w_i W_i` (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, a constant of the data.
    """
    w = weight if window is None else weight * window
    return (w @ basis) / jnp.sum(w)


def calibration_q_log_prior(
    log_q: jnp.ndarray,
    v: jnp.ndarray,
    *,
    prior_sigma: float = 1.0,
    cal_prior: str = "gaussian",
) -> jnp.ndarray:
    r"""Log prior on :math:`\ln q` that the coefficient prior contributes.

    With ``cal_prior="gaussian"`` the coefficient prior :math:`\mathcal{N}(0,
    \sigma_p^2 I)` has the marginal :math:`q \sim \mathcal{N}(1, \sigma_p^2
    v^\mathsf{T}v)`. Per unit :math:`\ln q` its log density is

    .. math::

        \ln\mathcal{N}(q;\,1,\,\sigma_p^2 v^\mathsf{T}v) + \ln q,

    the second term being :math:`\ln(dq/d\ln q)`. With ``"scale_free"`` the flat
    coefficient prior contributes nothing along :math:`q`, and the result is zero.
    Add the log-normal prior on :math:`\ln q` to this value.

    Parameters
    ----------
    log_q : array_like, shape ()
        :math:`\ln q`, the log of the mean multiplier (dimensionless).
    v : array_like, shape (k,)
        Vector from :func:`mean_multiplier_vector` (dimensionless).
    prior_sigma : float, optional
        Width :math:`\sigma_p` of the coefficient prior. Default 1.0.
    cal_prior : {"gaussian", "scale_free"}, optional
        Static. Default ``"gaussian"``.

    Returns
    -------
    ndarray, shape ()
        Log prior density per unit :math:`\ln q` (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.
    """
    if cal_prior == "scale_free":
        return jnp.zeros_like(log_q)
    q = jnp.exp(log_q)
    var = prior_sigma**2 * jnp.dot(v, v)
    return -0.5 * ((q - 1.0) ** 2 / var + jnp.log(2.0 * jnp.pi * var)) + log_q


@partial(jax.jit, static_argnames=("n_segments", "n_poly", "cal_prior", "q_prior"))
def marginalize_calibration_segments(
    model_flux: jnp.ndarray,
    obs_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    wavelength: jnp.ndarray,
    segments: jnp.ndarray,
    *,
    n_segments: int,
    n_poly: int = 0,
    prior_sigma: float = 1.0,
    cal_prior: str = "gaussian",
    pix_mask: jnp.ndarray | None = None,
    wave_range: tuple[float, float] | None = None,
    lines=None,
    line_col_mask: jnp.ndarray | None = None,
    log_q: jnp.ndarray | None = None,
    q_window: jnp.ndarray | None = None,
    q_prior: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Marginalize per-segment calibration scales and a Chebyshev tilt in closed form.

    The calibrated model of a multi-camera spectrum is
    :math:`M(\lambda) = \sum_s c_s \mathbb{1}_{\mathrm{seg}(\lambda)=s} + \sum_j a_j T_j(x)`
    times the model flux, with one free scale :math:`c_s` per camera segment and a
    shared tilt of order ``n_poly``. Every column is linear in the coefficients, so the
    data term is :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`
    with no fixed offset. Emission-line columns (``lines``) join the same marginal and
    keep their own prior; ``log_q`` pins the window-mean multiplier :math:`q`.

    Parameters
    ----------
    model_flux : array_like, shape (n,)
        Physical model spectrum [data flux units].
    obs_flux : array_like, shape (n,)
        Observed spectrum [data flux units].
    obs_err : array_like, shape (n,)
        One-sigma uncertainties [data flux units].
    wavelength : array_like, shape (n,)
        Wavelength grid [Angstrom]; normalizes the Chebyshev tilt.
    segments : array_like of int, shape (n,)
        Camera segment id of every pixel, in ``range(n_segments)``.
    n_segments : int, keyword-only
        Number of segment scales. Static under JIT.
    n_poly : int, keyword-only
        Order of the shared Chebyshev tilt. Zero keeps only the segment scales.
        Static under JIT. Default 0.
    prior_sigma : float, keyword-only
        Width of the Gaussian prior on each coefficient when ``cal_prior="gaussian"``.
        Default 1.0.
    cal_prior : {"gaussian", "scale_free"}, keyword-only
        ``"gaussian"`` puts :math:`\mathcal{N}(0, \sigma_p^2)` on each segment deviation
        :math:`u_s = s_s - 1` and on each tilt coefficient, so each segment scale
        :math:`s_s` has the prior :math:`\mathcal{N}(1, \sigma_p^2)`. ``"scale_free"``
        puts a flat prior on the coefficients in data-flux units: every calibration
        column is divided by :math:`a = \sum_i w_i m_i / N_\mathrm{eff}`, the masked
        mean model flux, and the flat-prior Jacobian :math:`k\ln a` is added, where
        :math:`k` counts only the calibration columns. Default ``"gaussian"``.
    pix_mask : array_like, shape (n,), optional, keyword-only
        Pixel mask in {0, 1}. Masked pixels are absent from every term. Default: all
        pixels used.
    wave_range : tuple of float, optional, keyword-only
        ``(wave_min, wave_max)`` [Angstrom] normalizing the tilt. Default: the grid
        extremes.
    lines : UnitLinePrior, optional, keyword-only
        Emission-line block from
        :func:`~tengri.observation.eline_marginalization.unit_line_prior`, built on the
        same ``obs_flux`` units. Default: no lines.
    line_col_mask : array_like, shape (n_lines,), optional, keyword-only
        1 for a line with support on the grid, 0 to remove it. Default: all ones.
    log_q : array_like, shape (), optional, keyword-only
        :math:`\ln q`, the log of the multiplier's mean over ``q_window``
        (dimensionless). Given, the coefficient block is conditioned on
        :math:`v^\mathsf{T}\beta = q - 1` and the return value is the marginal given
        :math:`q`. Needs at least two calibration columns.
    q_window : array_like, shape (n,), optional, keyword-only
        Window :math:`W_i \in \{0, 1\}` over which :math:`q` is the mean. Default:
        every unmasked pixel.
    q_prior : bool, keyword-only
        Static. With ``log_q`` given, add :func:`calibration_q_log_prior`, so the value is
        the log density of :math:`\ln q` under the coefficient prior times the data
        likelihood, up to the user's own prior on :math:`\ln q`. Default False.

    Returns
    -------
    log_likelihood_marginal : ndarray, shape ()
        Marginal log-likelihood with the calibration coefficients (and the lines)
        integrated out (dimensionless).
    coef_hat : ndarray, shape (n_segments + n_poly,)
        Posterior mean of the calibration coefficients. For ``"gaussian"`` the first
        ``n_segments`` are the segment deviations :math:`u_s`, so the scales are
        ``1 + coef_hat``, and the rest are the tilt coefficients. For ``"scale_free"``
        they are in data-flux units, and the multiplier coefficients are
        ``coef_hat / a``.
    coef_err : ndarray, shape (n_segments + n_poly,)
        Posterior standard deviations of the coefficients, in the units of
        ``coef_hat``.

    Raises
    ------
    ValueError
        If ``cal_prior`` is neither ``"gaussian"`` nor ``"scale_free"``, or ``log_q``
        is given with fewer than two calibration columns.

    Notes
    -----
    **JIT-compatible**: yes; ``n_segments``, ``n_poly`` and ``cal_prior`` are static.

    **Gradient-safe**: yes, with respect to ``model_flux``, ``obs_flux``, ``obs_err``
    and ``log_q``, inherited from
    :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`.
    The ``"scale_free"`` branch passes the normalization as the primitive's coefficient
    scale, so the whitened design stays order unity when the flux is far below unity.

    **vmap**: yes over any leading batch axis.

    The data model is :math:`d = w\,\cdot\,M\,m + \epsilon`, with :math:`\epsilon` drawn
    from :math:`\mathcal{N}(0, \sigma^2)`. The offset is :math:`m` and the segment
    columns carry :math:`u_s = s_s - 1`, which is an exact reparametrization of the
    :math:`\mathcal{N}(1, \sigma_p^2)` prior on :math:`s_s`. Keeping the offset at the
    model, rather than zero with the scale in the columns, keeps the residual small so
    float32 does not cancel two large sums of squares. The scale-free Jacobian is the change of
    variables from the multiplier coefficients :math:`c` to data-flux coefficients
    :math:`\beta = a\,c`, which contributes :math:`+k\ln a` to the log-density. Under
    rescaling of the model, data and errors by a common factor :math:`\lambda`, the
    marginal changes by :math:`(k - N_\mathrm{eff})\ln\lambda`: the density of the
    data moves with its units and the flat coefficient prior with the calibration
    columns only.

    With ``log_q`` the mean multiplier is :math:`q = 1 + v^\mathsf{T}\beta` with
    :math:`v` from :func:`mean_multiplier_vector`. The coefficients are written
    :math:`\beta = Q\gamma + u\,(q - 1)` with :math:`u = v/v^\mathsf{T}v` and
    :math:`Q` an orthonormal basis of the null space of :math:`v^\mathsf{T}`
    (:math:`v^\mathsf{T}Q = 0`, :math:`Q^\mathsf{T}Q = I`), so the design is
    :math:`XQ` (full rank), the offset is :math:`m + Xu\,(q - 1)`, and the Jacobian
    of :math:`(\gamma, q) \to \beta` is constant. For the isotropic Gaussian prior the
    conditional prior of :math:`\gamma` given :math:`q` is :math:`\mathcal{N}(0,
    \sigma_p^2 I)` because :math:`Q^\mathsf{T}u = 0`; the marginal of :math:`q`
    itself is :func:`calibration_q_log_prior`. With ``"scale_free"`` the flat branch
    counts the :math:`k - 1` columns of :math:`\gamma` only.

    """
    if cal_prior not in ("gaussian", "scale_free"):
        raise ValueError(f"cal_prior must be 'gaussian' or 'scale_free', got {cal_prior!r}")
    n_cal = n_segments + n_poly
    if log_q is not None and n_cal < 2:
        raise ValueError("log_q needs at least two calibration columns")
    if wave_range is None:
        wave_min, wave_max = wavelength.min(), wavelength.max()
    else:
        wave_min, wave_max = wave_range

    free = cal_prior == "scale_free"
    weight = jnp.ones_like(model_flux) if pix_mask is None else pix_mask
    n_eff = jnp.sum(weight)
    err_safe = jnp.maximum(obs_err, _representable_floor(_ERR_FLOOR))
    # Whiten by the masked mean sigma before the primitive: the data, model and
    # sigma are divided by one common unit, and the exact change of the marginal is
    # added back below. Keeps the whitened design representable at flux scale.
    unit = jnp.sum(weight * err_safe) / n_eff
    model_n = model_flux / unit
    data_n = obs_flux / unit
    err_n = err_safe / unit

    basis = calibration_basis(
        wavelength, wave_min, wave_max, n_poly, segments=segments, n_segments=n_segments
    )
    design_cal = model_n[:, jnp.newaxis] * basis
    mean_model = jnp.sum(weight * model_n) / n_eff
    cal_design = design_cal / mean_model if free else design_cal

    model0 = model_n
    if log_q is not None:
        v = mean_multiplier_vector(basis, weight, q_window)
        u_vec = v / jnp.dot(v, v)
        q_null = jnp.linalg.qr(v[:, jnp.newaxis], mode="complete")[0][:, 1:]
        shift = jnp.exp(log_q) - 1.0
        model0 = model_n + design_cal @ (u_vec * shift)
        cal_design = cal_design @ q_null
    k_cal = cal_design.shape[1]

    if lines is None:
        design = cal_design
        prior_precision = None if free else jnp.eye(k_cal) / prior_sigma**2
        prior_mean, col_mask, flat_mask = None, None, None
        coef_scale = mean_model if free else None
    else:
        n_line = lines.design.shape[1]
        line_cols = jnp.ones(n_line) if line_col_mask is None else line_col_mask
        design = jnp.concatenate([cal_design, lines.design / unit], axis=1)
        cal_precision = jnp.zeros((k_cal, k_cal)) if free else jnp.eye(k_cal) / prior_sigma**2
        prior_precision = block_diag(cal_precision, lines.prior_precision)
        prior_mean = jnp.concatenate([jnp.zeros(k_cal), lines.prior_mean])
        col_mask = jnp.concatenate([jnp.ones(k_cal), line_cols])
        flat_mask = jnp.concatenate([jnp.full(k_cal, float(free)), jnp.zeros(n_line)])
        coef_scale = lines.scale
    result = marginalize_linear_nuisance(
        data_n,
        model0,
        design,
        sigma=err_n,
        pix_mask=pix_mask,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        col_mask=col_mask,
        scale=coef_scale,
        flat_mask=flat_mask,
    )
    log_like = result.log_like - n_eff * jnp.log(unit)
    if log_q is not None and q_prior:
        log_like = log_like + calibration_q_log_prior(
            log_q, v, prior_sigma=prior_sigma, cal_prior=cal_prior
        )
    if free:
        log_like = log_like + k_cal * jnp.log(unit)

    gamma_hat = result.coef_hat[:k_cal]
    gamma_cov = result.coef_cov[:k_cal, :k_cal]
    coef_unit = unit if free else 1.0
    if log_q is None:
        coef_hat, cov = coef_unit * gamma_hat, coef_unit**2 * gamma_cov
    else:
        offset_scale = mean_model * unit if free else 1.0
        coef_hat = coef_unit * (q_null @ gamma_hat) + offset_scale * u_vec * shift
        cov = coef_unit**2 * (q_null @ gamma_cov @ q_null.T)
    return log_like, coef_hat, jnp.sqrt(jnp.diag(cov))


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
