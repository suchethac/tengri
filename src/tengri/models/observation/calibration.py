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

ParamSpec integration example
-----------------------------
To add calibration coefficients as free parameters::

    from tengri.distributions import Gaussian

    spec = ParamSpec(
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
Johnson et al. (2021) — Prospector calibration model.
"""

import jax
import jax.numpy as jnp


@jax.jit
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
        Wavelength grid (Angstrom).
    order : int
        Maximum polynomial order (returns order+1 basis functions,
        from T_0 through T_order).
    wave_min, wave_max : float
        Wavelength range for normalization to [-1, 1].

    Returns
    -------
    array, shape (order+1, n_wave)
        Chebyshev basis T_0(x), T_1(x), ..., T_order(x).
    """
    x = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    # Build basis via scan over the recurrence
    def _step(carry, _k):
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
    """Multiplicative calibration polynomial C(lambda).

    C(lambda) = 1 + sum_{n=1}^{order} a_n * T_n(x)

    The constant term is fixed to 1 (flat calibration); ``coeffs``
    supplies a_1 through a_order representing deviations from unity.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid (Angstrom).
    coeffs : array, shape (order,)
        Chebyshev coefficients a_1, ..., a_order.  Empty array gives
        C(lambda) = 1 everywhere.
    wave_min, wave_max : float
        Wavelength range for normalization to [-1, 1].

    Returns
    -------
    array, shape (n_wave,)
        Calibration factor at each wavelength.
    """
    x = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    # Clenshaw recurrence for S = c_0 + sum_{n=1}^{N} c_n T_n(x)
    # with c_0 = 1 (implicit), c_n = coeffs[n-1] for n >= 1.
    #
    # Iterate from k=N down to k=1:
    #   b_k = c_k + 2*x*b_{k+1} - b_{k+2}
    # Then S = c_0 + x*b_1 - b_2
    # (The k=0 step uses x, not 2x, because T_0 = 1.)

    def _clenshaw_step(carry, c_k):
        b_kp1, b_kp2 = carry
        b_k = c_k + 2.0 * x * b_kp1 - b_kp2
        return (b_k, b_kp1), None

    init = (jnp.zeros_like(x), jnp.zeros_like(x))
    # Iterate over coeffs from highest order (c_N) down to c_1
    (b1, b2), _ = jax.lax.scan(_clenshaw_step, init, coeffs[::-1])

    # Final step: S = c_0 + x*b_1 - b_2, with c_0 = 1
    return 1.0 + x * b1 - b2


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
        Wavelength grid (Angstrom).
    coeffs : array, shape (order,)
        Chebyshev coefficients a_1, ..., a_order.
    wave_min, wave_max : float
        Wavelength range for normalization.

    Returns
    -------
    array, shape (n_wave,)
        Calibrated spectrum.
    """
    cal = calibration_polynomial(wavelength, coeffs, wave_min, wave_max)
    return spectrum * cal
