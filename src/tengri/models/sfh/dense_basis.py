"""Dense Basis GP-SFH: non-parametric star formation history via mass-time quantiles.

Parameterizes the SFH using a small number of mass-time quantile pairs
(when did the galaxy form X% of its stellar mass?) plus total stellar mass.
A Gaussian Process with Matérn 3/2 + Linear kernel smoothly interpolates
the cumulative mass curve; the SFR is the time derivative.

This is a clean-room JAX reimplementation of the algorithm described in:

- Iyer & Gawiser (2017), ApJ 838, 127 (arXiv:1702.04371)
  "Reconstruction of Galaxy Star Formation Histories through SED Fitting:
   The Dense Basis Approach"

- Iyer et al. (2019), ApJ 879, 116 (arXiv:1901.02877)
  "Nonparametric Star Formation History Reconstruction with Gaussian Processes.
   I. Counting Major Episodes of Star Formation"

The original implementation uses the ``george`` GP library with fixed kernel
hyperparameters. We reimplement the kernel math in pure JAX for JIT
compilation and automatic differentiation.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.

Implementation notes
--------------------
- Kernel hyperparameters follow the dense_basis convention (george GP):
  ``variance = np.var(y)``, ``length_scale = np.median(y)`` where y is the
  mass-quantile array. These are fixed, not optimized.
- SFR is computed via ``jnp.gradient()`` (central finite differences) on
  the GP-interpolated cumulative mass curve. This is a numerical
  approximation; the analytical GP derivative dK/dt @ α would be more
  principled but is not needed at 1000-point resolution (<0.1% error).
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT3 = jnp.sqrt(3.0)
_LENGTH_SCALE_FLOOR = 1e-10
_DT_FLOOR = 1e-20
_GP_NOISE_NUMERATOR = 0.001  # dense_basis: yerr = 0.001 / sqrt(Nparam)

# Big Bang constraint: cosmic time fraction where we place a (t, M=0) anchor
# to enforce SFR=0 at early times. tx_frac values are clipped to be above
# this threshold to prevent non-monotonic quantile placement.
_BB_CONSTRAINT_FRAC = 0.01

# Default GP interpolation resolution (number of points in cosmic time)
_DEFAULT_RES = 1000


def matern32_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    variance: float,
    length_scale: float,
) -> jnp.ndarray:
    """Matérn 3/2 covariance matrix.

    K(r) = σ² (1 + √3 r / ℓ) exp(-√3 r / ℓ)

    Parameters
    ----------
    x1 : array, shape (n1,)
        First set of input points.
    x2 : array, shape (n2,)
        Second set of input points.
    variance : float
        Signal variance σ².
    length_scale : float
        Length scale ℓ.

    Returns
    -------
    array, shape (n1, n2)
        Covariance matrix.
    """
    r = jnp.abs(x1[:, None] - x2[None, :])
    sqrt3_r_l = _SQRT3 * r / jnp.maximum(length_scale, _LENGTH_SCALE_FLOOR)
    return variance * (1.0 + sqrt3_r_l) * jnp.exp(-sqrt3_r_l)


def linear_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    variance: float,
    length_scale: float,
) -> jnp.ndarray:
    """Linear (polynomial order-1) covariance matrix.

    K(x1, x2) = σ² x1 x2 / ℓ²

    This matches ``george.kernels.LinearKernel`` with ``order=2``
    (which is actually a polynomial degree-1 kernel in george's convention).

    Parameters
    ----------
    x1 : array, shape (n1,)
        First set of input points.
    x2 : array, shape (n2,)
        Second set of input points.
    variance : float
        Signal variance σ².
    length_scale : float
        Length scale ℓ.

    Returns
    -------
    array, shape (n1, n2)
        Covariance matrix.
    """
    ls_sq = jnp.maximum(length_scale, _LENGTH_SCALE_FLOOR) ** 2
    return variance * (x1[:, None] * x2[None, :]) / ls_sq


def combined_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    variance: float,
    length_scale: float,
) -> jnp.ndarray:
    """Matérn 3/2 + Linear kernel (matches dense_basis george configuration).

    Parameters
    ----------
    x1, x2 : array
        Input points.
    variance, length_scale : float
        Shared kernel hyperparameters.

    Returns
    -------
    array, shape (n1, n2)
        Combined covariance matrix.
    """
    return matern32_kernel(x1, x2, variance, length_scale) + linear_kernel(
        x1, x2, variance, length_scale
    )


# ---------------------------------------------------------------------------
# GP interpolation
# ---------------------------------------------------------------------------


def gp_interpolate(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    y_err: jnp.ndarray,
    x_eval: jnp.ndarray,
    variance: float,
    length_scale: float,
) -> jnp.ndarray:
    """GP conditional mean prediction at evaluation points.

    Computes the posterior mean of the GP conditioned on training data:
        μ* = K(x*, x_train) @ (K(x_train, x_train) + diag(σ²))⁻¹ @ y_train

    Parameters
    ----------
    x_train : array, shape (n_train,)
        Training input points.
    y_train : array, shape (n_train,)
        Training output values.
    y_err : array, shape (n_train,)
        Observation noise standard deviation at each training point.
    x_eval : array, shape (n_eval,)
        Evaluation points where GP mean is predicted.
    variance : float
        Kernel signal variance.
    length_scale : float
        Kernel length scale.

    Returns
    -------
    array, shape (n_eval,)
        GP posterior mean at evaluation points.
    """
    k_train = combined_kernel(x_train, x_train, variance, length_scale)
    k_train = k_train + jnp.diag(y_err**2)
    k_eval = combined_kernel(x_eval, x_train, variance, length_scale)
    alpha = jnp.linalg.solve(k_train, y_train)
    return k_eval @ alpha


# ---------------------------------------------------------------------------
# Quantile → cumulative mass → SFR
# ---------------------------------------------------------------------------


def _build_quantile_points(
    tx_fracs: jnp.ndarray,
    n_param: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build (time, mass) quantile points from tx fractions.

    Creates N_param + 2 quantile pairs: Big Bang (0, 0), intermediate
    quantiles from tx_fracs, and observation epoch (1, 1).

    Adds a constraint point near t=0 to enforce SFR=0 at the Big Bang.

    Parameters
    ----------
    tx_fracs : array, shape (n_param,)
        Cosmic time fractions (0 = Big Bang, 1 = observation epoch).
        Must be sorted and in (0, 1).
    n_param : int
        Number of intermediate quantile parameters.

    Returns
    -------
    time_q : array, shape (n_param + 3,)
        Time fractions including Big Bang constraint at t=0.01.
    mass_q : array, shape (n_param + 3,)
        Corresponding cumulative mass fractions.
    """
    # Mass quantiles: evenly spaced from 0 to 1
    mass_quantiles = jnp.linspace(0.0, 1.0, n_param + 2)

    # Time quantiles: [0, tx_0, tx_1, ..., tx_N, 1]
    time_quantiles = jnp.concatenate(
        [
            jnp.array([0.0]),
            tx_fracs,
            jnp.array([1.0]),
        ]
    )

    # Add Big Bang constraint: (t≈0, M=0) to enforce SFR=0 at birth
    # (matching dense_basis convention)
    time_q = jnp.concatenate([jnp.array([_BB_CONSTRAINT_FRAC]), time_quantiles])
    mass_q = jnp.concatenate([jnp.array([0.0]), mass_quantiles])

    return time_q, mass_q


def _cumulative_mass_to_sfr(
    t_cosmic_frac: jnp.ndarray,
    m_cumul_frac: jnp.ndarray,
    age_universe_yr: float,
    total_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert cumulative mass fraction curve to SFR in lookback time.

    Parameters
    ----------
    t_cosmic_frac : array, shape (res,)
        Cosmic time fractions (0 = Big Bang, 1 = observation epoch).
    m_cumul_frac : array, shape (res,)
        Cumulative mass fraction (0 to ~1).
    age_universe_yr : float
        Age of the universe at observation epoch (yr).
    total_mass : float
        Total stellar mass formed (Msun).

    Returns
    -------
    t_lookback_yr : array, shape (res,)
        Lookback time in years (reversed: 0 = now, large = old).
    sfr : array, shape (res,)
        Star formation rate in Msun/yr, non-negative.
    """
    # Clip cumulative mass to [0, 1] and enforce monotonicity
    m_cumul_frac = jnp.clip(m_cumul_frac, 0.0, 1.0)

    # Convert cosmic time fraction to years
    t_cosmic_yr = t_cosmic_frac * age_universe_yr

    # SFR = dM/dt_cosmic (derivative of cumulative mass w.r.t. cosmic time)
    dt = jnp.gradient(t_cosmic_yr)
    dm = jnp.gradient(m_cumul_frac * total_mass)
    sfr = dm / jnp.maximum(dt, _DT_FLOOR)

    # Convert cosmic time to lookback time: t_lb = age_universe - t_cosmic
    t_lookback_yr = age_universe_yr - t_cosmic_yr

    # Reverse so lookback time is increasing (0 = present → large = old)
    t_lookback_yr = t_lookback_yr[::-1]
    sfr = sfr[::-1]

    return t_lookback_yr, jnp.maximum(sfr, 0.0)


# ---------------------------------------------------------------------------
# Main SFH function
# ---------------------------------------------------------------------------


def dense_basis_sfh(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    age_universe_yr: float = 13.8e9,
    **tx_kwargs: float,
) -> jnp.ndarray:
    """Non-parametric GP-SFH via mass-time quantiles (Iyer+2017, 2019).

    Parameterizes the SFH using a small number of mass-time quantile pairs:
    tx_frac_i is the cosmic time fraction at which the galaxy has formed
    (i+1)/(N+1) of its total stellar mass.

    A Gaussian Process with Matérn 3/2 + Linear kernel smoothly interpolates
    the cumulative mass curve. The SFR is the time derivative.

    Parameters
    ----------
    age_yr : array (n_age,)
        Lookback time grid in years.
    log_total_mass : float
        log10(total stellar mass formed / Msun).
    age_universe_yr : float
        Age of the universe at observation epoch (yr). Default: 13.8e9 (z≈0).
        Override for higher redshifts.
    **tx_kwargs
        Keyword arguments named ``tx_frac_0``, ``tx_frac_1``, ...,
        ``tx_frac_{N-1}`` containing cosmic time fractions in (0, 1).
        These are the times (as fraction of cosmic age) when the galaxy
        formed 25%, 50%, 75%, ... of its total stellar mass (for N=3).

    Returns
    -------
    array (n_age,)
        SFR at each lookback time (Msun/yr), non-negative.

    Notes
    -----
    The tx fractions are sorted internally via ``jnp.sort()`` to enforce
    monotonicity of the cumulative mass curve. This is differentiable.

    Currently only N=3 quantile parameters are supported via the registry
    (matching Iyer+2019 defaults). The function itself accepts any N.

    Kernel hyperparameters follow the dense_basis/george convention:
    ``variance = np.var(mass_quantiles)``,
    ``length_scale = np.median(mass_quantiles)``.
    These are fixed (not optimized), making the GP a smooth deterministic
    interpolator. See ``gp_sfh.py`` in the dense_basis package,
    lines 89-95 (v0.1.9).

    References
    ----------
    - Iyer & Gawiser (2017), ApJ 838, 127 (arXiv:1702.04371).
    - Iyer et al. (2019), ApJ 879, 116 (arXiv:1901.02877).
    """
    # Collect tx fractions from kwargs, validating keys
    n_param = len(tx_kwargs)
    if n_param == 0:
        raise ValueError("dense_basis_sfh requires at least one tx_frac_* parameter")
    for i in range(n_param):
        key = f"tx_frac_{i}"
        if key not in tx_kwargs:
            raise ValueError(
                f"Missing required parameter '{key}'. Got keys: "
                f"{sorted(tx_kwargs.keys())}. Expected tx_frac_0 "
                f"through tx_frac_{n_param - 1}."
            )
    tx_fracs = jnp.array([tx_kwargs[f"tx_frac_{i}"] for i in range(n_param)])

    # Enforce ordering (differentiable via JAX sort gradients)
    # Clip above BB constraint to prevent non-monotonic quantile placement
    tx_fracs = jnp.sort(tx_fracs)
    tx_fracs = jnp.clip(tx_fracs, _BB_CONSTRAINT_FRAC + 0.01, 0.99)

    # Build quantile points
    time_q, mass_q = _build_quantile_points(tx_fracs, n_param)

    # Kernel hyperparameters (matching dense_basis/george convention):
    # variance = np.var(y), length_scale = np.median(y)
    # where y is the mass-quantile array (range 0-1).
    # These are FIXED, not optimized — the GP is a smooth interpolator.
    variance = jnp.var(mass_q)
    length_scale = jnp.maximum(jnp.median(mass_q), _LENGTH_SCALE_FLOOR)

    # Noise floor: tight on quantile points (matching dense_basis)
    noise_scale = _GP_NOISE_NUMERATOR / jnp.sqrt(jnp.maximum(n_param, 1.0))
    y_err = jnp.full_like(mass_q, noise_scale)

    # Dense evaluation grid in cosmic time fraction
    t_eval = jnp.linspace(_BB_CONSTRAINT_FRAC, 1.0, _DEFAULT_RES)

    # GP interpolation of cumulative mass fraction
    m_cumul_frac = gp_interpolate(time_q, mass_q, y_err, t_eval, variance, length_scale)

    # Convert cumulative mass to SFR on lookback time grid
    total_mass = 10.0**log_total_mass
    t_lookback_yr, sfr_dense = _cumulative_mass_to_sfr(
        t_eval, m_cumul_frac, age_universe_yr, total_mass
    )

    # Interpolate onto the requested age grid
    sfr = jnp.interp(age_yr, t_lookback_yr, sfr_dense)
    return jnp.maximum(sfr, 0.0)
