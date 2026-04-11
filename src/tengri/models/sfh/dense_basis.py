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
compilation and automatic differentiation. The algorithm follows
``dense_basis.gp_sfh.tuple_to_sfh()`` step-for-step.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT3 = jnp.sqrt(3.0)
_LENGTH_SCALE_FLOOR = 1e-10

# GP interpolation resolution (matching dense_basis default: 1000 points)
_DEFAULT_RES = 1000


# ---------------------------------------------------------------------------
# GP Kernel functions
# ---------------------------------------------------------------------------


def matern32_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    variance: float,
    length_scale: float,
) -> jnp.ndarray:
    """Matérn 3/2 covariance matrix.

    K(r) = σ² (1 + √3 r / ℓ) exp(-√3 r / ℓ)

    Matches ``george.kernels.Matern32Kernel(metric=ℓ²)``.

    Parameters
    ----------
    x1 : array, shape (n1,)
    x2 : array, shape (n2,)
    variance : float
        Signal variance σ².
    length_scale : float
        Length scale ℓ.

    Returns
    -------
    array, shape (n1, n2)
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

    Matches ``george.kernels.LinearKernel(order=2, log_gamma2=ln(ℓ))``.
    """
    ls_sq = jnp.maximum(length_scale, _LENGTH_SCALE_FLOOR) ** 2
    return variance * (x1[:, None] * x2[None, :]) / ls_sq


def combined_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    variance: float,
    length_scale: float,
) -> jnp.ndarray:
    """Matérn 3/2 + Linear kernel (matches dense_basis george config)."""
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
    """GP conditional mean: μ* = K* (K + σ²I)⁻¹ y."""
    k_train = combined_kernel(x_train, x_train, variance, length_scale)
    k_train = k_train + jnp.diag(y_err**2)
    k_eval = combined_kernel(x_eval, x_train, variance, length_scale)
    alpha = jnp.linalg.solve(k_train, y_train)
    return k_eval @ alpha


# ---------------------------------------------------------------------------
# Quantile point construction (matching dense_basis exactly)
# ---------------------------------------------------------------------------


def _build_quantile_points(
    tx_fracs: jnp.ndarray,
    n_param: int,
    log_total_mass: float,
    log_sfr_inst: float,
    age_universe_yr: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build (time, mass, yerr) quantile points — matching dense_basis.

    Constructs (in order):
    1. Endpoint: (0, 0)
    2. Big Bang constraint: (0.01, 0) — enforces SFR=0 at early times
    3. Intermediate quantiles from tx_fracs: (tx_i, mass_i)
    4. Endpoint: (1, 1)
    5. Observation-epoch SFR constraints: 3 points near t=1 that
       pin the cumulative mass curve to be consistent with the
       instantaneous SFR at observation (dense_basis lines 140-152).

    Parameters
    ----------
    tx_fracs : array, shape (n_param,)
        Cosmic time fractions in (0, 1), sorted.
    n_param : int
        Number of intermediate quantile parameters.
    log_total_mass : float
        log10(total stellar mass / Msun).
    log_sfr_inst : float
        log10(instantaneous SFR at observation / Msun/yr).
    age_universe_yr : float
        Age of the universe at observation epoch (yr).

    Returns
    -------
    time_q : array
        Time fractions for GP training.
    mass_q : array
        Cumulative mass fractions for GP training.
    yerr : array
        Noise standard deviations for each training point.
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

    # Insert Big Bang constraint at position 1: (t=0.01, M=0)
    # (dense_basis lines 135-136)
    time_q = jnp.concatenate(
        [
            time_quantiles[:1],  # t=0
            jnp.array([0.01]),  # BB constraint
            time_quantiles[1:],  # tx_0, ..., tx_N, 1.0
        ]
    )
    mass_q = jnp.concatenate(
        [
            mass_quantiles[:1],  # M=0
            jnp.array([0.0]),  # BB constraint M=0
            mass_quantiles[1:],  # intermediate + 1.0
        ]
    )

    # --- Observation-epoch SFR constraints (dense_basis lines 140-152) ---
    # Three points at 97%, 98%, 99% cumulative mass, timed to be
    # consistent with the instantaneous SFR at observation.
    # delta_mstar = M* * (1 - const_val)
    # delta_t_frac = 1 - delta_mstar / (SFR_inst * age_universe_yr)
    # These are inserted BEFORE the final (1, 1) endpoint.
    total_mass = 10.0**log_total_mass
    sfr_inst = 10.0**log_sfr_inst
    age_yr = age_universe_yr
    const_vals = jnp.array([0.97, 0.98, 0.99])

    sfr_time_q = []
    sfr_mass_q = []
    for cv in const_vals:
        delta_mstar = total_mass * (1.0 - cv)
        delta_t = 1.0 - delta_mstar / (sfr_inst * age_yr)
        delta_t = jnp.clip(delta_t, 0.01, 0.999)
        sfr_time_q.append(delta_t)
        sfr_mass_q.append(cv)

    # Insert SFR constraints BEFORE the final (1.0, 1.0) endpoint
    # (matching dense_basis point ordering)
    time_q = jnp.concatenate(
        [
            time_q[:-1],  # all except last
            jnp.array(sfr_time_q),  # SFR constraints
            time_q[-1:],  # final (1.0)
        ]
    )
    mass_q = jnp.concatenate(
        [
            mass_q[:-1],  # all except last
            jnp.array(sfr_mass_q),  # SFR constraints
            mass_q[-1:],  # final (1.0)
        ]
    )

    # --- Noise array (dense_basis gp_interpolator lines 63-68) ---
    # yerr = 0 everywhere except user quantile points (indices 2..2+Nparam)
    # which get 0.001/sqrt(Nparam). SFR constraint points get yerr=0
    # (tight constraint, matching dense_basis decouple_sfr=False).
    n_total = time_q.shape[0]
    yerr = jnp.zeros(n_total)
    noise_scale = 0.001 / jnp.sqrt(jnp.maximum(n_param, 1.0))
    yerr = yerr.at[2 : 2 + n_param].set(noise_scale)

    return time_q, mass_q, yerr


# ---------------------------------------------------------------------------
# Main SFH function
# ---------------------------------------------------------------------------


def dense_basis_sfh(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    log_sfr_inst: float = 0.0,
    age_universe_yr: float = 13.47e9,
    **tx_kwargs: float,
) -> jnp.ndarray:
    """Non-parametric GP-SFH via mass-time quantiles (Iyer+2017, 2019).

    Faithfully reimplements ``dense_basis.gp_sfh.tuple_to_sfh()`` in JAX.
    The SFR is derived from the GP-interpolated cumulative mass curve
    using the same ``sfh_scale * np.diff(mass_interp)`` approach as
    the original, NOT ``jnp.gradient()``.

    Parameters
    ----------
    age_yr : array (n_age,)
        Lookback time grid in years.
    log_total_mass : float
        log10(total stellar mass formed / Msun).
    log_sfr_inst : float
        log10(instantaneous SFR at observation / Msun/yr).
        Used to add 3 constraint points near t=1 that pin the
        recent SFH shape (dense_basis lines 140-152).
        Default: 0.0 (1 Msun/yr).
    age_universe_yr : float
        Age of the universe at observation epoch (yr).
        Default: 13.47e9 (FlatLambdaCDM H0=70, Om0=0.3 at z=0).
    **tx_kwargs
        Keyword arguments ``tx_frac_0``, ``tx_frac_1``, ...,
        ``tx_frac_{N-1}``: cosmic time fractions in (0, 1).

    Returns
    -------
    array (n_age,)
        SFR at each lookback time (Msun/yr), non-negative.

    Notes
    -----
    Default ``age_universe_yr=13.47e9`` matches the dense_basis cosmology
    (FlatLambdaCDM H0=70, Om0=0.3). Override for other cosmologies or
    redshifts.

    The SFR derivative uses ``np.diff`` on the cumulative mass curve
    (matching dense_basis lines 165-168), scaled by
    ``sfh_scale = 10^logM / (age_universe_gyr * 1e9 / res)``.

    References
    ----------
    - Iyer & Gawiser (2017), ApJ 838, 127 (arXiv:1702.04371).
    - Iyer et al. (2019), ApJ 879, 116 (arXiv:1901.02877).
    """
    # --- Validate and collect tx fractions ---
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

    # Enforce ordering and physical bounds
    tx_fracs = jnp.sort(tx_fracs)
    tx_fracs = jnp.clip(tx_fracs, 0.02, 0.99)

    # --- Build quantile points (matching dense_basis) ---
    time_q, mass_q, yerr = _build_quantile_points(
        tx_fracs, n_param, log_total_mass, log_sfr_inst, age_universe_yr
    )

    # --- GP kernel hyperparameters (dense_basis convention) ---
    # variance = np.var(y), length_scale = np.median(y)
    # where y = mass_quantiles (the training y-values)
    variance = jnp.var(mass_q)
    length_scale = jnp.maximum(jnp.median(mass_q), _LENGTH_SCALE_FLOOR)

    # --- GP interpolation on dense grid ---
    # x_pred = linspace(0, 1, res) — cosmic time fraction
    # (dense_basis gp_interpolator line 88)
    t_eval = jnp.linspace(0.0, 1.0, _DEFAULT_RES)
    m_cumul = gp_interpolate(time_q, mass_q, yerr, t_eval, variance, length_scale)

    # --- SFR = sfh_scale * diff(cumulative_mass) ---
    # (dense_basis lines 165-168)
    # sfh_scale = 10^logM / (age_gyr * 1e9 / res)
    age_universe_gyr = age_universe_yr / 1e9
    sfh_scale = 10.0**log_total_mass / (age_universe_gyr * 1e9 / _DEFAULT_RES)
    sfr = jnp.diff(m_cumul) * sfh_scale
    sfr = jnp.maximum(sfr, 0.0)
    # Prepend zero for the Big Bang bin (matching dense_basis line 168)
    sfr = jnp.concatenate([jnp.array([0.0]), sfr])

    # --- Convert cosmic time fraction → lookback time ---
    # dense_basis: timeax = time_arr_interp * cosmo.age(z).value
    # This is COSMIC time (0=BB, age=now). We convert to lookback.
    t_cosmic_yr = t_eval * age_universe_yr
    t_lookback_yr = age_universe_yr - t_cosmic_yr

    # Reverse to get ascending lookback (0=present → large=old)
    t_lookback_yr = t_lookback_yr[::-1]
    sfr = sfr[::-1]

    # --- Interpolate onto the requested age grid ---
    result = jnp.interp(age_yr, t_lookback_yr, sfr)
    return jnp.maximum(result, 0.0)
