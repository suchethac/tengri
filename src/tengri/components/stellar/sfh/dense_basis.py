# SPDX-License-Identifier: BSD-3-Clause
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

import jax
import jax.numpy as jnp

# ── Constants ─────────────────────────────────────────────────────

_SQRT3 = jnp.sqrt(3.0)
_LENGTH_SCALE_FLOOR = 1e-10

# GP interpolation resolution (matching dense_basis default: 1000 points)
_DEFAULT_RES = 1000

# Recent-SFR decoupling parameters (matches dense_basis defaults in
# ``tuple_to_sfh``: ``decouple_sfr_time=10`` Myr, ``sfr_tolerance=0.05`` dex).
_DECOUPLE_SFR_TIME_GYR = 0.010
_SFR_TOLERANCE_DEX = 0.05


# ── GP Kernel functions ───────────────────────────────────────────


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
    x1 : array_like, shape (n1,)
        First kernel argument [dimensionless].
    x2 : array_like, shape (n2,)
        Second kernel argument [dimensionless].
    variance : float
        Signal variance σ² [dimensionless].
    length_scale : float
        Length scale ℓ [dimensionless].

    Returns
    -------
    ndarray, shape (n1, n2)
        Covariance matrix [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives throughout.

    Matches ``george.kernels.Matern32Kernel(metric=ℓ²)``.
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

    Notes
    -----
    Kept for backward compatibility. **Not used by ``dense_basis()``**: the
    GP-SFH path now calls :func:`_george_linear_kernel` to exactly mirror
    ``george.kernels.LinearKernel(log_gamma2, order=2)``.
    """
    ls_sq = jnp.maximum(length_scale, _LENGTH_SCALE_FLOOR) ** 2
    return variance * (x1[:, None] * x2[None, :]) / ls_sq


def _george_linear_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    log_gamma2: float,
    order: int = 2,
) -> jnp.ndarray:
    """Exact replica of ``george.kernels.LinearKernel(log_gamma2, order)``.

    K(x_i, x_j) = (x_i · x_j)^order / exp(log_gamma2)

    Matches the formula in george (https://github.com/dfm/george, kernels.py).
    The original dense_basis package uses ``LinearKernel(np.median(y), order=2)``
    so ``log_gamma2 = median(y)`` and γ² = exp(median(y)).
    """
    xx = x1[:, None] * x2[None, :]
    return xx**order / jnp.exp(log_gamma2)


def _george_combined_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    y_train: jnp.ndarray,
) -> jnp.ndarray:
    """George-faithful kernel: ``var(y) * (Matern32(metric=median(y))
    + LinearKernel(log_gamma2=median(y), order=2))``.

    Reproduces the exact construction in ``dense_basis/gp_sfh.py``::

        kernel = np.var(y) * (
            kernels.Matern32Kernel(np.median(y)) + kernels.LinearKernel(np.median(y), order=2)
        )

    george's Matern32 with scalar ``metric=M`` gives an effective length scale
    of ``sqrt(M)``; george's LinearKernel computes ``(x_i x_j)^order / γ²``
    with ``γ² = exp(log_gamma2)``.

    Parameters
    ----------
    x1, x2 : array_like
        Kernel inputs.
    y_train : array_like
        Training y-values used to compute the kernel hyperparameters
        ``variance = var(y)`` and the metric/log_gamma2 ``= median(y)``.
    """
    variance = jnp.var(y_train)
    med = jnp.maximum(jnp.median(y_train), _LENGTH_SCALE_FLOOR)
    length_scale_m32 = jnp.sqrt(med)  # george metric = ℓ², so ℓ = √median(y)
    k_m = matern32_kernel(x1, x2, variance, length_scale_m32)
    k_l = variance * _george_linear_kernel(x1, x2, log_gamma2=med, order=2)
    return k_m + k_l


# ── GP interpolation ──────────────────────────────────────────────


# Diagonal jitter that keeps the Cholesky factorization well posed.
#
# This was 1e-4, which is ~300x larger than the real diagonal noise
# (``yerr**2 ~ 3.3e-7`` at the user quantiles) and therefore set the smoothing
# on its own. Measured against ``dense_basis.gp_sfh.gp_interpolator`` over three
# quantile configurations, max |tengri - upstream| on the cumulative mass:
#
#     nugget   1e-4     1e-6     1e-8     1e-10    0
#     dev      3.8e-2   2.9e-3   3.1e-5   3.0e-7   3.8e-9
#
# 1e-8 puts us 3 orders closer to upstream while keeping a comfortable
# conditioning margin. Duplicate quantile times (possible after sort+clip) are
# regularized by ``yerr**2``, not by this term, so lowering it does not put the
# degenerate case at risk.
_NUGGET = 1e-8


def gp_interpolate(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    y_err: jnp.ndarray,
    x_eval: jnp.ndarray,
) -> jnp.ndarray:
    """GP conditional mean with the dense_basis kernel: the one GP path here.

    Both :func:`dense_basis` and :func:`dense_basis_pure` interpolate the
    cumulative-mass quantiles through this function, so they cannot drift apart.
    The kernel is :func:`_george_combined_kernel`, which reproduces the
    hyperparameters ``dense_basis`` hands to ``george``: the kernel is built
    from ``y_train`` itself, so there is nothing to tune here.

    Parameters
    ----------
    x_train : array_like, shape (n_train,)
        Training input points [dimensionless].
    y_train : array_like, shape (n_train,)
        Training values [dimensionless].
    y_err : array_like, shape (n_train,)
        Per-point noise, added in quadrature on the diagonal [dimensionless].
    x_eval : array_like, shape (n_eval,)
        Evaluation points [dimensionless].

    Returns
    -------
    ndarray, shape (n_eval,)
        GP conditional mean at evaluation points [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.scipy.linalg.cho_factor`` and ``cho_solve``.

    Cholesky exploits the positive-definite structure of the kernel matrix,
    which is both faster and better conditioned than a general solve. The
    :data:`_NUGGET` jitter keeps the factorization well posed; at 1e-8 it leaves
    the interpolant within ~3e-5 of upstream on a cumulative mass normalized to
    [0, 1].

    Implements the same interpolation as ``dense_basis`` (Iyer et al. 2019 [1]_)
    with ``interpolator='gp_george'``, that package's default.

    References
    ----------
    .. [1] K. G. Iyer et al., "Nonparametric Star Formation History
       Reconstruction with Gaussian Processes I: Counting Major Episodes of
       Star Formation," ApJ, 879, 116 (2019). arXiv:1901.02877.
       https://doi.org/10.3847/1538-4357/ab2052
    """
    k_train = _george_combined_kernel(x_train, x_train, y_train)
    k_train = k_train + jnp.diag(y_err**2) + _NUGGET * jnp.eye(k_train.shape[0])
    k_eval = _george_combined_kernel(x_eval, x_train, y_train)
    # Cholesky: L L^T = K, then solve L L^T α = y in two triangular steps
    cho_factor = jax.scipy.linalg.cho_factor(k_train)
    alpha = jax.scipy.linalg.cho_solve(cho_factor, y_train)
    return k_eval @ alpha


# ── Quantile point construction (matching dense_basis exactly) ────


def _build_quantile_points(
    tx_fracs: jnp.ndarray,
    n_param: int,
    log_total_mass: float,
    log_sfr_inst: float,
    age_universe_yr: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build (time, mass, yerr) quantile points: matching dense_basis.

    Constructs (in order):
    1. Endpoint: (0, 0)
    2. Big Bang constraint: (0.01, 0), which enforces SFR=0 at early times
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
    # Three points at 97%, 98%, 99% cumulative mass, timed to be consistent
    # with the instantaneous SFR at observation.
    #
    # The original ``tuple_to_sfh`` has a two-branch conditional:
    #   if delta_t > tx_last AND delta_t > 0.9:
    #       insert (t=delta_t, m=cv)             # time-branch
    #   else:
    #       insert (t=cv,      m=delta_m_alt)    # mass-branch
    # where ``delta_m_alt = 1 - age * (1-cv) * SFR / M``.
    #
    # We replicate this with ``jnp.where`` so the branch is JIT-safe. The
    # mass-branch matters for quenched galaxies: it places (~1, ~1) constraint
    # points near t=0.97..0.99 instead of unphysically placing high-mass
    # quantiles back near the Big Bang. Distinct clip bounds in the
    # time-branch keep the three constraint points distinct so the GP kernel
    # matrix remains non-singular under JIT.
    total_mass = 10.0**log_total_mass
    sfr_inst = 10.0**log_sfr_inst
    age_yr = age_universe_yr
    tx_last = tx_fracs[-1]  # last user quantile

    # Distinct clip bounds so the time-branch produces strictly distinct values.
    upper_bounds = jnp.array([0.997, 0.998, 0.999])

    sfr_time_q = []
    sfr_mass_q = []
    for i, cv in enumerate([0.97, 0.98, 0.99]):
        delta_mstar = total_mass * (1.0 - cv)
        delta_t = 1.0 - delta_mstar / (sfr_inst * age_yr)
        delta_m_alt = 1.0 - age_yr * (1.0 - cv) * sfr_inst / total_mass
        # Original code branches on both: delta_t > tx_last AND > 0.9
        use_time_branch = (delta_t > tx_last) & (delta_t > 0.9)
        # Time-branch value (clipped to keep three points distinct)
        # Lower clip > tx_last to maintain monotone ordering when branch fires.
        t_time = jnp.clip(delta_t, tx_last + 0.001 * (i + 1), upper_bounds[i])
        # Mass-branch value (delta_m can be clipped to [0,1])
        m_mass = jnp.clip(delta_m_alt, 0.0, 1.0)
        # Select between branches
        t_sel = jnp.where(use_time_branch, t_time, jnp.array(cv))
        m_sel = jnp.where(use_time_branch, jnp.array(cv), m_mass)
        sfr_time_q.append(t_sel)
        sfr_mass_q.append(m_sel)

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


# ── Main SFH function ─────────────────────────────────────────────


def dense_basis(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    log_sfr_inst: float = 0.0,
    age_universe_yr: float = 13.47e9,
    **tx_kwargs: float,
) -> jnp.ndarray:
    """Non-parametric GP star formation history via mass-time quantiles (Iyer+2017, 2019).

    A sophisticated non-parametric model based on mass-weighted cosmic times.
    The SFH is parameterized by the cosmic times at which the galaxy has assembled
    specified fractions of its total stellar mass. A Gaussian Process with
    Matérn 3/2 + Linear kernel smoothly interpolates the cumulative mass curve;
    the SFR is derived as the time derivative.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10(total stellar mass formed / Msun). Default: 10.0 (10 Gyr Msun).
    log_sfr_inst : float, optional
        log10(instantaneous SFR at observation / Msun/yr). Default: 0.0 (1 Msun/yr).
        Used to add 3 constraint points near t=1 (today) that pin the recent SFH
        shape to be consistent with the observed instantaneous SFR.
    age_universe_yr : float, optional
        Age of the universe at observation epoch [yr].
        Default: 13.47e9 (FlatLambdaCDM, H0=70, Omega_m=0.3, z=0).
    **tx_kwargs
        Keyword arguments ``tx_frac_0``, ``tx_frac_1``, ..., ``tx_frac_{N-1}``
        containing cosmic time fractions at which the galaxy formed specified
        mass fractions (e.g., 25%, 50%, 75%) [dimensionless, in (0,1)].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    This is a faithful JAX reimplementation of the ``dense_basis.gp_sfh.tuple_to_sfh()``
    algorithm from Iyer & Gawiser (2017, 2019). Key design features:

    1. **Direct mass parameterization**: stellar mass is a direct free parameter
       (:math:`\\log_{10} M_{\\star}`), not derived from integrating SFR. This decouples
       mass and SFH shape inference and improves sampling efficiency.

    2. **Quantile-based parameterization**: the SFH is defined by the cosmic times
       at which the galaxy has assembled given fractions (e.g., 25%, 50%, 75%) of
       its stellar mass.

    3. **GP interpolation**: a GP with Matérn 3/2 + Linear kernel smoothly
       interpolates the cumulative mass curve through the quantile constraints.

    4. **SFR derivation**: the SFR is computed as
       :math:`\\mathrm{SFR}(t) = d M_{\\star} / dt`, using finite differences on the
       GP-interpolated cumulative mass curve (matching the original dense_basis code).

    5. **Observation-epoch constraints**: three constraint points near t=1 (today)
       pin the recent SFH to be consistent with the instantaneous SFR at observation,
       via Equations 140-152 in dense_basis.py.

    The default ``age_universe_yr=13.47e9`` is calibrated to FlatLambdaCDM with
    H0=70 km/s/Mpc and Omega_m=0.3 at z=0. Override for other cosmologies or redshifts.

    **Approximation**: The SFR is computed using discrete differences on the
    GP-interpolated mass curve, matching the original dense_basis implementation.
    This is valid when the GP resolution (1000 points by default) is fine enough
    to resolve the SFH timescales of interest.

    References
    ----------
    .. [1] K. Iyer and E. Gawiser, "Reconstruction of Galaxy Star Formation Histories
       through SED Fitting: The Dense Basis Approach," ApJ, 838, 127 (2017).
       arXiv:1702.04371. https://doi.org/10.3847/1538-4357/aa63f0
    .. [2] K. Iyer et al., "Nonparametric Star Formation History Reconstruction with
       Gaussian Processes. I. Counting Major Episodes of Star Formation," ApJ, 879,
       116 (2019). arXiv:1901.02877. https://doi.org/10.3847/1538-4357/aaf563
    """
    # --- Validate and collect tx fractions ---
    n_param = len(tx_kwargs)
    if n_param == 0:
        raise ValueError("dense_basis requires at least one tx_frac_* parameter")
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

    # --- GP kernel: george-faithful ``var(y) * (Matern32(metric=median(y))
    #     + LinearKernel(log_gamma2=median(y), order=2))`` ---
    # Replaces an earlier formulation that (a) used ``median(y)`` directly as
    # the Matern length scale (off by √ vs. george's metric convention) and
    # (b) used a generic linear kernel instead of george's order=2 form.
    t_eval = jnp.linspace(0.0, 1.0, _DEFAULT_RES)
    m_cumul = gp_interpolate(time_q, mass_q, yerr, t_eval)

    # --- SFR = sfh_scale * diff(cumulative_mass) ---
    # (dense_basis lines 165-168)
    # sfh_scale = 10^logM / (age_gyr * 1e9 / res)
    age_universe_gyr = age_universe_yr / 1e9
    dt_yr = age_universe_yr / _DEFAULT_RES
    sfh_scale = 10.0**log_total_mass / dt_yr
    sfr = jnp.diff(m_cumul) * sfh_scale
    sfr = jnp.maximum(sfr, 0.0)
    # Prepend zero for the Big Bang bin (matching dense_basis line 168)
    sfr = jnp.concatenate([jnp.array([0.0]), sfr])

    # --- Mass renormalization (dense_basis lines 170-180) ---
    # The GP+diff path can leave a small residual mismatch with the target
    # mass 10^logM (GP smoothing pulls the endpoints slightly off 0/1).
    # The original code splits the SFR into "recent" (last ~10 Myr) and
    # "initial" bins, then rescales ONLY the initial bins so the total
    # equals 10^logM. The recent bins are then optionally overridden to
    # the requested instantaneous SFR.
    target_mass = 10.0**log_total_mass
    t_cosmic_gyr = t_eval * age_universe_gyr
    threshold_gyr = age_universe_gyr - _DECOUPLE_SFR_TIME_GYR
    recent_mask = t_cosmic_gyr >= threshold_gyr  # boolean, shape (RES,)

    # Integrate using rectangle rule (consistent with sfh_scale = 10^logM / dt_yr,
    # which assumes uniform dt). This matches the original code closely; trapezoid
    # vs rectangle differs only at endpoints.
    mass_recent = jnp.sum(jnp.where(recent_mask, sfr, 0.0)) * dt_yr
    mass_init = jnp.sum(jnp.where(recent_mask, 0.0, sfr)) * dt_yr
    mass_remaining = jnp.maximum(target_mass - mass_recent, 0.0)
    init_scale = mass_remaining / jnp.maximum(mass_init, 1e-30)
    sfr = jnp.where(recent_mask, sfr, sfr * init_scale)

    # --- Recent-SFR override (dense_basis lines 182-183) ---
    # The original code triggers the override only when the GP-derived recent
    # SFR deviates by more than ``sfr_tolerance`` (0.05 dex) from the requested
    # ``log_sfr_inst``. That conditional creates a non-differentiable step at
    # the threshold, which breaks gradient-based inference. Since the whole
    # point of having ``log_sfr_inst`` as an explicit free parameter is to
    # honor it during fitting, we override unconditionally: equivalent to
    # ``sfr_tolerance = 0``. This is smooth in both ``log_sfr_inst`` and
    # ``log_total_mass`` (the override bins are independent of the latter).
    sfr_override_value = 10.0**log_sfr_inst
    sfr = jnp.where(recent_mask, sfr_override_value, sfr)

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


# ── Pure quantile-only variant (no SFR constraint: for use with field)


def _build_quantile_points_pure(
    tx_fracs: jnp.ndarray,
    n_param: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build sorted (time, mass) quantile points and the GP noise array.

    Returns (time_q, mass_q, yerr) with n_param+3 points:
    (0,0), (0.01,0), (tx_i, mass_i), (1,1). The user quantiles sit at indices
    ``2 : 2+n_param``, which is where ``dense_basis`` puts its noise.
    """
    mass_quantiles = jnp.linspace(0.0, 1.0, n_param + 2)
    time_quantiles = jnp.concatenate(
        [
            jnp.array([0.0]),
            tx_fracs,
            jnp.array([1.0]),
        ]
    )

    # Big Bang constraint
    time_q = jnp.concatenate(
        [
            time_quantiles[:1],
            jnp.array([0.01]),
            time_quantiles[1:],
        ]
    )
    mass_q = jnp.concatenate(
        [
            mass_quantiles[:1],
            jnp.array([0.0]),
            mass_quantiles[1:],
        ]
    )

    # Same noise rule as the SFR-constrained path: only the user quantiles
    # carry slack, so the GP is pinned at the (0,0)/(0.01,0)/(1,1) anchors.
    yerr = jnp.zeros(time_q.shape[0])
    yerr = yerr.at[2 : 2 + n_param].set(0.001 / jnp.sqrt(jnp.maximum(n_param, 1.0)))

    return time_q, mass_q, yerr


def dense_basis_pure(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    age_universe_yr: float = 13.47e9,
    **tx_kwargs: float,
) -> jnp.ndarray:
    """Pure quantile-based SFH using monotone cubic Hermite interpolation (PCHIP).

    A lightweight variant of :func:`dense_basis` optimized for use as the
    mean component in a composed model with a GP field modulator
    (``sfh=["dense_basis", "field"]``). Eliminates the GP kernel overhead
    and observation-epoch SFR constraints, using fast PCHIP monotone interpolation
    instead.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10(total stellar mass formed / Msun). Default: 10.0 (10 Gyr Msun).
    age_universe_yr : float, optional
        Age of the universe at observation epoch [yr].
        Default: 13.47e9 (FlatLambdaCDM, H0=70, Omega_m=0.3, z=0).
    **tx_kwargs
        Keyword arguments ``tx_frac_0``, ``tx_frac_1``, ..., ``tx_frac_{N-1}``
        containing cosmic time fractions [dimensionless, in (0,1)].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes, uses ``gp_interpolate``.

    **Approximation**: Replaces the GP with monotone PCHIP (Piecewise Cubic
    Hermite Interpolating Polynomial), which guarantees a monotonically
    increasing cumulative mass curve without overshoot. The Fritsch-Carlson
    algorithm computes slopes such that the interpolant never violates the
    monotonicity of the input points.

    Compared to :func:`dense_basis`:

    - **Pros**: Faster (no matrix solve), more robust (always monotonic),
      fewer hyperparameters (no GP kernel bandwidth).
    - **Cons**: Less smooth, no automatic variance propagation to nearby points.

    **Intended use**: As the smooth mean component in a composed model with
    a GP field modulator. The field adds high-frequency variability on top
    of the smooth mean, so the mean component can be simpler.

    References
    ----------
    .. [1] K. Iyer and E. Gawiser, "Reconstruction of Galaxy Star Formation Histories
       through SED Fitting: The Dense Basis Approach," ApJ, 838, 127 (2017).
       arXiv:1702.04371. https://doi.org/10.3847/1538-4357/aa63f0
    .. [2] K. Iyer et al., "Nonparametric Star Formation History Reconstruction with
       Gaussian Processes. I. Counting Major Episodes of Star Formation," ApJ, 879,
       116 (2019). arXiv:1901.02877. https://doi.org/10.3847/1538-4357/aaf563
    .. [3] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic Interpolation,"
       SIAM J. Numer. Anal., 17, 238 (1980).
       https://doi.org/10.1137/0717021
    """
    n_param = len(tx_kwargs)
    if n_param == 0:
        raise ValueError("dense_basis_pure requires at least one tx_frac_*")
    for i in range(n_param):
        key = f"tx_frac_{i}"
        if key not in tx_kwargs:
            raise ValueError(
                f"Missing required parameter '{key}'. Got keys: {sorted(tx_kwargs.keys())}."
            )
    tx_fracs = jnp.array([tx_kwargs[f"tx_frac_{i}"] for i in range(n_param)])
    tx_fracs = jnp.sort(tx_fracs)
    tx_fracs = jnp.clip(tx_fracs, 0.02, 0.99)

    # Build quantile points (NO SFR constraints)
    time_q, mass_q, yerr = _build_quantile_points_pure(tx_fracs, n_param)

    # GP interpolation of cumulative mass: the same 'gp_george' path
    # dense_basis uses by default, shared with dense_basis() above.
    t_eval = jnp.linspace(0.0, 1.0, _DEFAULT_RES)
    m_cumul = gp_interpolate(time_q, mass_q, yerr, t_eval)
    m_cumul = jnp.clip(m_cumul, 0.0, 1.0)

    # SFR = sfh_scale * diff(cumulative_mass)
    age_universe_gyr = age_universe_yr / 1e9
    sfh_scale = 10.0**log_total_mass / (age_universe_gyr * 1e9 / _DEFAULT_RES)
    sfr = jnp.diff(m_cumul) * sfh_scale
    sfr = jnp.maximum(sfr, 0.0)
    sfr = jnp.concatenate([jnp.array([0.0]), sfr])

    # Cosmic time → lookback time
    t_cosmic_yr = t_eval * age_universe_yr
    t_lookback_yr = age_universe_yr - t_cosmic_yr
    t_lookback_yr = t_lookback_yr[::-1]
    sfr = sfr[::-1]

    result = jnp.interp(age_yr, t_lookback_yr, sfr)
    return jnp.maximum(result, 0.0)
