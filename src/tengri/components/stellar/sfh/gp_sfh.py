# SPDX-License-Identifier: BSD-3-Clause
"""Gaussian Process realizations from Power Spectral Density functions.

Two modes:

- Stochastic: generate_gp_fourier() draws random GP realizations (for mocks)
- Deterministic: gp_from_xi() maps a fixed latent vector to a GP (for inference)

The GP is defined on a uniform log-age grid. The IFFT-based generation
implements the NIFTy correlated field model: s = IFFT(sqrt(P) * xi).

Key design: the latent vector xi ~ N(0, I) is the standardized variable
that samplers (geoVI, NUTS) explore. The PSD encodes the prior correlation.
"""

import jax
import jax.numpy as jnp
from jax import random

from tengri.utils.grid import DEFAULT_LOG_AGE_MAX, DEFAULT_LOG_AGE_MIN

# Default log10(age/yr) bounds for the SFH grid: 1 Myr → ~13.8 Gyr.
# Exposed as module constants so callers that need the static step size
# (e.g. ``StellarSEDComponent.apply`` under JIT, where indexing a traced
# ``jnp.linspace`` and calling ``float()`` would raise) can recompute it
# without re-tracing the grid.
#
# Aliases, not literals. Unifying ``make_log_age_grid`` alone leaves the same
# duplication one level down: ``log_age_grid_step`` below defaults off THESE
# constants while ``make_log_age_grid`` defaults off ``utils/grid``'s, so
# changing the canonical bounds would silently stop the analytic step size
# describing the grid the forward model is actually evaluated on. Measured on
# this branch before the alias: widening ``DEFAULT_LOG_AGE_MAX`` 10.14 -> 10.20
# left ``log_age_grid_step(256)`` 1.43 % adrift from the real grid spacing, and
# the contract test stayed green once its hardcoded 10.14 was updated: which is
# exactly what a maintainer making that change would do.
LOG_AGE_MIN: float = DEFAULT_LOG_AGE_MIN
LOG_AGE_MAX: float = DEFAULT_LOG_AGE_MAX


def log_age_grid_step(
    n_grid: int, log_age_min: float = LOG_AGE_MIN, log_age_max: float = LOG_AGE_MAX
) -> float:
    """Return the step size of :func:`make_log_age_grid` as a Python float.

    JIT-safe: takes ``n_grid`` (static config) and constants, never touches
    a traced array. Use this when a downstream function (e.g.
    ``compute_field_gp``) needs ``d_log_age`` as a static Python scalar.

    Parameters
    ----------
    n_grid : int
        Number of grid points (must be ≥ 2).
    log_age_min, log_age_max : float, optional
        Grid bounds; default to :data:`LOG_AGE_MIN` / :data:`LOG_AGE_MAX`.

    Returns
    -------
    float
        ``(log_age_max - log_age_min) / (n_grid - 1)``.
    """
    return (log_age_max - log_age_min) / (n_grid - 1)


# Re-exported, not redefined. This function had a second, byte-for-byte
# equivalent definition here until #1402; the two were consumed on opposite
# sides of the forward/inference boundary (components/stellar/component.py
# vs inference/standardized.py), so a fix landing in one would have silently
# desynced the SFH age grid from the grid inference standardizes against.
from tengri.utils.grid import make_log_age_grid as make_log_age_grid


def gp_from_xi(xi: jnp.ndarray, sqrt_power: jnp.ndarray, n_points: int) -> jnp.ndarray:
    r"""Deterministic GP realization from standardized latent vector.

    Maps a standardized Gaussian random vector to a correlated GP field via
    Fourier-space multiplication with a spectral amplitude operator. This
    is the core function used during inference and mock generation.

    Parameters
    ----------
    xi : array_like, shape (n_points,)
        Standardized latent vector :math:`\xi \sim \mathcal{N}(0, I)` under the prior.
    sqrt_power : array_like, shape (n_freq,)
        Amplitude operator :math:`\sqrt{P(\omega) / d_{\rm grid}}` at rfft frequencies
        (pre-compute with :func:`psd_to_sqrt_power`). [dimensionless]
    n_points : int
        Number of grid points (should match the length of xi).

    Returns
    -------
    ndarray, shape (n_points,)
        GP realization on the log-age grid [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.fft.rfft`` and ``jnp.fft.irfft``.

    **Gradient-safe**: yes, differentiable w.r.t. sqrt_power.

    Implements the NIFTy correlated field model:

    .. math::

        s = \mathrm{IFFT}(\sqrt{P} \cdot \hat{\xi})

    The rfft (real FFT) preserves Hermitian symmetry for real-valued output and
    ensures correct variance normalization: :math:`E[|\mathrm{rfft}(\xi)_k|^2] = N`,
    so with :math:`\sqrt{P/\Delta x}` we recover :math:`\mathrm{Var}[s] = \int P(f) df`.

    This is the primary function called during MCMC inference and mock galaxy generation.
    The sampler proposes values of :math:`\xi` and this function maps them to
    correlated SFH realizations.

    References
    ----------
    .. [1] Selig et al., "NIFTY - Numerical Information Field Theory. A versatile
       PYTHON library for signal inference," A&A, 554, A26 (2013).
       arXiv:1301.4499. https://doi.org/10.1051/0004-6361/201321236

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import gp_from_xi, make_log_age_grid, compute_sqrt_power_drw
    >>> n = 64
    >>> grid = make_log_age_grid(n)
    >>> d = float(grid[1] - grid[0])
    >>> sqrt_power = compute_sqrt_power_drw(n, d, psd_sigma=1.0, psd_tau_yr=1e8)
    >>> xi = jnp.zeros(n)
    >>> sfh = gp_from_xi(xi, sqrt_power, n)
    >>> sfh.shape
    (64,)
    """
    xi_hat = jnp.fft.rfft(xi)
    coeffs = sqrt_power * xi_hat
    return jnp.fft.irfft(coeffs, n=n_points)


# Relative Cholesky jitter for the linear-time DRW covariance. The young-age
# block (grid steps << tau) is near-rank-deficient (rows nearly identical), so a
# small diagonal loading keeps it positive-definite. Scaled by the covariance
# variance so it is amplitude-independent.
_DRW_CHOLESKY_JITTER: float = 1e-6


def drw_linear_gp_from_xi(xi, psd_sigma_dex, psd_tau_yr, log_age_grid):
    r"""DRW Gaussian-process realization stationary in LINEAR (physical) time.

    Builds the damped-random-walk covariance directly in cosmic time and samples
    it on the (log-spaced) SFH age grid:

    .. math::

        K_{ij} = (\sigma \ln 10)^2 \, \exp(-|t_i - t_j| / \tau),
        \qquad t_i = 10^{u_i},

    where :math:`u_i` is the log10-age grid, :math:`\sigma` is the modulation
    amplitude in **dex**, and :math:`\tau` the physical decorrelation timescale
    [yr]. The realization is ``gp_x = L \xi`` with ``K = L L^T`` (Cholesky) and
    ``xi ~ N(0, I)``: the same standardized latent the samplers explore.

    Unlike the Fourier/log-age construction (:func:`gp_from_xi` +
    :func:`compute_sqrt_power_drw`), the correlation length is a **fixed number
    of years at every age**: gas-cycling burstiness is a physical-time process,
    not a fixed number of dex; the log grid is only the sampling. The natural-log
    variance is stationary at :math:`(\sigma \ln 10)^2`, so the modulation std is
    exactly :math:`\sigma` dex and the log-normal bias correction is
    :math:`K(0)/2 = (\sigma \ln 10)^2 / 2`.

    Parameters
    ----------
    xi : array_like, shape (n,)
        Standardized latent vector, :math:`\xi \sim \mathcal{N}(0, I)`.
    psd_sigma_dex : float
        Modulation amplitude [dex] = std of :math:`\log_{10}(\mathrm{SFR})`.
    psd_tau_yr : float
        Physical DRW decorrelation timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age/yr)`` grid the SFH is represented on.

    Returns
    -------
    gp_x : ndarray, shape (n,)
        Natural-log SFH modulation (applied as ``exp(gp_x - k0_half)``).
    k0_half : ndarray, scalar
        Log-normal bias correction :math:`(\sigma \ln 10)^2 / 2`.

    Notes
    -----
    **JIT/grad/vmap-safe**: dense Cholesky of an ``(n, n)`` matrix via
    ``jnp.linalg.cholesky`` (:math:`O(n^3)`; ``n ~ 256`` is sub-ms on CPU and
    differentiable). A relative jitter keeps the near-rank-deficient young-age
    block positive-definite. At old ages where the grid step exceeds
    :math:`\tau`, the covariance is effectively diagonal: burstiness below the
    local grid resolution is unrepresentable there (and physically averages out).

    References
    ----------
    .. [1] K. G. Iyer et al., "The star formation history and variability of
       galaxies," MNRAS, 498, 430 (2020). [physical decorrelation timescale]
    .. [2] N. Caplar & S. Tacchella, MNRAS, 487, 3845 (2019). [PSD amplitude, dex]
    """
    ln10 = jnp.log(10.0)
    t = 10.0 ** jnp.asarray(log_age_grid)
    var = (jnp.asarray(psd_sigma_dex) * ln10) ** 2
    dt = jnp.abs(t[:, None] - t[None, :])
    cov = var * jnp.exp(-dt / jnp.asarray(psd_tau_yr))
    n = t.shape[0]
    cov = cov + _DRW_CHOLESKY_JITTER * var * jnp.eye(n)
    gp_x = jnp.linalg.cholesky(cov) @ jnp.asarray(xi)
    return gp_x, 0.5 * var


def drw_innovations_gp_from_xi(xi, psd_sigma_dex, psd_tau_yr, log_age_grid):
    r"""DRW realization via the OU state-space (innovations) recursion.

    Realizes the *same* linear-time damped-random-walk field as
    :func:`drw_linear_gp_from_xi` (same covariance, same prior) but through the
    exact first-order Markov (Ornstein–Uhlenbeck) forward recursion rather than a
    dense Cholesky factor:

    .. math::

        s_0 &= \sqrt{\mathrm{var}}\; \xi_0, \\
        s_i &= \rho_i\, s_{i-1} + \sqrt{\mathrm{var}\,(1 - \rho_i^2)}\; \xi_i,
        \qquad \rho_i = \exp(-\Delta t_i / \tau),

    with :math:`\mathrm{var} = (\sigma \ln 10)^2`, physical times
    :math:`t_i = 10^{u_i}` from the log-age grid, and step gaps
    :math:`\Delta t_i = |t_i - t_{i-1}| \ge 0`. Because a DRW is exactly Markov, this
    recursion reproduces :math:`K_{ij} = \mathrm{var}\,\exp(-|t_i-t_j|/\tau)` to
    machine precision.

    The gaps are taken as *magnitudes*, so the grid may run in either direction: a
    DRW kernel depends only on :math:`|t_i - t_j|`, and along any monotone sequence
    the consecutive :math:`|\Delta t|` telescope to exactly that. A descending grid
    therefore yields a valid square root of the same :math:`K`, transposed onto the
    reversed node order.

    **It is the same square root, not an alternative one.** Unrolling the recursion
    gives :math:`s = M \xi` with :math:`M` lower-triangular and positive on the
    diagonal (:math:`M_{00} = \sqrt{\mathrm{var}}`,
    :math:`M_{ii} = \sqrt{\mathrm{var}(1-\rho_i^2)}`), and :math:`M M^{T} = K`. The
    Cholesky factor is the *unique* lower-triangular matrix with positive diagonal
    satisfying :math:`L L^{T} = K`, hence :math:`M = L` exactly. The gain is
    therefore computational and numerical, not geometric: :math:`O(n)` instead of
    :math:`O(n^3)`, and no positive-definiteness jitter, so the realized prior is the
    exact :math:`K` rather than :math:`K + \epsilon\,\mathrm{var}\,I` (the dense path
    perturbs every realization by :math:`\sim 10^{-5}` relative; see
    ``_DRW_CHOLESKY_JITTER``).

    Because the :math:`\xi \to \mathrm{SFH}` map is numerically identical, this does
    **not** change the posterior geometry and is **not** a remedy for the #1301 HMC
    divergences: every *exact* square root of :math:`K(\tau)` carries the same
    :math:`\tau`-dependence, since the kernels at different :math:`\tau` do not
    commute and so share no :math:`\tau`-independent eigenbasis. Removing the
    :math:`\tau`-coupling requires changing the *representation* (the uniform
    linear-time Fourier basis, #1333), which changes the prior.

    Parameters
    ----------
    xi : array_like, shape (n,)
        Standardized latent vector, :math:`\xi \sim \mathcal{N}(0, I)`.
    psd_sigma_dex : float
        Modulation amplitude [dex] = std of :math:`\log_{10}(\mathrm{SFR})`.
    psd_tau_yr : float
        Physical DRW decorrelation timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age/yr)`` grid the SFH is represented on. Monotone: ascending is
        canonical, descending is equally valid (see above). A **non-monotone** grid
        has no DRW square root and the result is meaningless, though bounded.

    Returns
    -------
    gp_x : ndarray, shape (n,)
        Natural-log SFH modulation (applied as ``exp(gp_x - k0_half)``).
    k0_half : ndarray, scalar
        Log-normal bias correction :math:`(\sigma \ln 10)^2 / 2`.

    Notes
    -----
    **JIT/grad/vmap-safe**: the recursion is a single ``jax.lax.scan``, so :math:`O(n)`
    time, :math:`O(1)` memory, differentiable w.r.t. ``psd_sigma_dex``,
    ``psd_tau_yr`` and ``xi``. No dense matrix, so (unlike
    :func:`drw_linear_gp_from_xi`) there is no Cholesky and no
    positive-definiteness jitter. At young ages :math:`\Delta t_i \ll \tau`
    (:math:`\rho_i \to 1`) the field is strongly correlated; at old ages
    :math:`\Delta t_i \gg \tau` (:math:`\rho_i \to 0`) successive nodes become
    independent draws of variance ``var``: burstiness below the local grid
    resolution is unrepresentable there, matching :func:`drw_linear_gp_from_xi`.

    The innovation scale carries a ``clip(1 - rho**2, 0, None)`` floor so float
    round-off cannot drive the argument of the square root slightly negative. That
    floor is safe **only because** :math:`\Delta t_i` is a magnitude, which forces
    :math:`\rho_i \le 1` analytically. Computing the gaps as a signed
    :math:`t_i - t_{i-1}` instead makes the clip a guard that fails open: a
    descending grid gives :math:`\rho_i > 1`, the clip converts the resulting
    ``NaN`` into ``innov = 0``, and the recursion grows geometrically to a finite,
    unflagged :math:`\sim\!10^{17}\sigma` (#1370).

    References
    ----------
    .. [1] K. G. Iyer et al., "The star formation history and variability of
       galaxies," MNRAS, 498, 430 (2020). [physical decorrelation timescale]
    .. [2] N. Caplar & S. Tacchella, MNRAS, 487, 3845 (2019). [PSD amplitude, dex]
    """
    ln10 = jnp.log(10.0)
    t = 10.0 ** jnp.asarray(log_age_grid)
    xi = jnp.asarray(xi)
    var = (jnp.asarray(psd_sigma_dex) * ln10) ** 2
    sigma_s = jnp.sqrt(var)

    # Per-step correlation and fresh-innovation scale.
    #
    # ``abs`` is load-bearing, not defensive (#1370). On a descending grid
    # ``diff(t) < 0``, so ``rho > 1`` and ``1 - rho**2 < 0``; and the clip below
    # would turn the would-be-loud ``sqrt(negative) = NaN`` into ``innov = 0``,
    # leaving ``s_i = rho_i s_{i-1}`` with ``rho > 1``: silent exponential growth
    # (measured at 2.1e17 sigma_s, finite, no warning). Taking the magnitude makes
    # any *monotone* grid a valid square root of the same K instead, because the
    # DRW kernel depends on |t_i - t_j| and consecutive |dt| telescope to it.
    dt = jnp.abs(jnp.diff(t))
    rho = jnp.exp(-dt / jnp.asarray(psd_tau_yr))
    # With dt >= 0 above, rho <= 1 analytically; the clip now guards only float
    # round-off at rho -> 1, which is what it was always documented to do.
    innov = sigma_s * jnp.sqrt(jnp.clip(1.0 - rho**2, 0.0, None))

    s0 = sigma_s * xi[0]

    def _step(s_prev, step_inputs):
        rho_i, innov_i, xi_i = step_inputs
        s_i = rho_i * s_prev + innov_i * xi_i
        return s_i, s_i

    _, s_rest = jax.lax.scan(_step, s0, (rho, innov, xi[1:]))
    gp_x = jnp.concatenate([jnp.reshape(s0, (1,)), s_rest])
    return gp_x, 0.5 * var


def drw_unit_gp_from_xi(xi, psd_tau_yr, log_age_grid):
    r"""Unit-variance DRW realization: the amplitude-free square root.

    Identical to :func:`drw_innovations_gp_from_xi` with :math:`\sigma \ln 10 = 1`, so the
    implied operator is a square root of the *correlation* matrix rather than the
    covariance:

    .. math::

        C_{ij} = \exp(-|t_i - t_j| / \tau), \qquad u = \tilde{L}\,\xi,
        \qquad \tilde{L}\tilde{L}^{\!\top} = C

    with :math:`t_i = 10^{u_i}` [yr]. Factoring the amplitude out is exact because the
    recursion is **linear** in :math:`\sigma`: every term of
    :func:`drw_innovations_gp_from_xi` carries exactly one factor of
    :math:`\sigma_s = \sigma \ln 10`. That is what makes partial non-centering in the
    amplitude tractable (:func:`drw_partial_gp_from_zeta`).

    Parameters
    ----------
    xi : array_like, shape (n,)
        Standardized latent vector, :math:`\xi \sim \mathcal{N}(0, I)`.
    psd_tau_yr : float
        Damping timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age/yr)`` grid, monotone (either direction: see
        :func:`drw_innovations_gp_from_xi`).

    Returns
    -------
    u : ndarray, shape (n,)
        Unit-variance correlated field [dimensionless]; ``diag(Cov(u)) == 1``.

    Notes
    -----
    **JIT/grad/vmap compatible**: yes. **O(n)** time and memory; no dense
    :math:`n \times n` is ever formed.

    ``abs`` on the step gaps is load-bearing for the same reason as in
    :func:`drw_innovations_gp_from_xi` (#1370): a signed gap on a descending grid gives
    :math:`\rho > 1`, and the round-off clip would convert the resulting ``NaN`` into
    a silent geometric blow-up rather than an error.

    References
    ----------
    .. [1] K. G. Iyer et al., "The star formation history and variability of
       galaxies," MNRAS, 498, 430 (2020). [physical decorrelation timescale]
    """
    t = 10.0 ** jnp.asarray(log_age_grid)
    xi = jnp.asarray(xi)
    dt = jnp.abs(jnp.diff(t))
    rho = jnp.exp(-dt / jnp.asarray(psd_tau_yr))
    innov = jnp.sqrt(jnp.clip(1.0 - rho**2, 0.0, None))

    u0 = xi[0]

    def _step(u_prev, step_inputs):
        rho_i, innov_i, xi_i = step_inputs
        u_i = rho_i * u_prev + innov_i * xi_i
        return u_i, u_i

    _, u_rest = jax.lax.scan(_step, u0, (rho, innov, xi[1:]))
    return jnp.concatenate([jnp.reshape(u0, (1,)), u_rest])


def drw_partial_gp_from_zeta(zeta, psd_sigma_dex, psd_tau_yr, log_age_grid, centering=1.0):
    r"""Partially non-centered DRW field (#1355).

    Interpolates between the non-centered parameterization in use today and the fully
    centered one, following Papaspiliopoulos, Roberts & Skold [2]_:

    .. math::

        s = \sigma_s^{a}\,\tilde{L}(\tau)\,\zeta, \qquad
        \zeta \sim \mathcal{N}\!\left(0,\ \sigma_s^{2-2a} I\right)

    where :math:`\sigma_s = \sigma \ln 10` [dex, natural-log units], :math:`\tilde{L}`
    is the unit-variance square root (:func:`drw_unit_gp_from_xi`), and
    :math:`a \in [0, 1]` is ``centering``. Both ends give the same marginal
    :math:`s \sim \mathcal{N}(0, K)` with
    :math:`K_{ij} = \sigma_s^2 \exp(-|t_i-t_j|/\tau)`:

    * ``a = 1``: non-centered. The prior on :math:`\zeta` is :math:`\mathcal{N}(0, I)`,
      independent of :math:`\sigma`; the amplitude enters through the **likelihood**.
      This is the standardized parameterization and today's default.
    * ``a = 0``: centered. The transform is amplitude-free; :math:`\sigma` enters
      through the **prior**.

    Why it matters: at ``a = 1`` the map is *multiplicative* in
    :math:`(\sigma, \zeta)`, which is precisely Neal's funnel; a narrow neck at small
    :math:`\sigma` and a wide mouth at large :math:`\sigma`, with one step size for
    both. Lowering ``a`` moves amplitude dependence out of the map. PRS give the trade:
    non-centered is preferable for a prior-dominated block, centered for a
    data-dominated one.

    Parameters
    ----------
    zeta : array_like, shape (n,)
        Latent vector. Its prior is :math:`\mathcal{N}(0, \sigma_s^{2-2a} I)`: **not**
        :math:`\mathcal{N}(0, I)` unless ``centering == 1``. Pair it with
        :func:`drw_latent_log_prior`.
    psd_sigma_dex : float
        Modulation amplitude :math:`\sigma` [dex].
    psd_tau_yr : float
        Damping timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age/yr)`` grid, monotone.
    centering : float, optional
        Exponent :math:`a \in [0, 1]`. Default ``1.0``: today's non-centered field,
        reproduced through the original code path so the result is unchanged.

    Returns
    -------
    gp_x : ndarray, shape (n,)
        Correlated log-SFR modulation [natural log units].
    k0_over_2 : float
        Log-normal bias correction :math:`K(0)/2 = \sigma_s^2 / 2`. Independent of
        ``a``, since the marginal variance is.

    Notes
    -----
    **JIT/grad/vmap compatible**: yes. **O(n)**. ``centering`` is a build-time
    structural choice and must be a Python float, not a traced value: the ``a == 1``
    fast path is a Python-level branch so the default stays bit-identical to
    :func:`drw_innovations_gp_from_xi`, the production path.

    The posterior is **invariant** to ``a`` only when the latent prior carries its
    :math:`-n(1-a)\log\sigma_s` normalizer; see :func:`drw_latent_log_prior`. Omitting
    it yields a sampler that runs cleanly while targeting a different distribution for
    every ``a``.

    References
    ----------
    .. [1] K. G. Iyer et al., "The star formation history and variability of
       galaxies," MNRAS, 498, 430 (2020). [physical decorrelation timescale]
    .. [2] O. Papaspiliopoulos, G. O. Roberts & M. Skold, "A general framework for the
       parametrization of hierarchical models," Statistical Science, 22, 59 (2007).
       DOI: 10.1214/088342307000000014.
    """
    a = float(centering)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"centering must lie in [0, 1], got {a}.")
    if a == 1.0:
        # Bit-identical default: delegate to the PRODUCTION path (registry.py), which is
        # the jitter-free O(n) recursion. Not ``drw_linear_gp_from_xi``; that dense
        # Cholesky is retained only as the oracle and adds ``_DRW_CHOLESKY_JITTER``
        # (1e-6 * var) to the diagonal, so it reproduces K only to ~1e-6 relative.
        return drw_innovations_gp_from_xi(zeta, psd_sigma_dex, psd_tau_yr, log_age_grid)
    sigma_s = jnp.asarray(psd_sigma_dex) * jnp.log(10.0)
    u = drw_unit_gp_from_xi(zeta, psd_tau_yr, log_age_grid)
    return sigma_s**a * u, 0.5 * sigma_s**2


def drw_latent_log_prior(zeta, psd_sigma_dex, centering=1.0):
    r"""Log-prior of the partially non-centered latent (#1355).

    .. math::

        \log p(\zeta \mid \sigma) =
            -\frac{\zeta^{\!\top}\zeta}{2\,\sigma_s^{2-2a}}
            - \frac{n}{2}\log\!\left(2\pi\,\sigma_s^{2-2a}\right)

    with :math:`\sigma_s = \sigma \ln 10` and :math:`a` = ``centering``. At ``a = 1``
    this collapses to the standardized :math:`-\tfrac{1}{2}\zeta^{\!\top}\zeta` (plus a
    constant), which is what the rest of ``tengri`` assumes.

    Parameters
    ----------
    zeta : array_like, shape (n,)
        Latent vector from :func:`drw_partial_gp_from_zeta`.
    psd_sigma_dex : float
        Modulation amplitude :math:`\sigma` [dex].
    centering : float, optional
        Exponent :math:`a \in [0, 1]`. Default ``1.0``.

    Returns
    -------
    log_prior : float
        Log-density [nats], **fully normalized**.

    Notes
    -----
    **JIT/grad/vmap compatible**: yes.

    The :math:`-n(1-a)\log\sigma_s` half of the normalizer is not optional. It is the
    only term that couples the latent prior to :math:`\sigma` at :math:`a < 1`, and it
    is what makes the posterior invariant to ``a``. Drop it and the sampler still runs,
    reports nothing, and targets a different distribution at every ``a``: the
    signature is a recovered :math:`\sigma` that drifts with a knob that is supposed to
    be a change of coordinates.

    References
    ----------
    .. [1] O. Papaspiliopoulos, G. O. Roberts & M. Skold, "A general framework for the
       parametrization of hierarchical models," Statistical Science, 22, 59 (2007).
       DOI: 10.1214/088342307000000014.
    """
    a = float(centering)
    zeta = jnp.asarray(zeta)
    n = zeta.shape[0]
    sigma_s = jnp.asarray(psd_sigma_dex) * jnp.log(10.0)
    log_var = (2.0 - 2.0 * a) * jnp.log(sigma_s)
    var = jnp.exp(log_var)
    return -0.5 * jnp.sum(zeta**2) / var - 0.5 * n * (jnp.log(2.0 * jnp.pi) + log_var)


def generate_gp_fourier(key: jax.Array, sqrt_power: jnp.ndarray, n_points: int) -> jnp.ndarray:
    r"""Stochastic GP realization for mock galaxy generation.

    Draws a random standardized vector and maps it to a correlated GP field.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random key for reproducibility.
    sqrt_power : array_like, shape (n_freq,)
        Amplitude operator :math:`\sqrt{P(\omega) / d_{\rm grid}}` at rfft frequencies
        (pre-compute with :func:`psd_to_sqrt_power`). [dimensionless]
    n_points : int
        Number of grid points.

    Returns
    -------
    ndarray, shape (n_points,)
        GP realization on the log-age grid [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.random.normal`` and :func:`gp_from_xi`.

    This function is the primary interface for generating mock SFHs with
    stochastic variability. The random draw is always independent; for
    reproducibility, pass the same PRNGKey.

    See Also
    --------
    gp_from_xi : Deterministic GP mapping (used internally).
    generate_gp_batch : Generate multiple independent realizations.

    Examples
    --------
    >>> import jax
    >>> from tengri import generate_gp_fourier, make_log_age_grid, compute_sqrt_power_drw
    >>> n = 64
    >>> grid = make_log_age_grid(n)
    >>> d = float(grid[1] - grid[0])
    >>> sqrt_power = compute_sqrt_power_drw(n, d, psd_sigma=1.0, psd_tau_yr=1e8)
    >>> key = jax.random.PRNGKey(0)
    >>> sfh = generate_gp_fourier(key, sqrt_power, n)
    >>> sfh.shape
    (64,)
    """
    xi = random.normal(key, shape=(n_points,))
    return gp_from_xi(xi, sqrt_power, n_points)


def generate_gp_batch(
    key: jax.Array, sqrt_power: jnp.ndarray, n_points: int, n_realizations: int
) -> jnp.ndarray:
    r"""Batch of independent GP realizations via vectorization.

    Generates multiple independent SFH realizations in parallel using vmap.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random key (will be split into n_realizations independent keys).
    sqrt_power : array_like, shape (n_freq,)
        Amplitude operator :math:`\sqrt{P(\omega) / d_{\rm grid}}` at rfft frequencies.
        [dimensionless]
    n_points : int
        Number of grid points.
    n_realizations : int
        Number of independent realizations to generate.

    Returns
    -------
    ndarray, shape (n_realizations, n_points)
        Batch of independent GP realizations [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.vmap`` over :func:`generate_gp_fourier`.

    Each realization is independent with the specified PSD structure.
    This function is useful for generating mock catalogs or computing
    uncertainties via Monte Carlo sampling.

    See Also
    --------
    generate_gp_fourier : Single realization.

    Examples
    --------
    >>> import jax
    >>> from tengri import generate_gp_batch, make_log_age_grid, compute_sqrt_power_drw
    >>> n = 64
    >>> grid = make_log_age_grid(n)
    >>> d = float(grid[1] - grid[0])
    >>> sqrt_power = compute_sqrt_power_drw(n, d, psd_sigma=1.0, psd_tau_yr=1e8)
    >>> key = jax.random.PRNGKey(0)
    >>> batch = generate_gp_batch(key, sqrt_power, n_points=n, n_realizations=10)
    >>> batch.shape
    (10, 64)
    """
    keys = random.split(key, n_realizations)
    return jax.vmap(lambda k: generate_gp_fourier(k, sqrt_power, n_points))(keys)


def compute_sqrt_power_drw(
    n_points: int, d_log_age: float, psd_sigma: float, psd_tau_yr: float, log_age_ref: float = 8.0
) -> jnp.ndarray:
    r"""Pre-compute DRW amplitude operator for log-age grid.

    Converts the DRW PSD from physical frequency (rad/yr) to log-age frequency
    space (rad/dex) using the Jacobian correction for the change of variables
    from linear time :math:`t` to log-time :math:`u = \log_{10} t`.

    Parameters
    ----------
    n_points : int
        Grid size (number of age samples).
    d_log_age : float
        Grid spacing in dex [dimensionless].
    psd_sigma : float
        DRW PSD amplitude [dimensionless].
    psd_tau_yr : float
        DRW damping timescale [yr].
    log_age_ref : float, optional
        Reference log10(age/yr) for Jacobian correction. Default: 8.0 (100 Myr).

    Returns
    -------
    ndarray, shape (n_freq,)
        Amplitude operator :math:`\sqrt{P_u(q) / \Delta u}` at rfft frequencies
        in log-age space [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The Jacobian correction for the change of variables from cosmic time
    :math:`t` (in years) to log-age :math:`u = \log_{10}(t)` is:

    .. math::

        P_u(q) = P_t\left( \frac{q}{t_{\rm ref} \ln 10} \right) / (t_{\rm ref} \ln 10)

    where :math:`t_{\rm ref} = 10^{\log_{\rm age, ref}}` is a reference time,
    :math:`q` is the angular frequency in log-age space [rad/dex],
    and :math:`P_t` is the PSD in physical frequency space.

    The reference time scales out in the power spectrum but affects the
    mapping between physical and log-age frequencies. A typical choice is
    the midpoint of the age range (e.g., 100 Myr ~ 8 Gyr cosmic age).

    Examples
    --------
    >>> from tengri import compute_sqrt_power_drw, make_log_age_grid
    >>> grid = make_log_age_grid(n_grid=64)
    >>> d = float(grid[1] - grid[0])
    >>> sp = compute_sqrt_power_drw(64, d, psd_sigma=1.0, psd_tau_yr=1e8)
    >>> sp.shape
    (33,)
    """
    from tengri.components.stellar.sfh.psd_models import psd_drw, psd_to_sqrt_power

    t_ref = 10.0**log_age_ref
    ln10 = jnp.log(10.0)

    # FFT frequencies in log-age space (rad/dex)
    freqs = jnp.fft.rfftfreq(n_points, d=d_log_age)
    q = 2.0 * jnp.pi * freqs

    # Convert to physical frequency (rad/yr)
    omega_phys = q / (t_ref * ln10)

    # Evaluate PSD in physical domain and apply Jacobian
    p_phys = psd_drw(omega_phys, psd_sigma, psd_tau_yr)
    p_logage = p_phys / (t_ref * ln10)

    return psd_to_sqrt_power(p_logage, d_log_age)
