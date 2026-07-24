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

# Default log10(age/yr) bounds for the SFH grid: 1 Myr → ~13.8 Gyr.
# Exposed as module constants so callers that need the static step size
# (e.g. ``StellarSEDComponent.apply`` under JIT, where indexing a traced
# ``jnp.linspace`` and calling ``float()`` would raise) can recompute it
# without re-tracing the grid.
LOG_AGE_MIN: float = 6.0
LOG_AGE_MAX: float = 10.14


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


def make_log_age_grid(
    n_grid: int = 256, log_age_min: float = LOG_AGE_MIN, log_age_max: float = LOG_AGE_MAX
) -> jnp.ndarray:
    """Create uniform grid in log10(age/yr).

    Default range: 1 Myr to ~13.8 Gyr (approximately the age of the universe).

    Parameters
    ----------
    n_grid : int, optional
        Number of grid points (should be even for FFT efficiency). Default: 256.
    log_age_min : float, optional
        Minimum log10(age/yr). Default: 6.0 (1 Myr).
    log_age_max : float, optional
        Maximum log10(age/yr). Default: 10.14 (~13.8 Gyr).

    Returns
    -------
    ndarray, shape (n_grid,)
        Uniform grid in log10(age/yr) [dimensionless log values].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.linspace``.

    This grid is used as the internal representation for age in GP-based SFH models.
    The log-space parametrization provides better resolution at young ages and
    maps naturally to the logarithmic timescales of stellar evolution.

    Examples
    --------
    >>> from tengri import make_log_age_grid
    >>> grid = make_log_age_grid(n_grid=64)
    >>> grid.shape
    (64,)
    >>> float(grid[0]), float(grid[-1])
    (6.0, 10.14)
    """
    return jnp.linspace(log_age_min, log_age_max, n_grid)


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
    **JIT-compatible**: yes — uses ``jnp.fft.rfft`` and ``jnp.fft.irfft``.

    **Gradient-safe**: yes — differentiable w.r.t. sqrt_power.

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
    .. [1] Selig et al., "NIFTY - Numerical Information Field Theory in Python,"
       A&A, 554, A26 (2013). arXiv:1301.4499.
       https://doi.org/10.1051/0004-6361/201321236

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
    ``xi ~ N(0, I)`` — the same standardized latent the samplers explore.

    Unlike the Fourier/log-age construction (:func:`gp_from_xi` +
    :func:`compute_sqrt_power_drw`), the correlation length is a **fixed number
    of years at every age** — gas-cycling burstiness is a physical-time process,
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
    :math:`\tau`, the covariance is effectively diagonal — burstiness below the
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
    :func:`drw_linear_gp_from_xi` — same covariance, same prior — but through the
    exact first-order Markov (Ornstein–Uhlenbeck) forward recursion rather than a
    dense Cholesky factor:

    .. math::

        s_0 &= \sqrt{\mathrm{var}}\; \xi_0, \\
        s_i &= \rho_i\, s_{i-1} + \sqrt{\mathrm{var}\,(1 - \rho_i^2)}\; \xi_i,
        \qquad \rho_i = \exp(-\Delta t_i / \tau),

    with :math:`\mathrm{var} = (\sigma \ln 10)^2`, physical times
    :math:`t_i = 10^{u_i}` from the (ascending) log-age grid, and step gaps
    :math:`\Delta t_i = t_i - t_{i-1} \ge 0`. Because a DRW is exactly Markov, this
    recursion reproduces :math:`K_{ij} = \mathrm{var}\,\exp(-|t_i-t_j|/\tau)` to
    machine precision — it is a *bit-exact-same-prior* reparameterization of
    :func:`drw_linear_gp_from_xi`.

    The reason to prefer it for inference: the timescale :math:`\tau` enters the
    :math:`\xi \to \mathrm{SFH}` map only through the per-step scalars
    :math:`\rho_i(\tau)` — a **banded, local** dependence — whereas the dense
    Cholesky factor :math:`L(\sigma,\tau)` is a **rotation** that re-orients as
    :math:`\tau` moves. HMC carries one global mass matrix (a single fixed metric)
    and cannot track a target whose principal axes rotate with a sampled
    hyperparameter; the banded coupling is far better conditioned (#1301).

    Parameters
    ----------
    xi : array_like, shape (n,)
        Standardized latent vector, :math:`\xi \sim \mathcal{N}(0, I)`.
    psd_sigma_dex : float
        Modulation amplitude [dex] = std of :math:`\log_{10}(\mathrm{SFR})`.
    psd_tau_yr : float
        Physical DRW decorrelation timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age/yr)`` grid, ascending, the SFH is represented on.

    Returns
    -------
    gp_x : ndarray, shape (n,)
        Natural-log SFH modulation (applied as ``exp(gp_x - k0_half)``).
    k0_half : ndarray, scalar
        Log-normal bias correction :math:`(\sigma \ln 10)^2 / 2`.

    Notes
    -----
    **JIT/grad/vmap-safe**: the recursion is a single ``jax.lax.scan`` — :math:`O(n)`
    time, :math:`O(1)` memory, differentiable w.r.t. ``psd_sigma_dex``,
    ``psd_tau_yr`` and ``xi``. No dense matrix, so (unlike
    :func:`drw_linear_gp_from_xi`) there is no Cholesky and no
    positive-definiteness jitter. At young ages :math:`\Delta t_i \ll \tau`
    (:math:`\rho_i \to 1`) the field is strongly correlated; at old ages
    :math:`\Delta t_i \gg \tau` (:math:`\rho_i \to 0`) successive nodes become
    independent draws of variance ``var`` — burstiness below the local grid
    resolution is unrepresentable there, matching :func:`drw_linear_gp_from_xi`.

    The innovation scale carries a ``clip(1 - rho**2, 0, None)`` floor so float
    round-off cannot drive the argument of the square root slightly negative; the
    grid is strictly ascending so :math:`1 - \rho_i^2 > 0` analytically.

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

    # Per-step correlation and fresh-innovation scale on the (ascending) grid.
    dt = jnp.diff(t)
    rho = jnp.exp(-dt / jnp.asarray(psd_tau_yr))
    innov = sigma_s * jnp.sqrt(jnp.clip(1.0 - rho**2, 0.0, None))

    s0 = sigma_s * xi[0]

    def _step(s_prev, step_inputs):
        rho_i, innov_i, xi_i = step_inputs
        s_i = rho_i * s_prev + innov_i * xi_i
        return s_i, s_i

    _, s_rest = jax.lax.scan(_step, s0, (rho, innov, xi[1:]))
    gp_x = jnp.concatenate([jnp.reshape(s0, (1,)), s_rest])
    return gp_x, 0.5 * var


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
    **JIT-compatible**: yes — uses ``jax.random.normal`` and :func:`gp_from_xi`.

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
    **JIT-compatible**: yes — uses ``jax.vmap`` over :func:`generate_gp_fourier`.

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
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

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
