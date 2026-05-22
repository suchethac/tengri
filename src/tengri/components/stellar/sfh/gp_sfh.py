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


def make_log_age_grid(
    n_grid: int = 256, log_age_min: float = 6.0, log_age_max: float = 10.14
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
    """Deterministic GP realization from standardized latent vector.

    Maps a standardized Gaussian random vector to a correlated GP field via
    Fourier-space multiplication with a spectral amplitude operator. This
    is the core function used during inference and mock generation.

    Parameters
    ----------
    xi : array_like, shape (n_points,)
        Standardized latent vector :math:`\\xi \\sim \\mathcal{N}(0, I)` under the prior.
    sqrt_power : array_like, shape (n_freq,)
        Amplitude operator :math:`\\sqrt{P(\\omega) / d_{\rm grid}}` at rfft frequencies
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

        s = \\mathrm{IFFT}(\\sqrt{P} \\cdot \\hat{\\xi})

    The rfft (real FFT) preserves Hermitian symmetry for real-valued output and
    ensures correct variance normalization: :math:`E[|\\mathrm{rfft}(\\xi)_k|^2] = N`,
    so with :math:`\\sqrt{P/\\Delta x}` we recover :math:`\\mathrm{Var}[s] = \\int P(f) df`.

    This is the primary function called during MCMC inference and mock galaxy generation.
    The sampler proposes values of :math:`\\xi` and this function maps them to
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


def generate_gp_fourier(key: jax.Array, sqrt_power: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """Stochastic GP realization for mock galaxy generation.

    Draws a random standardized vector and maps it to a correlated GP field.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random key for reproducibility.
    sqrt_power : array_like, shape (n_freq,)
        Amplitude operator :math:`\\sqrt{P(\\omega) / d_{\rm grid}}` at rfft frequencies
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
    """Batch of independent GP realizations via vectorization.

    Generates multiple independent SFH realizations in parallel using vmap.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random key (will be split into n_realizations independent keys).
    sqrt_power : array_like, shape (n_freq,)
        Amplitude operator :math:`\\sqrt{P(\\omega) / d_{\rm grid}}` at rfft frequencies.
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
    """Pre-compute DRW amplitude operator for log-age grid.

    Converts the DRW PSD from physical frequency (rad/yr) to log-age frequency
    space (rad/dex) using the Jacobian correction for the change of variables
    from linear time :math:`t` to log-time :math:`u = \\log_{10} t`.

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
        Amplitude operator :math:`\\sqrt{P_u(q) / \\Delta u}` at rfft frequencies
        in log-age space [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The Jacobian correction for the change of variables from cosmic time
    :math:`t` (in years) to log-age :math:`u = \\log_{10}(t)` is:

    .. math::

        P_u(q) = P_t\\left( \\frac{q}{t_{\\rm ref} \\ln 10} \\right) / (t_{\\rm ref} \\ln 10)

    where :math:`t_{\\rm ref} = 10^{\\log_{\rm age, ref}}` is a reference time,
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
