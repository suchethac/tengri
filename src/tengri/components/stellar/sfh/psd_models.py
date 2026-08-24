# SPDX-License-Identifier: BSD-3-Clause
"""Power spectral density models for stochastic star formation histories.

All functions are pure JAX and JIT-compatible. Each PSD function takes
frequency arrays and returns power spectral density P(omega).

The DRW (damped random walk / Lorentzian) is the primary model used
in the paper; others are provided for extensibility.

References
----------

- DRW: Munoz+2026 (arXiv:2601.07912)
- Extended Regulator: Tacchella+2020, Caplar & Tacchella 2019
- Flex-PSD: Burnham+2026 (arXiv:2601.20930)
- Matern: generalizes DRW (nu=0.5 recovers DRW)

"""

import jax.numpy as jnp

# ── Primary: Damped Random Walk ───────────────────────────────────


def psd_drw(omega: jnp.ndarray, psd_sigma: float, psd_tau_yr: float) -> jnp.ndarray:
    r"""Damped random walk (Lorentzian) power spectral density.

    Parameters
    ----------
    omega : array_like, shape (n_freq,)
        Angular frequency [rad/yr].
    psd_sigma : float
        PSD amplitude (dimensionless).
    psd_tau_yr : float
        Characteristic damping timescale [yr].

    Returns
    -------
    ndarray, shape (n_freq,)
        Power spectral density at each frequency [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The damped random walk (Lorentzian) power spectral density is:

    .. math::

        P(\\omega) = \\sigma_{\rm PS}^2 \\,\tau_{\rm PS} / (1 + (\tau_{\rm PS} \\omega)^2)

    where :math:`\\sigma_{\rm PS}` is the PSD amplitude [dimensionless],
    :math:`\tau_{\rm PS}` is the damping timescale [yr], and :math:`\\omega`
    is the angular frequency [rad/yr].

    This is the primary PSD model used in tengri for stochastic SFH modeling.

    References
    ----------
    .. [1] J. B. Munoz and K. G. Iyer, "Measuring the Power Spectral Density
       of Star Formation Histories," arXiv:2601.07912 (2026).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import psd_drw
    >>> omega = jnp.linspace(0.0, 2e-7, 128)  # angular frequencies [rad/yr]
    >>> psd = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=1e8)
    >>> psd.shape
    (128,)
    >>> float(psd[0])  # peak power at omega=0
    1e+08
    """
    return psd_sigma**2 * psd_tau_yr / (1.0 + (psd_tau_yr * omega) ** 2)


def drw_acf(delta_t: jnp.ndarray, psd_sigma: float, psd_tau_yr: float) -> jnp.ndarray:
    """Analytic autocorrelation function (autocovariance) for DRW.

    Parameters
    ----------
    delta_t : array_like, shape (n_lag,)
        Time lag [yr].
    psd_sigma : float
        PSD amplitude (dimensionless).
    psd_tau_yr : float
        Damping timescale [yr].

    Returns
    -------
    ndarray, shape (n_lag,)
        Autocovariance at each lag [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The autocovariance (autocorrelation function) of the DRW is:

    .. math::

        \\xi(\\Delta t) = \\frac{\\sigma_{\\rm PS}^2}{2}
            \\exp\\!\\left( -\\frac{|\\Delta t|}{\\tau_{\\rm PS}} \\right)

    where :math:`\\sigma_{\\rm PS}` is the PSD amplitude, :math:`\\tau_{\\rm PS}`
    is the damping timescale [yr], and :math:`\\Delta t` is the time lag [yr].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import drw_acf
    >>> lags = jnp.array([0.0, 1e7, 1e8, 1e9])  # time lags [yr]
    >>> acf = drw_acf(lags, psd_sigma=1.0, psd_tau_yr=1e8)
    >>> acf.shape
    (4,)
    >>> float(acf[0])  # variance at zero lag = sigma^2 / 2
    0.5
    """
    return 0.5 * psd_sigma**2 * jnp.exp(-jnp.abs(delta_t) / psd_tau_yr)


def drw_variance(psd_sigma: float) -> float:
    r"""Stationary variance of DRW.

    Parameters
    ----------
    psd_sigma : float
        PSD amplitude (dimensionless).

    Returns
    -------
    float
        Stationary variance [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, scalar arithmetic.

    The stationary variance of a DRW is:

    .. math::

        \sigma_x^2 = \frac{\sigma_{\rm PS}^2}{2}

    where :math:`\sigma_{\rm PS}` is the PSD amplitude.

    Examples
    --------
    >>> from tengri import drw_variance
    >>> float(drw_variance(psd_sigma=1.0))
    0.5
    >>> float(drw_variance(psd_sigma=2.0))
    2.0
    """
    return 0.5 * psd_sigma**2


def psd_to_sqrt_power(psd_values: jnp.ndarray, d_grid: float) -> jnp.ndarray:
    r"""Convert PSD to amplitude operator for GP generation.

    Computes the factor that multiplies the standardized latent vector in
    Fourier space to produce a GP realization:
    :math:`s = \mathrm{IFFT}(\sqrt{P/d} \cdot \hat{\xi})`.

    Parameters
    ----------
    psd_values : array_like, shape (n_freq,)
        PSD evaluated at rfft frequencies [dimensionless].
    d_grid : float
        Grid spacing in the original domain (needed for FFT normalization) [dimensionless].

    Returns
    -------
    ndarray, shape (n_freq,)
        Amplitude operator :math:`\sqrt{P(\omega) / d_{\rm grid}}` [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The amplitude operator is:

    .. math::

        A(\omega) = \sqrt{\frac{P(\omega)}{d_{\rm grid}}}

    where :math:`P(\omega)` is the power spectral density and :math:`d_{\rm grid}`
    is the grid spacing. This normalization ensures that the GP realization
    has the correct variance: :math:`\mathrm{Var}[s] = \int P(f) \, df`.

    A floor of :math:`10^{-30}` is applied to avoid division by zero.
    """
    return jnp.sqrt(jnp.maximum(psd_values, 1e-30) / d_grid)


# ── Alternative PSD models ────────────────────────────────────────


def psd_matern(omega: jnp.ndarray, variance: float, length_scale: float, nu: float) -> jnp.ndarray:
    """Matern power spectral density (1D).

    Parameters
    ----------
    omega : array_like, shape (n_freq,)
        Angular frequency [rad/yr].
    variance : float
        Signal variance [dimensionless].
    length_scale : float
        Length scale parameter [yr].
    nu : float
        Matern smoothness parameter (dimensionless). nu=0.5 recovers the DRW.

    Returns
    -------
    ndarray, shape (n_freq,)
        Power spectral density [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.scipy.special.gammaln`` for log-gamma.

    The Matern PSD is a generalization of the DRW (which corresponds to nu=0.5).
    Larger nu values produce smoother realizations with steeper spectral fall-off
    at high frequencies.

    References
    ----------
    .. [1] Rasmussen & Williams, "Gaussian Processes for Machine Learning,"
       MIT Press (2006). Section 4.2.
    """
    from jax.scipy.special import gammaln

    lam = 2.0 * nu / length_scale**2
    log_norm = (
        jnp.log(2.0 * jnp.sqrt(jnp.pi))
        + gammaln(nu + 0.5)
        - gammaln(nu)
        + nu * jnp.log(2.0 * nu)
        - 2.0 * nu * jnp.log(length_scale)
    )
    log_spectral = -(nu + 0.5) * jnp.log(lam + omega**2)
    return variance * jnp.exp(log_norm + log_spectral)


def psd_extended_regulator(
    f: jnp.ndarray, s_reg: float, tau_in: float, tau_eq: float, s_dyn: float, tau_dyn: float
) -> jnp.ndarray:
    """Extended regulator power spectral density (Tacchella+2020).

    Two-component PSD combining a feedback-regulated component and a dynamical component.

    Parameters
    ----------
    f : array_like, shape (n_freq,)
        Cyclic frequency [Hz or 1/yr] (must match timescale parameters).
    s_reg : float
        Regulator component amplitude [dimensionless].
    tau_in : float
        Inflow timescale [inverse units of f].
    tau_eq : float
        Equilibrium/feedback timescale [inverse units of f].
    s_dyn : float
        Dynamical component amplitude [dimensionless].
    tau_dyn : float
        Dynamical timescale [inverse units of f].

    Returns
    -------
    ndarray, shape (n_freq,)
        Power spectral density [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The extended regulator PSD combines feedback-regulated accretion and
    dynamical instability:

    .. math::

        P(f) = \\frac{s_{\\rm reg}^2}{(1 + (2\\pi f \\tau_{\\rm in})^2)(1 + (2\\pi f \\tau_{\\rm eq})^2)}
               + \\frac{s_{\\rm dyn}^2}{1 + (2\\pi f \\tau_{\\rm dyn})^2}

    where all timescales are in the same units as the inverse of frequency f.

    References
    ----------
    .. [1] S. Tacchella et al., "A Redshift-independent Efficiency Model: Star
       Formation and Stellar Masses in Dark Matter Halos at z >= 4," ApJ, 868,
       92 (2018). arXiv:1806.03299. https://doi.org/10.3847/1538-4357/aae8e0
    .. [2] N. Caplar and S. Tacchella, "Stochastic modeling of star-formation
       histories I: the scatter of the star-forming main sequence," MNRAS, 487,
       3845 (2019). arXiv:1901.07556. https://doi.org/10.1093/mnras/stz1449
    """
    two_pi_f = 2.0 * jnp.pi * f
    regulator = s_reg**2 / ((1.0 + (tau_in * two_pi_f) ** 2) * (1.0 + (tau_eq * two_pi_f) ** 2))
    dynamical = s_dyn**2 / (1.0 + (tau_dyn * two_pi_f) ** 2)
    return regulator + dynamical
