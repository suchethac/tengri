# SPDX-License-Identifier: BSD-3-Clause
r"""Faithful analytic ADAF spectrum (Mahadevan 1997).

Differentiable JAX implementation of the analytic scaling laws for an
advection-dominated accretion flow (ADAF / RIAF), following Mahadevan (1997
[1]_) equation-for-equation. The model gives the radio-to-X-ray spectrum of a
low-luminosity AGN from three cooling processes: cyclo-synchrotron,
bremsstrahlung, and inverse Compton: as closed-form functions of the black
hole mass, accretion rate, viscosity :math:`\alpha`, and plasma parameters
:math:`\beta` (gas-to-total pressure) and :math:`\delta` (electron viscous
heating fraction).

This module is the differentiable, native-JAX sibling of the template-based
Nemmen (2014) model (#898 Phase 2) and the AGNNES emulator (#952).

Conventions (Mahadevan 1997)
----------------------------

- Mass ``m = M / M_sun``; accretion rate ``mdot = Mdot / Mdot_Edd`` with
  ``Mdot_Edd = L_Edd / (eta_eff c^2)`` and ``eta_eff = 0.1`` (Eq. 4).
- ``beta = P_gas / P_total`` (magnetic fraction is ``1 - beta``; Eq. 1).
- ``delta`` = fraction of viscous energy heating electrons directly.
- Fiducial constants: ``c1 = 0.5``, ``c3 = 0.3``, ``r_min = 3``, ``r_max = 1000``.
- The ADAF solution requires ``mdot < mdot_crit ~ 0.28 alpha^2`` (Eq. 52).

References
----------
.. [1] R. Mahadevan, "Scaling Laws for Advection-dominated Flows: Applications
   to Low-Luminosity Galactic Nuclei," ApJ, 477, 585 (1997).
   arXiv:astro-ph/9609107. https://doi.org/10.1086/303727
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import i0 as _i0, i1 as _i1

from tengri.components.agn._params import DEFAULT_AGN_LOG_MBH, DEFAULT_AGN_LUM_RATIO
from tengri.components.agn._phys import (
    C_LIGHT as _C_LIGHT,
    H_PLANCK as _H_PLANCK,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.physics_constants import (
    K_BOLTZ as _K_BOLTZ,
    L_SUN as _LSUN_ERG,
    M_ELECTRON as _M_ELECTRON,
)
from tengri.utils.scale import representable_floor as _representable_floor

# Fiducial self-similar constants (Mahadevan 1997, Narayan & Yi 1995b).
_C1: float = 0.5
_C3: float = 0.3
_R_MIN: float = 3.0
_R_MAX: float = 1000.0
_ETA_EFF: float = 0.1

# theta_e = k T_e / (m_e c^2)
_THETA_PER_TE: float = _K_BOLTZ / (_M_ELECTRON * _C_LIGHT**2)


# ── Modified Bessel K_2 (differentiable) ──────────────────────────────────
#
# JAX ships only Bessel I (i0/i1), so K_0, K_1 are built from the
# Abramowitz & Stegun (1964) rational approximations 9.8.5-9.8.8 (accurate to
# ~1e-7), and K_2 from the recurrence K_2(x) = K_0(x) + (2/x) K_1(x). We return
# the exponentially-scaled K_n(x) e^x to avoid overflow, mirroring SciPy's kve.


def _bessel_k0e_k1e(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Exponentially-scaled modified Bessel functions ``K_0(x) e^x`` and ``K_1(x) e^x``.

    Parameters
    ----------
    x : array_like
        Argument, ``x > 0``.

    Returns
    -------
    tuple of ndarray
        ``(K_0(x) e^x, K_1(x) e^x)``.

    Notes
    -----
    **JIT/grad-safe**: yes. Each branch is evaluated on a clipped argument so the
    unused branch of the ``jnp.where`` never overflows or produces a non-finite
    value that could poison the gradient.
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    # Small-argument branch (0 < x <= 2): A&S 9.8.5 / 9.8.7. Clip so the
    # discarded branch stays finite for large x (i0/i1 grow ~ e^x).
    xs = jnp.clip(x, 1e-300, 2.0)
    t = (xs / 2.0) ** 2
    ln_half = jnp.log(xs / 2.0)
    k0_small = -ln_half * _i0(xs) + (
        -0.57721566
        + 0.42278420 * t
        + 0.23069756 * t**2
        + 0.03488590 * t**3
        + 0.00262698 * t**4
        + 0.00010750 * t**5
        + 0.00000740 * t**6
    )
    xk1_small = xs * ln_half * _i1(xs) + (
        1.0
        + 0.15443144 * t
        - 0.67278579 * t**2
        - 0.18156897 * t**3
        - 0.01919402 * t**4
        - 0.00110404 * t**5
        - 0.00004686 * t**6
    )
    k1_small = xk1_small / xs
    # Scale to K_n e^x.
    k0e_small = k0_small * jnp.exp(xs)
    k1e_small = k1_small * jnp.exp(xs)

    # Large-argument branch (x >= 2): A&S 9.8.6 / 9.8.8 give sqrt(x) e^x K_n.
    xl = jnp.clip(x, 2.0, jnp.inf)
    u = 2.0 / xl
    sqrt_xl = jnp.sqrt(xl)
    k0e_large = (
        1.25331414
        - 0.07832358 * u
        + 0.02189568 * u**2
        - 0.01062446 * u**3
        + 0.00587872 * u**4
        - 0.00251540 * u**5
        + 0.00053208 * u**6
    ) / sqrt_xl
    k1e_large = (
        1.25331414
        + 0.23498619 * u
        - 0.03655620 * u**2
        + 0.01504268 * u**3
        - 0.00780353 * u**4
        + 0.00325614 * u**5
        - 0.00068245 * u**6
    ) / sqrt_xl

    small = x <= 2.0
    return jnp.where(small, k0e_small, k0e_large), jnp.where(small, k1e_small, k1e_large)


def _bessel_k2e(x: jnp.ndarray) -> jnp.ndarray:
    r"""Exponentially-scaled modified Bessel function ``K_2(x) e^x``.

    Uses the recurrence :math:`K_2(x) = K_0(x) + (2/x) K_1(x)`.

    Parameters
    ----------
    x : array_like
        Argument, ``x > 0``.

    Returns
    -------
    ndarray
        ``K_2(x) e^x``.
    """
    k0e, k1e = _bessel_k0e_k1e(x)
    return k0e + (2.0 / x) * k1e


def _adaf_g_theta(t_e: jnp.ndarray) -> jnp.ndarray:
    r"""Relativistic Maxwellian factor ``g(theta_e)`` (Mahadevan 1997, Eq. 11).

    .. math::

        g(\theta_e) \equiv \frac{1}{K_2(1/\theta_e)}
            \left(2 + 2\theta_e + \frac{1}{\theta_e}\right) e^{-1/\theta_e}

    with :math:`\theta_e = k T_e / (m_e c^2)`. Enters the Coulomb ion-electron
    heating rate (Eq. 10) and hence the total ADAF luminosity (Eq. 49).

    Parameters
    ----------
    t_e : array_like
        Electron temperature [K].

    Returns
    -------
    ndarray
        ``g(theta_e)`` [dimensionless]. For the ADAF range (T_e ~ 1e9-1e10 K)
        this is ~1-7; ~7 at the high-mdot limit (T_e ~ 1.5e9 K).

    Notes
    -----
    **JIT/grad-safe**: yes.
    """
    theta = _THETA_PER_TE * jnp.asarray(t_e, dtype=jnp.float64)
    x = 1.0 / theta  # = 1/theta_e
    # g = (2 + 2 theta + 1/theta) e^{-x} / K_2(x) = numerator / (K_2(x) e^x).
    return (2.0 + 2.0 * theta + 1.0 / theta) / _bessel_k2e(x)


# ── Self-similar flow structure (Mahadevan 1997 / Narayan & Yi 1995b, Eq. 5) ──


def _adaf_s1(alpha: float, beta: float) -> jnp.ndarray:
    r"""Magnetic-field self-similar constant ``s_1`` (Mahadevan 1997 Eq. 5, p.4).

    :math:`s_1 = 1.42\times10^9\,\alpha^{-1/2}(1-\beta)^{1/2}c_1^{-1/2}c_3^{1/2}`.

    Notes
    -----
    This is the *Mahadevan 1997* (spherical-accretion) coefficient. The paper
    states Eq. 5 "differ[s] from Narayan & Yi (1995b) since we have assumed
    spherical accretion," and Eq. 1's footnote carries a factor 1/3 for a 3-D
    tangled field. The alternative NY95b B-normalization (~6.55e8 with a
    :math:`c_3^{1/4}`) is a different convention and must NOT be substituted here.
    """
    return 1.42e9 * alpha**-0.5 * (1.0 - beta) ** 0.5 * _C1**-0.5 * _C3**0.5


def _adaf_ne_b_rmin(
    m: float, mdot: float, alpha: float, beta: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Electron number density and magnetic field at ``r_min`` (Mahadevan 1997 Eq. 5, p.4).

    .. math::

        n_e = b_1\,m^{-1}\dot m\,r^{-3/2}\ \mathrm{cm^{-3}}, \qquad
        B   = s_1\,m^{-1/2}\dot m^{1/2}\,r^{-5/4}\ \mathrm{G},

    with :math:`b_1 = 3.16\times10^{19}\alpha^{-1}c_1^{-1}`, evaluated at
    :math:`r = r_{\min} = 3`. ``s_1`` (and the spherical-accretion convention that
    distinguishes it from Narayan & Yi 1995b) is in :func:`_adaf_s1`.

    Parameters
    ----------
    m : float
        Black hole mass ``M / M_sun``.
    mdot : float
        Accretion rate in Eddington units.
    alpha, beta : float
        Viscosity and gas-to-total pressure ratio.

    Returns
    -------
    tuple of ndarray
        ``(n_e [cm^-3], B [G])`` at ``r_min``.
    """
    b1 = 3.16e19 * alpha**-1.0 * _C1**-1.0
    n_e = b1 * m**-1.0 * mdot * _R_MIN**-1.5
    b_field = _adaf_s1(alpha, beta) * m**-0.5 * mdot**0.5 * _R_MIN**-1.25
    return n_e, b_field


def _adaf_tau_es(mdot: float, alpha: float) -> jnp.ndarray:
    r"""Electron-scattering optical depth ``tau_es`` at ``r_min`` (Mahadevan 1997 Eq. 31, p.12).

    .. math::

        \tau_{es} = 6.2\,\alpha^{-1}c_1^{-1}\dot m\,r_{\min}^{-1/2}
                  = 23.87\,\dot m\,(\alpha/0.3)^{-1}

    (with :math:`c_1=0.5`, :math:`r_{\min}=3`). This is **half** the total
    electron-scattering depth of Narayan & Yi (1995b): the paper takes the mean
    photon to traverse half the total depth ("we therefore take the optical depth
    to electron scattering to be half of that as given in Narayan & Yi 1995b").

    Parameters
    ----------
    mdot : float
        Accretion rate in Eddington units.
    alpha : float
        Viscosity parameter.

    Returns
    -------
    ndarray
        ``tau_es`` [dimensionless].
    """
    return 6.2 * alpha**-1.0 * _C1**-1.0 * mdot * _R_MIN**-0.5


def _adaf_amplification(t_e: jnp.ndarray) -> jnp.ndarray:
    r"""Mean Compton amplification factor per scattering ``A`` (Mahadevan Eq. 32).

    :math:`A = 1 + 4\theta_e + 16\theta_e^2`, with :math:`\theta_e = kT_e/m_ec^2`.
    """
    theta = _THETA_PER_TE * jnp.asarray(t_e, dtype=jnp.float64)
    return 1.0 + 4.0 * theta + 16.0 * theta**2


def _adaf_alpha_c(tau_es: jnp.ndarray, t_e: jnp.ndarray) -> jnp.ndarray:
    r"""Comptonization spectral slope ``alpha_c`` (Mahadevan Eq. 34).

    .. math::

        \alpha_c \equiv \frac{-\ln \tau_{es}}{\ln A}

    where :math:`A` is the amplification factor (Eq. 32). Governs the sub-mm to
    X-ray Compton spectrum :math:`L_\nu \propto \nu^{-\alpha_c}`; ``alpha_c < 1``
    means Compton cooling dominates.

    Parameters
    ----------
    tau_es : array_like
        Electron-scattering optical depth (Eq. 31).
    t_e : array_like
        Electron temperature [K].

    Returns
    -------
    ndarray
        ``alpha_c`` [dimensionless].
    """
    a_factor = _adaf_amplification(t_e)
    return -jnp.log(tau_es) / jnp.log(a_factor)


# ── Synchrotron self-absorption parameter x_M (Mahadevan Eq. 20) ──────────

# Schwarzschild radius R_schw = 2.95e5 * m  cm  (Eq. 3).
_R_SCHW_PER_M: float = 2.95e5


def _adaf_x_m(t_e: jnp.ndarray, m: float, mdot: float, alpha: float, beta: float) -> jnp.ndarray:
    r"""Solve for the synchrotron self-absorption parameter ``x_M`` (Mahadevan Eq. 20).

    The cyclo-synchrotron photons self-absorb up to a critical frequency; ``x_M``
    (the scaled critical frequency at ``r_min``) is the root of the transcendental

    .. math::

        e^{1.8899\,x_M^{1/3}} = 2.49\times10^{-10}\,\frac{4\pi n_e R}{B}\,
            \frac{1}{\theta_e^3 K_2(1/\theta_e)}
            \left(x_M^{-7/6} + 0.40\,x_M^{-17/12} + 0.5316\,x_M^{-5/3}\right).

    Solved by Newton's method in :math:`y = x_M^{1/3}` (the function is monotonic,
    so the root is unique and Newton is stable from a log-based initial guess).

    Parameters
    ----------
    t_e : array_like
        Electron temperature [K] (sets :math:`\theta_e`).
    m, mdot, alpha, beta : float
        Mass, accretion rate, viscosity, gas-to-total pressure ratio.

    Returns
    -------
    ndarray
        ``x_M`` [dimensionless]; ~1e3 for typical ADAF parameters.

    Notes
    -----
    **JIT/grad-safe**: yes, fixed 8-step unrolled Newton iteration.
    """
    n_e, b_field = _adaf_ne_b_rmin(m, mdot, alpha, beta)
    r_cm = _R_MIN * _R_SCHW_PER_M * m
    theta = _THETA_PER_TE * jnp.asarray(t_e, dtype=jnp.float64)
    x_arg = 1.0 / theta
    k2 = _bessel_k2e(x_arg) * jnp.exp(-x_arg)  # unscaled K_2(1/theta_e)
    ln_c = jnp.log(2.49e-10 * (4.0 * jnp.pi * n_e * r_cm / b_field) / (theta**3 * k2))

    def _log_h_and_dlogh(y):
        xm = y**3
        h = xm ** (-7.0 / 6.0) + 0.40 * xm ** (-17.0 / 12.0) + 0.5316 * xm ** (-5.0 / 3.0)
        dh_dxm = (
            (-7.0 / 6.0) * xm ** (-7.0 / 6.0 - 1.0)
            + 0.40 * (-17.0 / 12.0) * xm ** (-17.0 / 12.0 - 1.0)
            + 0.5316 * (-5.0 / 3.0) * xm ** (-5.0 / 3.0 - 1.0)
        )
        return jnp.log(h), (dh_dxm * 3.0 * y**2) / h

    y = jnp.clip(ln_c / 1.8899, 1.0, 100.0)
    for _ in range(8):
        log_h, dlogh = _log_h_and_dlogh(y)
        f = 1.8899 * y - ln_c - log_h
        df = 1.8899 - dlogh
        y = jnp.clip(y - f / df, 1.0, 100.0)
    return y**3


# ── Equilibrium electron temperature (Mahadevan Eqs. 40 & 43) ─────────────


def _adaf_electron_temperature(
    m: float, mdot: float, alpha: float, beta: float, delta: float
) -> jnp.ndarray:
    r"""Self-consistent equilibrium electron temperature ``T_e`` (Mahadevan Eqs. 40/43).

    The electrons cool via synchrotron, bremsstrahlung, and Compton; equating the
    total cooling to the heating :math:`Q^{e+} = Q^-` fixes ``T_e``. The paper gives
    two analytic branches selected by the Compton slope :math:`\alpha_c`:

    - :math:`\alpha_c > 1` (weak Compton, low ``mdot``), Eq. 40:

      .. math::

          T_e = 1.1\times10^9 (2000\delta)^{1/7} (x_M/300)^{-3/7}
                (\alpha/0.3)^{3/14} ((1-\beta)/0.5)^{-1/14}
                m^{1/14} \dot m^{-1/14}\ \mathrm{K}

    - :math:`\alpha_c < 1` (strong Compton, high ``mdot``), Eq. 43:

      .. math::

          T_e = 0.744\times10^9 \left[(4\,\tau_{es}^{-1/\alpha_c} - 3)^{1/2} - 1\right]\ \mathrm{K}

    Since ``x_M`` and :math:`\alpha_c` themselves depend on ``T_e``, the equations
    are solved together by a short fixed-point iteration (the paper notes accurate
    results without iteration; we iterate for robustness). ``c1=0.5``, ``c3=0.3``,
    ``r_min=3`` reduce those factors to unity. The A_c fitting factor of Eq. 40
    (~0.95-1.4) is taken as unity.

    Parameters
    ----------
    m, mdot, alpha, beta, delta : float
        Mass, accretion rate, viscosity, gas-to-total pressure ratio, and electron
        viscous-heating fraction.

    Returns
    -------
    ndarray
        Equilibrium electron temperature [K], clipped to ``[1e8, 5e11]`` K. For the
        ADAF range this is ~1e9-1e10 K, ~2e9 at the high-``mdot`` limit.

    Notes
    -----
    **JIT/grad-safe**: yes, fixed 8-step unrolled fixed point; the Eq. 40 /
    Eq. 43 regime choice is a branchless :func:`jnp.where` on the traced
    ``alpha_c``, so both branches are always evaluated (no data-dependent
    control flow).

    **Gradient kink**: ``T_e`` (and hence the whole spectrum) has a first-order
    kink at ``alpha_c = 1`` where the branch switches. Samplers crossing this
    boundary see a discontinuous derivative, the same class as the
    ``agn_torus_frac`` / ``cos(theta_torus)`` discontinuity noted in the project
    gotchas. It is continuous in value (both branches meet at ``alpha_c = 1``),
    only the slope jumps.
    """
    tau_es = _adaf_tau_es(mdot, alpha)
    t_e = jnp.asarray(2.0e9, dtype=jnp.float64)
    for _ in range(8):
        x_m = _adaf_x_m(t_e, m, mdot, alpha, beta)
        alpha_c = _adaf_alpha_c(tau_es, t_e)
        # Eq. 40 (alpha_c > 1).
        t_e_40 = (
            1.1e9
            * (2000.0 * delta) ** (1.0 / 7.0)
            * (x_m / 300.0) ** (-3.0 / 7.0)
            * (alpha / 0.3) ** (3.0 / 14.0)
            * ((1.0 - beta) / 0.5) ** (-1.0 / 14.0)
            * m ** (1.0 / 14.0)
            * mdot ** (-1.0 / 14.0)
        )
        # Eq. 43 (alpha_c < 1).
        sqrt_arg = jnp.maximum(4.0 * tau_es ** (-1.0 / jnp.maximum(alpha_c, 1e-3)) - 3.0, 0.0)
        t_e_43 = jnp.maximum(0.744e9 * (jnp.sqrt(sqrt_arg) - 1.0), 1e8)
        t_e = jnp.clip(jnp.where(alpha_c > 1.0, t_e_40, t_e_43), 1e8, 5e11)
    return t_e


# ── Spectral component amplitudes (Mahadevan Eqs. 21-23, 28, 30) ──────────


def _adaf_F_theta(t_e: jnp.ndarray) -> jnp.ndarray:
    r"""Relativistic thermal bremsstrahlung factor ``F(theta_e)`` (Mahadevan Eq. 28).

    Piecewise in :math:`\theta_e = kT_e/m_ec^2` (Stepney & Guilbert 1983):

    .. math::

        F(\theta_e) = \begin{cases}
          4(2\theta_e/\pi^3)^{1/2}(1+1.781\theta_e^{1.34})
            + 1.73\theta_e^{3/2}(1+1.1\theta_e+\theta_e^2-1.25\theta_e^{5/2}),
            & \theta_e < 1,\\[4pt]
          (9\theta_e/2\pi)[\ln(1.123\theta_e+0.48)+1.5]
            + 2.30\theta_e[\ln(1.123\theta_e)+1.28], & \theta_e > 1.
        \end{cases}

    Parameters
    ----------
    t_e : array_like
        Electron temperature [K].

    Returns
    -------
    ndarray
        ``F(theta_e)`` [dimensionless].
    """
    theta = _THETA_PER_TE * jnp.asarray(t_e, dtype=jnp.float64)
    # Clip each branch's argument so the discarded jnp.where branch stays finite.
    th_lo = jnp.clip(theta, 1e-8, 1.0)
    f_lo = 4.0 * (2.0 * th_lo / jnp.pi**3) ** 0.5 * (
        1.0 + 1.781 * th_lo**1.34
    ) + 1.73 * th_lo**1.5 * (1.0 + 1.1 * th_lo + th_lo**2 - 1.25 * th_lo**2.5)
    th_hi = jnp.clip(theta, 1.0, 1e6)
    f_hi = (9.0 * th_hi / (2.0 * jnp.pi)) * (
        jnp.log(1.123 * th_hi + 0.48) + 1.5
    ) + 2.30 * th_hi * (jnp.log(1.123 * th_hi) + 1.28)
    return jnp.where(theta < 1.0, f_lo, f_hi)


def _adaf_nu_peak(
    t_e: jnp.ndarray, x_m: jnp.ndarray, m: float, mdot: float, alpha: float, beta: float
) -> jnp.ndarray:
    r"""Synchrotron peak (self-absorption) frequency at ``r_min`` (Mahadevan Eqs. 21-22).

    .. math::

        \nu_p = s_1\,s_2\,m^{-1/2}\dot m^{1/2}\,T_e^2\,r_{\min}^{-5/4}\ \mathrm{Hz},
        \qquad s_2 = 1.19\times10^{-13}\,x_M.

    The ``T_e^2`` dependence (dropped by the previous implementation) is essential.

    Returns
    -------
    ndarray
        Peak frequency [Hz]; ~few x 1e11 Hz (sub-mm) for typical ADAF parameters.
    """
    s2 = 1.19e-13 * x_m
    return _adaf_s1(alpha, beta) * s2 * m**-0.5 * mdot**0.5 * t_e**2 * _R_MIN**-1.25


def _adaf_lnu_peak(t_e: jnp.ndarray, nu_p: jnp.ndarray, m: float) -> jnp.ndarray:
    r"""Synchrotron luminosity at the peak frequency (Mahadevan Eq. 23, at ``r_min``).

    .. math::

        L_{\nu_p} = s_3\,T_e\,\nu_p^2\,m^2\,r_{\min}^2\ \mathrm{erg\,s^{-1}\,Hz^{-1}},
        \qquad s_3 = 1.05\times10^{-24}.
    """
    return 1.05e-24 * t_e * nu_p**2 * m**2 * _R_MIN**2


def _adaf_lbrems0(t_e: jnp.ndarray, m: float, mdot: float, alpha: float) -> jnp.ndarray:
    r"""Bremsstrahlung luminosity prefactor (Mahadevan Eq. 30, at ``hv << kT_e``).

    :math:`L_\nu^{\rm brems}(\nu) = L_{\rm brems,0}\,e^{-h\nu/kT_e}` with

    .. math::

        L_{\rm brems,0} = 2.29\times10^{24}\,\alpha^{-2}c_1^{-2}
            \ln(r_{\max}/r_{\min})\,F(\theta_e)\,T_e^{-1}\,m\,\dot m^2
            \ \mathrm{erg\,s^{-1}\,Hz^{-1}}.
    """
    return (
        2.29e24
        * alpha**-2.0
        * _C1**-2.0
        * jnp.log(_R_MAX / _R_MIN)
        * _adaf_F_theta(t_e)
        * t_e**-1.0
        * m
        * mdot**2
    )


# ── L_bol -> mdot inversion (Mahadevan Eq. 49) ───────────────────────────


def _adaf_mdot_from_lbol(
    l_bol_erg: jnp.ndarray,
    m: float,
    alpha: float,
    beta: float,
    delta: float,
    float32: bool = False,
) -> jnp.ndarray:
    r"""Derive the accretion rate ``mdot`` from the canonical bolometric luminosity.

    Inverts the total ADAF luminosity (Mahadevan Eq. 49, ``mdot > 1e-3 alpha^2``):

    .. math::

        L_{\rm ADAF} = 1.2\times10^{38}\,g(\theta_e)\,c_1^{-2}c_3\,\beta\,
            r_{\min}^{-1}\alpha^{-2}\,m\,\dot m^2
            = 4.8\times10^{37}\,g(\theta_e)\,\beta\,\alpha^{-2}\,m\,\dot m^2

    (with ``c1=0.5``, ``c3=0.3``, ``r_min=3``). Since ``g(theta_e)`` depends on
    ``T_e(mdot)``, we iterate a short fixed point (``g`` varies slowly). Keeping
    ``agn_log_lbol`` as the canonical luminosity knob (deriving ``mdot`` rather
    than taking it as an independent input) is what makes the ADAF consistent
    with the disc convention of #846: ``agn_log_ledd`` is retired here.

    The result is clipped to ``(0, mdot_crit]`` with ``mdot_crit = 0.28 alpha^2``
    (Eq. 52): the ADAF solution does not exist above the critical rate.

    Parameters
    ----------
    l_bol_erg : array_like
        ADAF bolometric (radiated) luminosity [erg/s].
    m, alpha, beta, delta : float
        Mass, viscosity, gas-to-total pressure, electron-heating fraction.

    Returns
    -------
    ndarray
        ``mdot`` in Eddington units.

    Notes
    -----
    **JIT/grad-safe**: yes, fixed 3-step fixed point.
    """
    mdot_crit = 0.28 * alpha**2
    # Float32 (#1206): ``coeff`` ~3e46 and ``l_bol_erg`` ~1e44 erg/s both overflow,
    # but only their ratio (~mdot**2 ~1e-3) is needed. Work in L_sun: ``l_bol_erg``
    # arrives as ``10**log_lbol`` (L_sun) and ``coeff`` divides by L_sun via a
    # pre-divided constant, so the sqrt argument stays representable.
    if float32:
        coeff = (4.8e37 / _LSUN_ERG) * beta * alpha**-2.0 * m
    else:
        coeff = 4.8e37 * beta * alpha**-2.0 * m
    g = 7.0  # high-mdot equilibrium value as initial guess
    mdot = jnp.sqrt(l_bol_erg / (coeff * g))
    for _ in range(3):
        mdot = jnp.clip(mdot, 1e-8, mdot_crit)
        g = _adaf_g_theta(_adaf_electron_temperature(m, mdot, alpha, beta, delta))
        mdot = jnp.sqrt(l_bol_erg / (coeff * g))
    return jnp.clip(mdot, 1e-8, mdot_crit)


# ── Public spectrum ──────────────────────────────────────────────────────


def adaf_spectrum(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_adaf_alpha: float = 0.3,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.1,
    agn_log_lbol_shape: float | None = None,
    **_kwargs,
) -> jnp.ndarray:
    r"""Faithful analytic ADAF spectrum (Mahadevan 1997).

    Radio-to-X-ray SED of a radiatively inefficient (advection-dominated)
    accretion flow: rising cyclo-synchrotron (:math:`L_\nu \propto \nu^{2/5}`)
    up to the self-absorption peak :math:`\nu_p`, a Comptonized power law
    (:math:`\nu^{-\alpha_c}`) from :math:`\nu_p` to :math:`3kT_e/h`, and a
    bremsstrahlung tail (flat with an exponential cutoff at :math:`kT_e/h`).

    ``agn_log_lbol`` is the canonical ADAF radiated luminosity; the accretion
    rate ``mdot`` is derived from it (Eq. 49, :func:`_adaf_mdot_from_lbol`) and
    drives the whole spectrum. The three component *amplitudes* (Eqs. 23 & 30)
    set their relative weights (synchrotron/Compton join continuously at
    :math:`\nu_p`), and the total is renormalized to ``L_bol``, which both fixes
    the canonical scale and absorbs the ``T_e^7``-sensitivity of the absolute
    synchrotron power (see the module notes).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    agn_log_lbol : float
        log10 of the ADAF bolometric luminosity [log10(L_sun)].
    agn_lum_ratio : float, optional
        Fraction of the bolometric assigned to the ADAF. Default 1.0.
    agn_log_mbh : float, optional
        Black hole mass [log10(M_sun)]. Default 8.0.
    agn_adaf_alpha : float, optional
        Viscosity parameter :math:`\alpha`. Default 0.3.
    agn_adaf_beta : float, optional
        Gas-to-total pressure ratio :math:`\beta` (magnetic fraction is
        :math:`1-\beta`). Default 0.5.
    agn_adaf_delta : float, optional
        Fraction of viscous energy heating electrons directly :math:`\delta`,
        the single most consequential ADAF parameter (it sets the flow luminosity
        at fixed :math:`\dot m`). Default ``0.1`` **departs from Mahadevan 1997's
        own fiducial** :math:`\delta \sim m_e/m_i \sim 1/2000`; ``0.1`` follows the
        modern post-GRMHD preference (:math:`\delta \sim 0.1`–``0.5``; Yuan &
        Narayan 2014) that better matches observed LLAGN.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz], normalized so that
        :math:`\int L_\nu\,d\nu = \mathrm{agn\_frac}\times L_{\rm bol}`.

    Notes
    -----
    **JIT-compatible**: yes, all-``jnp`` with fixed-count unrolled solves.
    Retains the ``alpha_c=1`` gradient kink of the underlying ``T_e`` solve.
    Valid in the ADAF regime ``mdot < mdot_crit ~ 0.28 alpha^2``; the derived
    ``mdot`` is clipped there.

    References
    ----------
    .. [1] R. Mahadevan, ApJ, 477, 585 (1997). arXiv:astro-ph/9609107.
    """
    nu = _wavelength_to_nu(wavelength)
    _f32 = wavelength.dtype == jnp.float32
    # Shape luminosity (mdot -> whole spectrum) vs normalization magnitude. They
    # coincide by default (float64). On float32 the AGN component passes the true
    # L_bol for the SHAPE while normalizing MAGNITUDE to a low reference, so the
    # runner's ~1e40 L_lambda arithmetic stays in range (#1206).
    _lbol_shape = agn_log_lbol if agn_log_lbol_shape is None else agn_log_lbol_shape
    m = 10.0**agn_log_mbh
    alpha, beta, delta = agn_adaf_alpha, agn_adaf_beta, agn_adaf_delta

    # mdot from the SHAPE luminosity (float32: pass it in L_sun so the ~1e44 erg/s
    # l_bol_erg never forms).
    if _f32:
        mdot = _adaf_mdot_from_lbol(10.0**_lbol_shape, m, alpha, beta, delta, float32=True)
    else:
        mdot = _adaf_mdot_from_lbol(10.0**_lbol_shape * _LSUN_ERG, m, alpha, beta, delta)
    t_e = _adaf_electron_temperature(m, mdot, alpha, beta, delta)
    x_m = _adaf_x_m(t_e, m, mdot, alpha, beta)
    alpha_c = _adaf_alpha_c(_adaf_tau_es(mdot, alpha), t_e)
    nu_p = _adaf_nu_peak(t_e, x_m, m, mdot, alpha, beta)
    l_nu_p = _adaf_lnu_peak(t_e, nu_p, m)
    l_brems0 = _adaf_lbrems0(t_e, m, mdot, alpha)

    # Synchrotron (nu^{2/5}, nu<nu_p) + Compton (nu^{-alpha_c}, nu>nu_p), joined
    # continuously at nu_p (both = l_nu_p there).
    ratio = nu / nu_p
    shape_sc = jnp.where(nu <= nu_p, ratio**0.4, ratio ** (-alpha_c))
    # Low cutoff at the largest-radius synchrotron frequency (nu ~ r^{-5/4});
    # high cutoff at the Comptonization ceiling 3 k T_e / h.
    nu_min = nu_p * (_R_MIN / _R_MAX) ** 1.25
    nu_max_c = 3.0 * _K_BOLTZ * t_e / _H_PLANCK
    shape_sc = shape_sc * jnp.exp(-nu_min / nu) * jnp.exp(-jnp.clip(nu / nu_max_c, 0.0, 500.0))
    sc = l_nu_p * shape_sc

    # Bremsstrahlung: flat with an exponential cutoff at k T_e / h.
    brems = l_brems0 * jnp.exp(-jnp.clip(_H_PLANCK * nu / (_K_BOLTZ * t_e), 0.0, 500.0))

    total = sc + brems
    # Renormalize to the canonical L_bol (magnitude from agn_log_lbol: the
    # reference on the float32 path). nu descending -> reverse for trapezoid.
    if _f32:
        # ``l_bol_erg`` ~1e44 and the ~1e43 erg/s spectral integral overflow;
        # work the normalization in L_sun (total/L_sun keeps the integral in
        # range) and order 10**log_lbol / integral before the ~1e28 shape.
        integral = jnp.trapezoid((total / _LSUN_ERG)[::-1], nu[::-1])
        # ``representable_floor``, not the bare ``1e-100`` (#1492): float32's
        # smallest subnormal is 1.4e-45, so the literal IS 0.0 there, in this,
        # the float32 branch, the divide-by-zero guard guarded nothing. Returns
        # ``1e-100`` unchanged under x64, so float64 is bit-identical.
        l_nu = (
            (10.0**agn_log_lbol / jnp.maximum(integral, _representable_floor(1e-100)))
            * agn_lum_ratio
            * total
        )
    else:
        integral = jnp.trapezoid(total[::-1], nu[::-1])
        l_nu = (
            10.0**agn_log_lbol
            * _LSUN_ERG
            * agn_lum_ratio
            * total
            / jnp.maximum(integral, _representable_floor(1e-100))
        )
    return l_nu
