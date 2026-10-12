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

from itertools import pairwise
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0 as _i0, i1 as _i1, logsumexp

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
from tengri.utils.scale import pow10 as _pow10

# Fiducial self-similar constants (Mahadevan 1997, Narayan & Yi 1995b).
_C1: float = 0.5
_C3: float = 0.3
_R_MIN: float = 3.0
_R_MAX: float = 1000.0
_ETA_EFF: float = 0.1

# theta_e = k T_e / (m_e c^2)
_THETA_PER_TE: float = _K_BOLTZ / (_M_ELECTRON * _C_LIGHT**2)

# Natural logs of 10 and of L_sun [erg/s] for the log-space power bookkeeping (#2783).
_LN10: float = float(np.log(10.0))
_LOG_3K_OVER_H: float = float(np.log(3.0 * _K_BOLTZ / _H_PLANCK))
_LOG_LSUN_ERG: float = float(np.log(_LSUN_ERG))


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
    heating rate (Eq. 10) and hence the total ADAF luminosity (Sect. 5.2; Eqs. 13, 49).

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

    **Precision**: ``ln c`` is the sum of the logs of the bracket's factors. The bracket
    itself reaches ~4e39 at the electron-temperature floor, above the float32 maximum, and
    the ratio-first form returned ``inf`` there and a stuck Newton iterate (#2783).
    """
    n_e, b_field = _adaf_ne_b_rmin(m, mdot, alpha, beta)
    r_cm = _R_MIN * _R_SCHW_PER_M * m
    theta = _THETA_PER_TE * jnp.asarray(t_e, dtype=jnp.float64)
    x_arg = 1.0 / theta
    # ln of the bracket 2.49e-10 * 4 pi n_e R / (B theta^3 K_2(1/theta)), summed term by term.
    # The bracket itself (~4e39 at the T_e floor, mdot at its clip) overflows float32 (max
    # 3.4e38), so forming the ratio first turned ln_c into inf and the Newton solve for x_M
    # stuck at its upper bound (#2783). Each term is representable on its own; the
    # unscaled K_2 is taken as ln K_2 = ln(K_2 e^x) - x, so e^{-x} never forms.
    log_k2 = jnp.log(_bessel_k2e(x_arg)) - x_arg  # ln[K_2(1/theta_e)]
    ln_c = (
        jnp.log(2.49e-10 * 4.0 * jnp.pi)
        + jnp.log(n_e)
        + jnp.log(r_cm)
        - jnp.log(b_field)
        - 3.0 * jnp.log(theta)
        - log_k2
    )

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


# ── Electron energy balance (Mahadevan Sect. 3, 4 and 5.1.1; Eqs. 8, 9, 13, 14, 26, 29, 35) ──
#
# The electron temperature is the root of the paper's full balance (Eq. 8)
#
#     Q_e+ = Q_ie + delta Q+ = Q- = P_synch + P_Compton + P_brems,
#
# at every point of the box. Each side is evaluated as a logarithm: the powers are ~1e38
# erg/s, which float32 cannot hold.

# Q_ie = 1.2e38 g(theta_e) c1^-2 c3 beta r_min^-1 alpha^-2 m mdot^2 [erg/s] (Eq. 13).
_Q_IE_COEFF: float = 1.2e38 * _C1**-2.0 * _C3 / _R_MIN
_LOG_Q_IE_COEFF: float = float(np.log(_Q_IE_COEFF))
# Q+ = 9.39e38 ((1 - beta)/f) c3 m mdot r_min^-1 [erg/s] (Eq. 9); eps' = (1 - beta)/f in Eq. 14.
_Q_PLUS_COEFF: float = 9.39e38 * _C3 / _R_MIN
_LOG_Q_PLUS_COEFF: float = float(np.log(_Q_PLUS_COEFF))
# Fraction of the viscously dissipated energy that is advected, f in Eq. 9. The paper solves
# for it with the ion balance; the scalings of Eqs. 15 and 49 are quoted at f = 1.
_F_ADVECTED: float = 1.0
# P_synch = 0.71 nu_p L_nu_p (Eq. 26).
_P_SYNCH_OVER_NUPLNUP: float = 0.71
_LOG_P_SYNCH_FRAC: float = float(np.log(_P_SYNCH_OVER_NUPLNUP))
# P_brems = 4.78e34 alpha^-2 c1^-2 ln(r_max/r_min) F(theta_e) m mdot^2 [erg/s] (Eq. 29).
_LOG_P_BREMS_COEFF: float = float(np.log(4.78e34 * _C1**-2.0 * np.log(_R_MAX / _R_MIN)))
# Bracket for the balance root [K]. The upper end is a derived bound, not a margin: the root
# cannot exceed the temperature at which synchrotron cooling alone equals the largest heating
# over the clip domain. With x_M >= 1 (Eq. 20 floor), P_synch >= 0.71 s1^3 (1.19e-13)^3 s3
# r_min^(-7/4) m^(1/2) mdot^(3/2) T_e^7 (Eqs. 21-24, 26), while Q_ie + delta Q+ <= (Eq. 13 with
# g <= 365 for T_e >= 1e8 K) + (Eqs. 9, 14). The ratio is largest at the box corner
# m = 1e10, alpha = 0.5, delta = 0.5, mdot = 1e-8 (the low edge of the mdot clip), where the
# bound is 8.03e11 K; the true root there is 6.27e11 K (x_M > 1 and Compton cooling lower it).
# 1e12 K therefore encloses every root for mdot in [1e-8, mdot_crit] and the declared box.
_T_LO: float = 1e8
_T_HI: float = 1e12


def _adaf_log_q_ie(
    t_e: jnp.ndarray, m: float, mdot: float, alpha: float, beta: float
) -> jnp.ndarray:
    r"""Natural log of the ion-electron Coulomb heating ``Q_ie`` (Mahadevan Eq. 13).

    .. math::

        Q_{ie} = 1.2\times10^{38}\,g(\theta_e)\,c_1^{-2}c_3\,\beta\,
            r_{\min}^{-1}\,\alpha^{-2}\,m\,\dot m^2 \quad [\mathrm{erg\,s^{-1}}]

    with :math:`g(\theta_e)` from Eq. 11.

    Parameters
    ----------
    t_e : array_like, shape ()
        Electron temperature [K].
    m : float
        Black hole mass ``M / M_sun`` [dimensionless].
    mdot : float
        Accretion rate in Eddington units [dimensionless].
    alpha : float
        Viscosity parameter [dimensionless].
    beta : float
        Gas-to-total pressure ratio [dimensionless].

    Returns
    -------
    ndarray, shape ()
        ``ln Q_ie`` with ``Q_ie`` in [erg/s].

    Notes
    -----
    **JIT/grad-safe**: yes. Computed in log space for float32.
    """
    return (
        # ln of the constant is added, not differentiated: d/dbeta of ln(4.8e37 beta) formed as
        # ln(4.8e37 beta) puts a cotangent of order 1e-39 on a float32 subnormal, which the
        # backend flushes to zero (the beta gradient of the balance was 46% low, #2825).
        _LOG_Q_IE_COEFF
        + jnp.log(beta)
        + jnp.log(_adaf_g_theta(t_e))
        + jnp.log(m)
        + 2.0 * jnp.log(mdot)
        - 2.0 * jnp.log(alpha)
    )


def _adaf_log_delta_q_plus(m: float, mdot: float, beta: float, delta: float) -> jnp.ndarray:
    r"""Natural log of the direct viscous electron heating ``delta Q+`` (Mahadevan Eqs. 9, 14).

    .. math::

        \delta Q^+ = \delta\,9.39\times10^{38}\,\epsilon'\,c_3\,m\,\dot m\,r_{\min}^{-1}
            \quad [\mathrm{erg\,s^{-1}}], \qquad \epsilon' = \frac{1-\beta}{f},

    with :math:`f = 1` (Eq. 9 carries the factor :math:`(1-\beta)/f`; the extracted text of
    Eq. 14 does not print :math:`\epsilon'`, and this reading reproduces the transition
    accretion rate of Eq. 15, :math:`\dot m \simeq 8.8\times10^{-5}` at the fiducial values).

    Parameters
    ----------
    m, mdot, beta, delta : float
        Mass ``M / M_sun``, accretion rate in Eddington units, gas-to-total pressure ratio,
        and electron viscous-heating fraction [dimensionless].

    Returns
    -------
    ndarray, shape ()
        ``ln(delta Q+)`` with ``delta Q+`` in [erg/s].

    Notes
    -----
    **JIT/grad-safe**: yes. Independent of ``T_e``.
    """
    return (
        _LOG_Q_PLUS_COEFF
        + jnp.log(delta)
        + jnp.log(1.0 - beta)
        - jnp.log(_F_ADVECTED)
        + jnp.log(m)
        + jnp.log(mdot)
    )


def _adaf_log_cooling_terms(
    t_e: jnp.ndarray, m: float, mdot: float, alpha: float, beta: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Natural logs of the three electron cooling rates (Mahadevan Eqs. 26, 29, 35).

    .. math::

        P_{\rm synch} = 0.71\,\nu_p L_{\nu_p}, \qquad
        P_{\rm brems} = 4.78\times10^{34}\,\alpha^{-2}c_1^{-2}\ln(r_{\max}/r_{\min})\,
            F(\theta_e)\,m\,\dot m^2,

    .. math::

        P_{\rm Compton} = \frac{\nu_p L_{\nu_p}}{1-\alpha_c}
            \left[\left(\frac{3 k T_e/h}{\nu_p}\right)^{1-\alpha_c} - 1\right]
            \quad [\mathrm{erg\,s^{-1}}],

    where :math:`\nu_p L_{\nu_p}` is Eq. 23 at :math:`r_{\min}` (equal to Eq. 24),
    :math:`\alpha_c` is Eq. 34, :math:`F(\theta_e)` is Eq. 28, and :math:`3kT_e/h` is the upper
    limit of the Compton integral (Sect. 4.3: :math:`h\nu_f = 3kT_e`). Eq. 35 prints it as
    :math:`6.2\times10^{19}\,(T_e/10^9\,{\rm K})` Hz; the exact constant
    (:math:`6.251\times10^{19}`, 0.8% larger) is used here so that the solve and the spectrum
    share one ceiling. Eq. 35 is evaluated as
    :math:`\nu_p L_{\nu_p}\,\ln X\,(e^{y}-1)/y` with :math:`X` the bracketed ratio and
    :math:`y=(1-\alpha_c)\ln X`, so no erg/s-scale product forms and the removable singularity
    at :math:`\alpha_c = 1` is the series :math:`(e^y-1)/y \approx 1 + y/2`. It is continuous
    through :math:`\alpha_c = 1` and reduces to :math:`\nu_p L_{\nu_p}/(\alpha_c-1)` for
    :math:`\alpha_c \gg 1` (the Compton term of Eq. 39).

    Parameters
    ----------
    t_e : array_like, shape ()
        Electron temperature [K].
    m, mdot, alpha, beta : float
        Mass, accretion rate, viscosity, gas-to-total pressure ratio.

    Returns
    -------
    tuple of ndarray, shape ()
        ``(ln P_synch, ln P_Compton, ln P_brems)``, each with the power in [erg/s].

    Notes
    -----
    Valid where ``X > 1`` (so that ``ln X > 0``); the bracket ``[_T_LO, _T_HI]`` lies inside
    that domain for the declared box.

    **JIT/grad-safe**: yes. Branchless; the series branch is selected with :func:`jnp.where`
    on a safe argument.
    """
    x_m = _adaf_x_m(t_e, m, mdot, alpha, beta)
    nu_p = _adaf_nu_peak(t_e, x_m, m, mdot, alpha, beta)
    log_nu_l_nu_p = jnp.log(nu_p) + _LN10 * _adaf_log10_lnu_peak(t_e, nu_p, m)
    alpha_c = _adaf_alpha_c(_adaf_tau_es(mdot, alpha), t_e)
    log_x = _LOG_3K_OVER_H + jnp.log(t_e) - jnp.log(nu_p)
    y = (1.0 - alpha_c) * log_x
    small = jnp.abs(y) < 1e-6
    y_safe = jnp.where(small, 1.0, y)
    ratio = jnp.where(small, 1.0 + 0.5 * y, jnp.expm1(y_safe) / y_safe)
    log_p_compton = log_nu_l_nu_p + jnp.log(log_x) + jnp.log(ratio)
    log_p_synch = _LOG_P_SYNCH_FRAC + log_nu_l_nu_p
    log_p_brems = (
        _LOG_P_BREMS_COEFF
        - 2.0 * jnp.log(alpha)
        + jnp.log(_adaf_F_theta(t_e))
        + jnp.log(m)
        + 2.0 * jnp.log(mdot)
    )
    return log_p_synch, log_p_compton, log_p_brems


def _adaf_balance_residual(
    log_te: jnp.ndarray, m: float, mdot: float, alpha: float, beta: float, delta: float
) -> jnp.ndarray:
    r"""Full electron energy-balance residual (Mahadevan Eq. 8).

    .. math::

        R = \ln(Q_{ie} + \delta Q^+) - \ln(P_{\rm synch} + P_{\rm Compton} + P_{\rm brems}),

    zero at the electron temperature of the balance of Sect. 5.1.1. It decreases in
    :math:`\ln T_e` on the solve bracket: the heating falls as :math:`g(\theta_e)` falls
    (:math:`\delta Q^+` is independent of :math:`T_e`), while the cooling rises through
    :math:`\nu_p L_{\nu_p} \propto T_e^7`.

    Parameters
    ----------
    log_te : array_like, shape ()
        Natural log of the electron temperature, ``ln(T_e / K)``.
    m, mdot, alpha, beta, delta : float
        Mass, accretion rate, viscosity, gas-to-total pressure ratio, electron
        viscous-heating fraction.

    Returns
    -------
    ndarray, shape ()
        The residual [dimensionless].

    Notes
    -----
    **JIT/grad-safe**: yes.
    """
    t_e = jnp.exp(log_te)
    log_heat = jnp.logaddexp(
        _adaf_log_q_ie(t_e, m, mdot, alpha, beta), _adaf_log_delta_q_plus(m, mdot, beta, delta)
    )
    log_cool = logsumexp(jnp.stack(_adaf_log_cooling_terms(t_e, m, mdot, alpha, beta)))
    return log_heat - log_cool


# ── Safeguarded secant (shared by the T_e solve and the mdot solve) ──────

# Iteration cap of the secant loops. The loops exit on convergence (typically 6 to 12 steps);
# the cap only bounds a pathological input.
_N_MAX: int = 64
# Starting electron temperature [K] for the secant: the paper's high-mdot value (Sect. 5.1.1).
_T_START: float = 2.0e9


def _adaf_solver_tol(dtype) -> float:
    """Residual tolerance of the secant loops: 1000 machine epsilons in float64, 500 in float32.

    The residuals are differences of logarithms of order 90, whose rounding floor is about
    90 eps (measured 5.7e-14 in float64); the float64 tolerance, 2.2e-13, sits a factor
    four above it so that recomputing the residual at ``ln exp(u)`` stays inside it, and
    the float32 one, 6e-5, a factor five above its floor (float32 also takes a second
    iterate below it).
    """
    eps = float(jnp.finfo(dtype).eps)
    return (1000.0 if eps < 1e-10 else 500.0) * eps


def _bracketed_secant(f, u0, aux0, ln_lo, ln_hi, *, sign, slope0, frozen=None):
    r"""Root of a monotone scalar residual by a safeguarded secant, exiting on convergence.

    ``f(u, aux) -> (r, aux)`` is the residual and a carried auxiliary state (a warm start for
    a nested solve). ``sign`` is ``+1`` when ``r`` decreases with ``u`` and ``-1`` when it
    increases. The bracket ``[ln_lo, ln_hi]`` is narrowed by the sign of ``r`` at each
    iterate; a secant step that leaves the closed bracket is replaced by its midpoint. The
    first step uses the slope magnitude ``slope0``; later steps use the secant slope (a
    non-finite step falls outside the bracket and is bisected). A lane is done at the first
    iterate with :math:`|r| \le` :func:`_adaf_solver_tol` in float64 and at the second
    consecutive one in float32 (whose tolerance is within a factor five of its rounding floor),
    when the bracket has collapsed to a few machine epsilons (a root at or beyond an end of the
    bracket), or at the cap ``_N_MAX``; lanes flagged ``frozen`` do not iterate (under ``vmap``
    a finished lane is held while the others continue). A lane still open at the cap did not
    converge and returns NaN in ``u``, so the failure propagates instead of returning an iterate.

    Returns ``(u, aux)``.
    """
    dtype = u0.dtype
    tol = _adaf_solver_tol(dtype)
    eps = float(jnp.finfo(dtype).eps)
    n_polish = 1 if eps < 1e-10 else 2
    r0, aux = f(u0, aux0)
    done0 = jnp.zeros_like(u0, dtype=bool) if frozen is None else frozen
    state = (
        jnp.asarray(0),
        jnp.zeros_like(u0, dtype=jnp.int32),
        u0,
        r0,
        u0,
        r0,
        jnp.broadcast_to(jnp.asarray(ln_lo, dtype), u0.shape),
        jnp.broadcast_to(jnp.asarray(ln_hi, dtype), u0.shape),
        aux,
        done0,
    )

    def cond(st):
        return (st[0] < _N_MAX) & ~st[9]

    def body(st):
        k, n_ok, u, r, u_prev, r_prev, lo, hi, aux, _done = st
        s = sign * r
        lo = jnp.where(s > 0.0, u, lo)
        hi = jnp.where(s < 0.0, u, hi)
        du = u - u_prev
        slope = jnp.where(du == 0.0, -slope0, sign * (r - r_prev) / jnp.where(du == 0.0, 1.0, du))
        u_new = u - s / slope
        inside = (u_new >= lo) & (u_new <= hi)
        u_next = jnp.where(r == 0.0, u, jnp.where(inside, u_new, 0.5 * (lo + hi)))
        r_next, aux_next = f(u_next, aux)
        n_ok_next = jnp.where(jnp.abs(r_next) <= tol, n_ok + 1, 0)
        collapsed = (hi - lo) <= 64.0 * eps * jnp.maximum(1.0, jnp.abs(u_next))
        finished = (n_ok_next >= n_polish) | (r_next == 0.0) | collapsed
        return (k + 1, n_ok_next, u_next, r_next, u, r, lo, hi, aux_next, finished)

    final = jax.lax.while_loop(cond, body, state)
    return jnp.where(final[9], final[2], jnp.nan), final[8]


def _adaf_te_root_raw(
    m, mdot, alpha, beta, delta, u0, t_lo: float = _T_LO, t_hi: float = _T_HI
) -> jnp.ndarray:
    """``ln T_e`` of the Eq. 8 root from the start ``u0`` (no custom derivative rule)."""

    def f(u, aux):
        return _adaf_balance_residual(u, m, mdot, alpha, beta, delta), aux

    u, _ = _bracketed_secant(
        f, u0, None, float(np.log(t_lo)), float(np.log(t_hi)), sign=1.0, slope0=2.0
    )
    return u


def _adaf_solve_te_balance(
    m: float,
    mdot: float,
    alpha: float,
    beta: float,
    delta: float,
    t_lo: float = _T_LO,
    t_hi: float = _T_HI,
    t_start: float = _T_START,
) -> jnp.ndarray:
    r"""Electron temperature of the full energy balance (Mahadevan Eq. 8, Sect. 5.1.1).

    Solves :func:`_adaf_balance_residual` :math:`= 0`, i.e. :math:`Q_{ie} + \delta Q^+ =
    P_{\rm synch} + P_{\rm Compton} + P_{\rm brems}`, for :math:`T_e` by a safeguarded secant in
    :math:`\ln T_e` (:func:`_bracketed_secant`). The bracket ``[t_lo, t_hi]`` is narrowed by the
    sign of the residual at each step (the residual decreases with :math:`\ln T_e`); a secant
    step that leaves the bracket is replaced by its midpoint. The loop exits when the residual
    is below :func:`_adaf_solver_tol` (two consecutive iterates in float32).

    Derivatives are the implicit-function derivatives of the root, exact to every order. The
    solve is wrapped in :func:`jax.lax.custom_root`, whose tangent rule is
    :math:`\partial \ln T_e / \partial p = -(\partial R / \partial p)/(\partial R / \partial
    \ln T_e)` with :math:`R` differentiated by JAX; the iteration itself is never
    differentiated, and ``jax.hessian`` is correct.

    Parameters
    ----------
    m, mdot, alpha, beta, delta : float
        Mass, accretion rate, viscosity, gas-to-total pressure ratio, electron
        viscous-heating fraction.
    t_lo, t_hi : float, optional
        Bracket [K]. The root must lie inside for the result to be meaningful: the residual is
        positive at ``t_lo`` and negative at ``t_hi`` over the declared box and the whole mdot
        clip range ``[1e-8, 0.28 alpha^2]`` (tested; the ceiling is derived in the comment on
        ``_T_HI``).
    t_start : float, optional
        Starting temperature [K]. The converged root does not depend on it (tested).

    Returns
    -------
    ndarray, shape ()
        Electron temperature [K] of the balance root, inside ``[t_lo, t_hi]``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. The loop is a :func:`jax.lax.while_loop` whose trip count
    depends on the data (6 to 12 steps over the box, at most ``_N_MAX``); it runs only inside the
    solve of :func:`jax.lax.custom_root`, which is never differentiated through.
    """

    def _resid(u):
        return _adaf_balance_residual(u, m, mdot, alpha, beta, delta)

    u_start = jnp.log(jnp.asarray(t_start))
    u_start = u_start.astype(jax.eval_shape(_resid, u_start).dtype)

    def _solve(f, u0):
        u, _ = _bracketed_secant(
            lambda u_, aux: (f(u_), aux),
            u0,
            None,
            float(np.log(t_lo)),
            float(np.log(t_hi)),
            sign=1.0,
            slope0=2.0,
        )
        return u

    log_te = jax.lax.custom_root(_resid, u_start, _solve, lambda g, y: y / g(jnp.ones_like(y)))
    return jnp.exp(log_te)


# ── Equilibrium electron temperature (Mahadevan Eq. 8, Sect. 5.1.1) ───────


def _adaf_electron_temperature(
    m: float, mdot: float, alpha: float, beta: float, delta: float
) -> jnp.ndarray:
    r"""Self-consistent equilibrium electron temperature ``T_e`` (Mahadevan Eq. 8, Sect. 5.1.1).

    The electrons are heated by Coulomb coupling to the ions and by a fraction :math:`\delta`
    of the viscous dissipation, and cool by synchrotron, Compton and bremsstrahlung emission:

    .. math::

        Q_{ie} + \delta Q^+ = P_{\rm synch} + P_{\rm Compton} + P_{\rm brems}
        \qquad \text{(Eq. 8)},

    with :math:`Q_{ie}` from Eq. 13, :math:`Q^+` from Eq. 9 (:math:`\epsilon' = (1-\beta)/f`,
    :math:`f = 1`, Eq. 14), :math:`P_{\rm synch} = 0.71\,\nu_p L_{\nu_p}` (Eq. 26),
    :math:`P_{\rm Compton}` from Eq. 35 with the slope :math:`\alpha_c` of Eq. 34 (optical
    depth Eq. 31, amplification Eq. 32), and :math:`P_{\rm brems}` from Eq. 29;
    :math:`\nu_p L_{\nu_p}` is Eqs. 21-23 with :math:`x_M` from Eq. 20. This is the paper's
    numerical method (Sect. 5.1.1: "Q_e+ = P_synch + P_brems + P_Compton"), solved for
    :math:`T_e` at every point by :func:`_adaf_solve_te_balance`, so there is no regime
    junction. The slope :math:`\alpha_c = -\ln\tau_{es}/\ln A` evaluated at the solved
    :math:`T_e` is a diagnostic; it is not an input to the solve.

    The paper's closed forms are limits of this balance, and the tests check them as such:

    - Eq. 40 (:math:`\alpha_c > 1`, :math:`\dot m < 10^{-3}\alpha^2`): :math:`Q_{ie}`,
      :math:`P_{\rm brems}` and the part of the Compton term beyond
      :math:`\nu_p L_{\nu_p}/(\alpha_c-1)` are dropped, leaving
      :math:`\delta Q^+ = A_c\,\nu_p L_{\nu_p}` (Eq. 39).
    - Eqs. 43-45 (:math:`\alpha_c < 1`, :math:`\delta Q^+ \ll Q_{ie}`): :math:`Q_{ie} =
      P_{\rm Compton}`; Eq. 43 is the closed-form inverse of Eq. 34 for a given :math:`\alpha_c`.

    Parameters
    ----------
    m, mdot, alpha, beta, delta : float
        Mass, accretion rate, viscosity, gas-to-total pressure ratio, and electron
        viscous-heating fraction.

    Returns
    -------
    ndarray, shape ()
        Equilibrium electron temperature [K], inside ``[1e8, 1e12]`` K. For the ADAF range
        this is ~1e9-5e9 K.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. First and second derivatives are the exact implicit-function
    derivatives of the solved root (:func:`jax.lax.custom_root`; ``jax.hessian`` is tested
    against finite differences of the gradient). The solve is a :func:`jax.lax.while_loop`
    with a data-dependent trip count, which is never differentiated through.

    **Approximations**: :math:`f = 1` in Eq. 9; :math:`P_{\rm synch} = 0.71\,\nu_p L_{\nu_p}`
    assumes :math:`\nu_{\min} \ll \nu_p` (Eq. 26); Eq. 35 assumes a single Compton power law
    from :math:`\nu_p` to :math:`3kT_e/h`. The balance has one root on the bracket over the
    declared box and the whole mdot clip range ``[1e-8, 0.28 alpha^2]`` (tested).

    **Smoothness**: :math:`T_e` is continuous, but it is not smooth to every order.
    :math:`F(\theta_e)` of Eq. 28 is the paper's piecewise fit, with a 0.16% step at
    :math:`\theta_e = 1` (:math:`F = 6.0257` below, :math:`6.0353` above; :math:`T_e =
    5.93\times10^9` K). The step makes the balance residual jump across :math:`T_e^\ast`, so for
    a band of parameters the root is pinned at :math:`T_e^\ast`, with a derivative kink at each
    end of the band. The band is up to :math:`6.0\times10^{-5}` wide in :math:`\ln T_e`, worst
    at :math:`M = 10^{10}`, :math:`\alpha = 0.05`, :math:`\delta = 10^{-3}`, where bremsstrahlung
    carries up to 21% of the cooling. Inside the band the derivative is that of one branch
    while the function is flat. The derived ``mdot`` has two further derivative
    discontinuities, at its clips (``mdot_crit`` and the ``1e-8`` floor).

    **Eq. 40 prefactor**: the printed :math:`1.1\times10^9` K differs by 7.5% from the
    :math:`1.023\times10^9` K re-derived from Eqs. 9, 14 and 24 at the fiducial point with
    :math:`A_c = 1` (a factor 1.66 in :math:`\delta Q^+/\nu_p L_{\nu_p}`). The balance uses
    the equations, not the printed prefactor; the tests check the printed form against the
    paper's own :math:`A_c^{1/7}` band and the tight limit with the re-derived value.

    **Outside the paper's range**: for :math:`\tau_{es} \geq 1` (:math:`\alpha_c \leq 0`,
    reached at :math:`\alpha \gtrsim 0.4988` on :math:`\dot m_{\rm crit}`), Eq. 34 gives no
    positive Compton slope and the power-law spectrum of Eq. 33 does not apply. The paper
    states :math:`0.5 \leq \alpha_c \leq 1` for "all cases of interest" and does not treat
    :math:`\alpha_c \leq 0`. Eq. 35 is evaluated there as written; it is continuous through
    :math:`\tau_{es} = 1`, so :math:`T_e` is smooth across it, but the value is an
    extrapolation, not a result of the paper.
    """
    return _adaf_solve_te_balance(m, mdot, alpha, beta, delta)


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
    return _pow10(_adaf_log10_lnu_peak(t_e, nu_p, m))


def _adaf_log10_lnu_peak(t_e: jnp.ndarray, nu_p: jnp.ndarray, m: float) -> jnp.ndarray:
    r"""``log10`` of the peak synchrotron luminosity (Mahadevan Eq. 23, at ``r_min``).

    Formed in log10 (utils/scale.pow10): under jit in float32 the intermediate nu_p^2 m^2
    (~5e39) exceeds the float32 maximum (3.4e38), while the product (~2e26) does not, so the
    linear form returned a non-finite log_integral at (log M 6, alpha 0.1, log L 14).
    """
    return (
        jnp.log10(1.05e-24 * _R_MIN**2)
        + jnp.log10(t_e)
        + 2.0 * jnp.log10(nu_p)
        + 2.0 * jnp.log10(m)
    )


_LOG_LBREMS0_CONST: float = float(np.log(2.29e24 * _C1**-2.0 * np.log(_R_MAX / _R_MIN)))


def _adaf_log_lbrems0(t_e: jnp.ndarray, m: float, mdot: float, alpha: float) -> jnp.ndarray:
    r"""Natural log of the bremsstrahlung prefactor (Mahadevan Eq. 30), in log space.

    Same quantity as :func:`_adaf_lbrems0`, with every factor a logarithm of a moderate
    number, so that its derivative is not formed on a float32 subnormal (#2825).
    """
    return (
        _LOG_LBREMS0_CONST
        - 2.0 * jnp.log(alpha)
        + jnp.log(_adaf_F_theta(t_e))
        - jnp.log(t_e)
        + jnp.log(m)
        + 2.0 * jnp.log(mdot)
    )


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


# ── L_bol -> mdot inversion (Mahadevan Sect. 5.2: L_ADAF = Q_e+ = Q_ie + delta Q+) ──

# Accretion-rate clip range: the ADAF solution does not exist above mdot_crit = 0.28 alpha^2
# (Eq. 52); 1e-8 is the floor of the T_e solve.
_MDOT_FLOOR: float = 1e-8
_MDOT_CRIT_COEFF: float = 0.28
# A solve that ends this close (in ln mdot) to the floor is a luminosity below the floor's.
# Below the floor the fixed-g start is clipped onto ln(1e-8) at once: one outer iteration,
# exactly 1e-8, zero gradient. In float32 a bracket collapse would land within 64 epsilons,
# up to 1.4e-4 in ln mdot, which is wider than this snap, so the snap does not rely on it.
_FLOOR_SNAP: float = 1e-9
_LN2: float = float(np.log(2.0))


def _adaf_log_mdot_fixed_g(
    ln_l: jnp.ndarray, ln_a_g: jnp.ndarray, ln_b: jnp.ndarray
) -> jnp.ndarray:
    r"""``ln mdot`` of the radiated power at fixed ``g(theta_e)`` (Mahadevan Sect. 5.2).

    At fixed :math:`g` the radiated power :math:`L = Q_{ie} + \delta Q^+` is the quadratic
    :math:`L = A g\,\dot m^2 + B\,\dot m` with

    .. math::

        A = 1.2\times10^{38}\,c_1^{-2}c_3\,\beta\,r_{\min}^{-1}\alpha^{-2}\,m
          = 4.8\times10^{37}\,\beta\,\alpha^{-2}\,m, \qquad
        B = \delta\,9.39\times10^{38}\,\frac{1-\beta}{f}\,c_3\,r_{\min}^{-1}\,m
          = 9.39\times10^{37}\,\delta\,(1-\beta)\,m \quad (f = 1)

    [erg/s], whose positive root is

    .. math::

        \dot m = \frac{2L}{B + \sqrt{B^2 + 4AgL}}
               = \frac{2x}{1 + \sqrt{1 + 4xy}}, \qquad x = L/B,\; y = Ag/B,

    formed from the logarithms so that no erg/s-scale product (B reaches 1e45 erg/s at
    :math:`m = 10^{10}`) is ever built. For :math:`B \to 0` it is Eq. 49,
    :math:`\dot m = \sqrt{L/Ag}`, and for :math:`Ag \to 0` it is Eq. 50, :math:`\dot m = L/B`.

    Parameters
    ----------
    ln_l, ln_a_g, ln_b : array_like, shape ()
        ``ln L``, ``ln(A g)`` and ``ln B`` with each power in [erg/s].

    Returns
    -------
    ndarray, shape ()
        ``ln mdot`` [dimensionless accretion rate in Eddington units].

    Notes
    -----
    **JIT/grad-safe**: yes. Branchless, logs only.
    """
    ln_x = ln_l - ln_b
    ln_y = ln_a_g - ln_b
    ln_q = 2.0 * _LN2 + ln_x + ln_y
    return _LN2 + ln_x - jax.nn.softplus(0.5 * jnp.logaddexp(0.0, ln_q))


def _adaf_log_heating(
    t_e: jnp.ndarray, m: float, mdot: float, alpha: float, beta: float, delta: float
) -> jnp.ndarray:
    """``ln(Q_ie + delta Q+)`` [erg/s] at ``t_e`` (Mahadevan Eqs. 13, 9, 14)."""
    return jnp.logaddexp(
        _adaf_log_q_ie(t_e, m, mdot, alpha, beta), _adaf_log_delta_q_plus(m, mdot, beta, delta)
    )


def _adaf_log_radiated_power(
    ln_mdot: jnp.ndarray, m: float, alpha: float, beta: float, delta: float
) -> jnp.ndarray:
    r"""``ln L_ADAF`` of Sect. 5.2: ``ln(Q_ie + delta Q+)`` at the balance temperature.

    Parameters
    ----------
    ln_mdot : array_like, shape ()
        Natural log of the accretion rate in Eddington units.
    m, alpha, beta, delta : float
        Mass, viscosity, gas-to-total pressure ratio, electron viscous-heating fraction.

    Returns
    -------
    ndarray, shape ()
        ``ln L_ADAF`` with ``L_ADAF`` in [erg/s]; strictly increasing in ``ln_mdot`` with slope
        at least one (:math:`Q_{ie} \propto g\,\dot m^2`, :math:`\delta Q^+ \propto \dot m`).

    Notes
    -----
    **JIT/grad/vmap-safe**: yes; ``T_e`` enters through :func:`_adaf_solve_te_balance`, so the
    derivatives are the exact nested implicit-function derivatives.
    """
    mdot = jnp.exp(ln_mdot)
    t_e = _adaf_solve_te_balance(m, mdot, alpha, beta, delta)
    return _adaf_log_heating(t_e, m, mdot, alpha, beta, delta)


def _adaf_log_radiated_power_raw(ln_mdot, u_te, m, alpha, beta, delta):
    """Forward value of :func:`_adaf_log_radiated_power`, warm-started at ``ln T_e = u_te``.

    No derivative rule: used inside the solve of :func:`_adaf_solve_mdot` and for the clip
    tests. Returns ``(ln L, ln T_e)``.
    """
    mdot = jnp.exp(ln_mdot)
    u_te = _adaf_te_root_raw(m, mdot, alpha, beta, delta, u_te)
    t_e = jnp.exp(u_te)
    return _adaf_log_heating(t_e, m, mdot, alpha, beta, delta), u_te


def _adaf_solve_mdot(
    ln_l: jnp.ndarray, m: float, alpha: float, beta: float, delta: float
) -> jnp.ndarray:
    """Accretion rate of the radiated power ``exp(ln_l)`` [erg/s], clipped to the ADAF range."""
    ln_a = _LOG_Q_IE_COEFF + jnp.log(beta) + jnp.log(m) - 2.0 * jnp.log(alpha)
    ln_b = _LOG_Q_PLUS_COEFF + jnp.log(delta) + jnp.log(1.0 - beta) - jnp.log(_F_ADVECTED)
    ln_b = ln_b + jnp.log(m)
    mdot_crit = _MDOT_CRIT_COEFF * alpha**2
    ln_hi = jnp.log(mdot_crit)
    ln_lo = jnp.full_like(ln_hi, float(np.log(_MDOT_FLOOR)))
    dtype = ln_hi.dtype
    u_te0 = jnp.full_like(ln_hi, float(np.log(_T_START)))

    # Radiated power at mdot_crit (forward value only). L above it clips to mdot_crit; the root of
    # the monotone ln L(ln mdot) is otherwise inside the bracket, or below the floor.
    sg = jax.lax.stop_gradient
    p_fixed = tuple(sg(x) for x in (m, alpha, beta, delta))
    ln_l_hi, _ = _adaf_log_radiated_power_raw(sg(ln_hi), u_te0, *p_fixed)
    over = sg(ln_l) >= ln_l_hi

    # Closed form at the g of the starting temperature: the exact root of the quadratic for a
    # constant g, and the start of the nested solve.
    ln_g0 = jnp.log(_adaf_g_theta(jnp.asarray(_T_START, dtype)))
    w0 = _adaf_log_mdot_fixed_g(sg(ln_l), sg(ln_a) + ln_g0, sg(ln_b))
    w0 = jnp.clip(w0, ln_lo, sg(ln_hi))
    w0 = jnp.where(over, sg(ln_hi), w0).astype(dtype)

    def _resid(w):
        return _adaf_log_radiated_power(w, m, alpha, beta, delta) - ln_l

    def _solve(f, w_start):
        def g(w, u_te):
            ln_p, u_te = _adaf_log_radiated_power_raw(w, u_te, *p_fixed)
            return ln_p - sg(ln_l), u_te

        w, _ = _bracketed_secant(
            g, w_start, u_te0, ln_lo, sg(ln_hi), sign=-1.0, slope0=1.5, frozen=over
        )
        return w

    w = jax.lax.custom_root(_resid, w0, _solve, lambda g, y: y / g(jnp.ones_like(y)))
    # A luminosity below that of the floor leaves the bracket collapsed onto its lower end.
    under = ~over & (sg(w) <= ln_lo + _FLOOR_SNAP)
    return jnp.where(over, mdot_crit, jnp.where(under, _MDOT_FLOOR, jnp.exp(w)))


def _adaf_mdot_from_lbol(
    l_bol_erg: jnp.ndarray,
    m: float,
    alpha: float,
    beta: float,
    delta: float,
    float32: bool = False,
) -> jnp.ndarray:
    r"""Derive the accretion rate ``mdot`` from the canonical bolometric luminosity.

    The radiated power of an ADAF is the electron heating, :math:`L_{\rm ADAF} = Q^- = Q_e^+ =
    Q_{ie} + \delta Q^+` (Mahadevan Sect. 5.2), evaluated at the electron temperature of the
    Eq. 8 balance:

    .. math::

        L_{\rm ADAF} = 1.2\times10^{38}\,g(\theta_e)\,c_1^{-2}c_3\,\beta\,r_{\min}^{-1}
            \alpha^{-2}\,m\,\dot m^2
          + \delta\,9.39\times10^{38}\,\frac{1-\beta}{f}\,c_3\,r_{\min}^{-1}\,m\,\dot m
          \qquad \text{(Eqs. 13, 14)},

    with :math:`g(\theta_e)` from Eq. 11 at :math:`T_e(\dot m)`. Eq. 49 is its limit for
    :math:`\dot m > 10^{-3}\alpha^2` (the first term dominates; it "also gives the luminosity for
    the case :math:`\delta = 0`") and Eq. 50 its limit for :math:`\dot m < 10^{-3}\alpha^2` (the
    second term dominates); the inversion solves the sum, so it holds in both and between.
    Keeping ``agn_log_lbol`` as the canonical luminosity knob (deriving ``mdot`` rather than
    taking it as an independent input) is what makes the ADAF consistent with the disc
    convention: ``agn_log_ledd`` is retired here.

    The solve is a safeguarded secant in :math:`\ln\dot m` on :math:`[10^{-8}, \dot m_{\rm crit}]`
    for the strictly increasing :math:`\ln L_{\rm ADAF}(\ln\dot m)` (slope at least one), with
    the quadratic root at fixed :math:`g` (:func:`_adaf_log_mdot_fixed_g`) as its start and the
    :math:`T_e` solve nested inside, warm-started from the previous iterate. It is wrapped in
    :func:`jax.lax.custom_root`, so derivatives of every order through the nested
    implicit solves are exact.

    The result is clipped to ``[1e-8, mdot_crit]`` with ``mdot_crit = 0.28 alpha^2`` (Eq. 52):
    the ADAF solution does not exist above the critical rate. A luminosity above that of
    ``mdot_crit`` returns ``mdot_crit`` and its derivative is that of ``0.28 alpha^2``.

    Parameters
    ----------
    l_bol_erg : array_like
        ADAF bolometric (radiated) luminosity [erg/s]; in :math:`L_\odot` when ``float32``.
    m, alpha, beta, delta : float
        Mass, viscosity, gas-to-total pressure, electron-heating fraction.
    float32 : bool, optional
        Take ``l_bol_erg`` in :math:`L_\odot` (so that the :math:`10^{44}` erg/s value never
        forms in pure float32); the solve works with its logarithm either way.

    Returns
    -------
    ndarray
        ``mdot`` in Eddington units.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes; data-dependent loop trip counts only inside the solves of
    :func:`jax.lax.custom_root`.

    **Approximation**: :math:`f = 1` in Eq. 9; the electron temperature carries the
    approximations of :func:`_adaf_electron_temperature`. The renormalization of the spectrum to
    ``agn_log_lbol`` in :func:`adaf_scalar_state` then rescales only the simplified spectral
    shape against the integrated cooling of the balance.
    """
    ln_l = jnp.log(jnp.asarray(l_bol_erg))
    if float32:
        ln_l = ln_l + _LOG_LSUN_ERG
    return _adaf_solve_mdot(ln_l, m, alpha, beta, delta)


def _adaf_mdot_of_log_lbol(
    agn_log_lbol: float, m: float, alpha: float, beta: float, delta: float, float32: bool
) -> jnp.ndarray:
    r"""``mdot`` of the radiated luminosity ``10**agn_log_lbol L_sun``, in the dtype's own form.

    In float32 the luminosity is passed in :math:`L_\odot` (so the ~1e44 erg/s value never
    forms); in float64 it is formed in erg/s. Both are :func:`_adaf_mdot_from_lbol`.

    Parameters
    ----------
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm ADAF}/L_\odot)` [dex].
    m, alpha, beta, delta : float
        Mass, viscosity, gas-to-total pressure ratio, electron viscous-heating fraction.
    float32 : bool
        Whether the evaluation runs in float32.

    Returns
    -------
    ndarray
        ``mdot`` in Eddington units [dimensionless].
    """
    if float32:
        return _adaf_mdot_from_lbol(10.0**agn_log_lbol, m, alpha, beta, delta, float32=True)
    return _adaf_mdot_from_lbol(10.0**agn_log_lbol * _LSUN_ERG, m, alpha, beta, delta)


#: Mahadevan (1997) Eq. 4: ``Mdot_Edd = L_Edd / (eta_eff c^2) = 1.39e18 m g/s`` with
#: ``eta_eff = 0.1``, so ``mdot L_Edd = eta_eff Mdot c^2``: the power the paper compares the
#: ADAF's radiated luminosity ``L_ADAF`` against.
_MDOT_EDD_G_PER_S_PER_MSUN: float = 1.39e18
_LOG10_ETA_EFF_MDOT_EDD_C2_ERG_S_PER_MSUN: float = float(
    np.log10(_ETA_EFF * _MDOT_EDD_G_PER_S_PER_MSUN * _C_LIGHT**2)
)


def adaf_accretion_diagnostics(
    agn_log_lbol: float,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_adaf_alpha: float = 0.3,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.1,
    *,
    dtype=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Accretion rate and accretion power of the ADAF that radiates ``agn_log_lbol``.

    ``agn_log_lbol`` is the radiated luminosity :math:`L_{\rm ADAF} = Q_{ie} + \delta Q^+`
    (Mahadevan 1997, Sect. 5.2, Eqs. 13 and 14); the accretion rate follows from it
    (:func:`_adaf_mdot_from_lbol`). The paper scales it by the Eddington rate of Eq. 4,

    .. math::

        \dot M = \dot m\,\dot M_{\rm Edd}, \qquad
        \dot M_{\rm Edd} = \frac{L_{\rm Edd}}{\eta_{\rm eff} c^2} = 1.39\times10^{18}\,m\
        \mathrm{g\,s^{-1}}, \qquad \eta_{\rm eff} = 0.1 ,

    with :math:`m = M/M_\odot`. The accretion power published is the one the paper compares the
    radiated luminosity against, the Eddington luminosity times the accretion rate in Eddington
    units,

    .. math::

        \dot m\,L_{\rm Edd} = \eta_{\rm eff}\,\dot M c^2
        = \dot m\,(1.39\times10^{18}\,m\ \mathrm{g\,s^{-1}})\,\eta_{\rm eff} c^2
        \quad [\mathrm{erg\,s^{-1}}] ,

    :math:`c` the speed of light [cm/s] and :math:`L_{\rm Edd}` the Eddington luminosity
    [erg/s]. The efficiency is the paper's :math:`\eta_{\rm eff}` of Eq. 4, not a model
    choice; it is the conversion the paper itself uses to define :math:`\dot m`. The rest-mass
    rate :math:`\dot M c^2` is :math:`1/\eta_{\rm eff} = 10` times larger. ``agn_log_lbol``
    stays the ADAF's radiated :math:`Q_{ie} + \delta Q^+`, which is
    :math:`L_{\rm ADAF} < \dot m L_{\rm Edd}`.

    Parameters
    ----------
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm ADAF}/L_\odot)`, the radiated luminosity the spectrum is
        normalized to [dex].
    agn_log_mbh : float, optional
        :math:`\log_{10}(M_{\rm BH}/M_\odot)` [dex]. Default ``DEFAULT_AGN_LOG_MBH``.
    agn_adaf_alpha : float, optional
        Viscosity parameter :math:`\alpha` [dimensionless]. Default 0.3.
    agn_adaf_beta : float, optional
        Gas-to-total pressure ratio :math:`\beta` [dimensionless]. Default 0.5.
    agn_adaf_delta : float, optional
        Electron viscous-heating fraction :math:`\delta` [dimensionless]. Default 0.1.
    dtype : dtype, optional
        Selects the float32 branch of the :math:`\dot m` solve (luminosity in
        :math:`L_\odot`). Default: the canonical float dtype of the process.

    Returns
    -------
    mdot : ndarray, shape ()
        :math:`\dot m = \dot M/\dot M_{\rm Edd}` [dimensionless], clipped to the ADAF range
        ``[1e-8, 0.28 alpha^2]`` (Eq. 52).
    log_accretion_power : ndarray, shape ()
        :math:`\log_{10}(\dot m L_{\rm Edd} / \mathrm{erg\,s^{-1}})` [dex re erg/s], formed in
        log space (the power reaches 1e44 erg/s).

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes; the same solve as the spectrum's own accretion rate.
    The derivative is discontinuous where :math:`\dot m` meets its clip.

    Implements the same model as Mahadevan (1997 [1]_).

    References
    ----------
    .. [1] Mahadevan, R. 1997, "Scaling Laws for Advection-dominated Flows: Applications to
       Low-Luminosity Galactic Nuclei", ApJ, 477, 585. arXiv:astro-ph/9609107.
       https://doi.org/10.1086/303727
    """
    dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
    m = 10.0**agn_log_mbh
    mdot = _adaf_mdot_of_log_lbol(
        agn_log_lbol, m, agn_adaf_alpha, agn_adaf_beta, agn_adaf_delta, dtype == jnp.float32
    )
    log_power = jnp.log10(mdot) + agn_log_mbh + _LOG10_ETA_EFF_MDOT_EDD_C2_ERG_S_PER_MSUN
    return mdot, log_power


# ── Gauss-Legendre quadrature for the normalization integral ─────────────
#
# The spectrum is smooth in ln(nu) between its own breaks (nu_min, nu_p, 3 k T_e / h), so the
# power is integrated by one 30-point Gauss-Legendre rule per segment between them, in
# ln(nu), instead of a fixed grid; the bremsstrahlung tail has a closed form.

_GL_ORDER = 30
_GL_X_RAW, _GL_W_RAW = np.polynomial.legendre.leggauss(_GL_ORDER)  # on [-1, 1]
_GL_X = (_GL_X_RAW + 1.0) / 2.0  # nodes on [0, 1]
_GL_W = _GL_W_RAW / 2.0  # weights on [0, 1]


def _segment_log_power(log_f, nu_a, nu_b, dtype):
    r"""``log`` of the Gauss-Legendre integral of ``f(nu) d nu`` over ``[nu_a, nu_b]``.

    The quadrature runs in ``ln(nu)`` and is summed in log space (log-sum-exp over the
    nodes), so no linear erg/s-scale product forms and float32 cotangents stay bounded by
    the loss (#2783).

    Parameters
    ----------
    log_f : callable
        ``log_f(nu)``: the natural log of the integrand, smooth on the segment.
    nu_a, nu_b : array_like, shape ()
        Segment limits [Hz], ``nu_a < nu_b``.
    dtype : dtype
        Working dtype of the nodes and weights (float32 stays float32).

    Returns
    -------
    ndarray, shape ()
        :math:`\ln \int_{\nu_a}^{\nu_b} f\,d\nu` with ``f`` in its own units times Hz.
    """
    log_a, log_b = jnp.log(nu_a), jnp.log(nu_b)
    width = log_b - log_a
    log_nu = log_a + jnp.asarray(_GL_X, dtype=dtype) * width
    nu = jnp.exp(log_nu)
    terms = jnp.asarray(np.log(_GL_W), dtype=dtype) + log_f(nu) + log_nu
    return jnp.log(width) + logsumexp(terms)


# ── Public spectrum ──────────────────────────────────────────────────────


class AdafState(NamedTuple):
    r"""Wavelength-independent part of :func:`adaf_spectrum`: the solved ADAF and its power.

    All fields are scalars. The spectrum is

    .. math::

        L_\nu = r\,\exp\!\left[\ln P_{\rm bol} - \ln P_{\rm shape} + \ln(S + B)\right],

    with :math:`S` the synchrotron + Compton shape and :math:`B` the bremsstrahlung shape
    (``_adaf_log_total``); :math:`P_{\rm shape} = \int (S + B)\,d\nu` over the model's whole
    support and :math:`P_{\rm bol}` the L_bol power, so :math:`\int L_\nu d\nu = r\,P_{\rm
    bol}` with :math:`r` = ``lum_ratio``. Both powers are carried as natural logarithms of
    erg/s (#2783).
    """

    t_e: jnp.ndarray  # electron temperature [K]
    alpha_c: jnp.ndarray  # Compton slope [dimensionless]
    nu_p: jnp.ndarray  # self-absorption peak [Hz]
    nu_min: jnp.ndarray  # low cutoff [Hz]
    nu_max_c: jnp.ndarray  # Comptonization ceiling 3 k T_e / h [Hz]
    l_nu_p: jnp.ndarray  # peak L_nu before normalization [erg/s/Hz]
    l_brems0: jnp.ndarray  # bremsstrahlung level before normalization [erg/s/Hz]
    log_l_nu_p: jnp.ndarray  # ln of l_nu_p, formed in log space (#2825)
    log_l_brems0: jnp.ndarray  # ln of l_brems0, formed in log space (#2825)
    log_nu_max_c: jnp.ndarray  # ln of nu_max_c [Hz], formed from ln T_e (#2825)
    log_numer: jnp.ndarray  # ln of the L_bol power 10**agn_log_lbol * L_sun [erg/s]
    log_integral: jnp.ndarray  # ln of the power of the unnormalized shape [erg/s]
    lum_ratio: jnp.ndarray  # linear agn_lum_ratio multiplying the spectrum [dimensionless]


def _adaf_log_synch_compton(nu_, s):
    """``log`` of the synchrotron + Compton term, ``log(l_nu_p) + log(shape)``.

    The shape is synchrotron ``nu^{2/5}`` below ``nu_p`` and Compton ``nu^{-alpha_c}`` above,
    joined at ``nu_p``, with the low cutoff ``exp(-nu_min/nu)`` and the Comptonization
    ceiling ``exp(-nu/nu_max_c)`` (clipped at 500). Each factor is a log, so the term is
    formed in log space (#2783).
    """
    log_nu = jnp.log(nu_)
    log_ratio = log_nu - jnp.log(s.nu_p)
    log_shape = jnp.where(nu_ <= s.nu_p, 0.4 * log_ratio, -s.alpha_c * log_ratio)
    # Both cutoff arguments are exp(difference of logs). The quotient nu / nu_max_c has a
    # reverse-mode partial -nu / nu_max_c**2, and nu_max_c**2 ~ 1e39 exceeds the float32
    # maximum. The ceiling reads the logarithm carried by the state: differentiating
    # log(nu_max_c) from the linear field puts a cotangent of order 1e-39 on nu_max_c, a
    # float32 subnormal that the backend flushes to zero, which dropped the T_e derivative of
    # the Comptonization ceiling (1% of d(sum L_nu)/dT_e, #2825).
    low_cut = jnp.exp(jnp.log(s.nu_min) - log_nu)
    ceiling = jnp.clip(jnp.exp(log_nu - s.log_nu_max_c), 0.0, 500.0)
    log_shape = log_shape - low_cut - ceiling
    return s.log_l_nu_p + log_shape


def _adaf_log_brems(nu_, s):
    """``log`` of the bremsstrahlung term: flat, with an exponential cutoff at ``k T_e / h``."""
    return s.log_l_brems0 - jnp.clip(_H_PLANCK * nu_ / (_K_BOLTZ * s.t_e), 0.0, 500.0)


def _adaf_log_total(nu_, s):
    """``log(S + B)``: the synchrotron + Compton and bremsstrahlung terms, summed in log space.

    Every term is a positive exponential or power law, so the normalized spectrum
    ``exp(log numer - log integral + log(S + B))`` never forms an erg/s-scale linear product.
    In float32 the linear product's reverse pass overflowed at the clipped-mdot corner and
    returned NaN gradients (#2783).
    """
    return jnp.logaddexp(_adaf_log_synch_compton(nu_, s), _adaf_log_brems(nu_, s))


def _adaf_log_brems_power(
    log_l_brems0: jnp.ndarray, t_e: jnp.ndarray, nu_lo: jnp.ndarray
) -> jnp.ndarray:
    r"""Natural log of the bremsstrahlung power between ``nu_lo`` and ``100 k T_e / h`` (Eq. 30).

    .. math::

        \int_{\nu_{lo}}^{100 kT_e/h} L_{\rm br}\,e^{-h\nu/kT_e}\,d\nu
          = L_{\rm br}\,\frac{kT_e}{h}\left[e^{-a} - e^{-100}\right], \qquad a = h\nu_{lo}/kT_e,

    formed as :math:`\ln L_{\rm br} + \ln t - a + \ln(1 - e^{-(100-a)})` with
    :math:`t = kT_e/h`. The upper limit is :math:`100\,t`, so :math:`(\nu_{hi} - \nu_{lo})/t =
    100 - a`; the quotient :math:`a` is :math:`e^{\ln\nu_{lo} - \ln t}` because
    :math:`t^2 \sim 10^{39}` overflows the float32 reverse partial of :math:`\nu_{lo}/t`.

    Parameters
    ----------
    log_l_brems0 : array_like, shape ()
        ``ln`` of the level ``L_br`` [erg/s/Hz].
    t_e : array_like, shape ()
        Electron temperature [K].
    nu_lo : array_like, shape ()
        Lower frequency of the support [Hz].

    Returns
    -------
    ndarray, shape ()
        ``ln`` of the power [erg/s].

    Notes
    -----
    **JIT/grad-safe**: yes. Algebraically identical to the linear closed form (tested in
    float64).
    """
    t_scale = _K_BOLTZ * t_e / _H_PLANCK
    lo_over_scale = jnp.exp(jnp.log(nu_lo) - jnp.log(t_scale))
    return (
        log_l_brems0
        + jnp.log(t_scale)
        - lo_over_scale
        + jnp.log1p(-jnp.exp(-(100.0 - lo_over_scale)))
    )


def adaf_scalar_state(
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_adaf_alpha: float = 0.3,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.1,
    agn_log_lbol_shape: float | None = None,
    *,
    dtype=None,
) -> AdafState:
    r"""Solve the ADAF once: everything :func:`adaf_spectrum` needs that is not a wavelength.

    Derives the accretion rate from the shape luminosity (Mahadevan Sect. 5.2, Eqs. 13, 14),
    solves the electron temperature (the Eq. 8 balance: Eqs. 9, 13, 26, 29, 35), builds the
    spectrum's break frequencies and amplitudes (Eqs. 21-23, 28, 30), and computes the power of
    that shape over the model's whole support, so that the spectrum can be normalized without
    touching any wavelength grid.

    The power is

    .. math::

        P = \sum_{k=1}^{4} \int_{\nu_k}^{\nu_{k+1}} S(\nu)\,d\nu
          + L_{\rm br}\,\frac{k T_e}{h}
            \left[e^{-h\nu_0/kT_e} - e^{-h\nu_4/kT_e}\right],

    where :math:`S` is the synchrotron + Compton shape [erg/s/Hz], :math:`L_{\rm br}` the
    bremsstrahlung level [erg/s/Hz], and the breaks are :math:`\nu_0 = 0.02\,\nu_{\min}`,
    :math:`\nu_1 = \nu_{\min}`, :math:`\nu_2 = \nu_p`, :math:`\nu_3 = 3kT_e/h`,
    :math:`\nu_4 = 100\,kT_e/h` [Hz]. The bremsstrahlung integral
    :math:`\int L_{\rm br} e^{-h\nu/kT_e} d\nu` is closed-form (the exponent clip at 500
    never binds below :math:`\nu_4`); each synchrotron + Compton segment is a 30-point
    Gauss-Legendre rule in :math:`\ln\nu`. The only non-smooth point of :math:`S` inside the
    support, the switch at :math:`\nu_p`, is a segment boundary. Measured against an
    independent 2e6-node trapezoid over a wider span: 3.9e-12 relative (declared prior box
    corners and 300 draws). The target power is
    :math:`10^{\mathtt{agn\_log\_lbol}} L_\odot \times \mathtt{agn\_lum\_ratio}` [erg/s].

    Parameters
    ----------
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` the spectrum is normalized to (the float32
        reference magnitude on the float32 path).
    agn_lum_ratio : float, optional
        Fraction of ``L_bol`` assigned to the ADAF [dimensionless]. Default
        ``DEFAULT_AGN_LUM_RATIO``.
    agn_log_mbh : float, optional
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`. Default ``DEFAULT_AGN_LOG_MBH``.
    agn_adaf_alpha : float, optional
        Viscosity parameter :math:`\alpha` [dimensionless]. Default 0.3.
    agn_adaf_beta : float, optional
        Gas-to-total pressure ratio :math:`\beta` [dimensionless]. Default 0.5.
    agn_adaf_delta : float, optional
        Electron viscous-heating fraction :math:`\delta` [dimensionless]. Default 0.1.
    agn_log_lbol_shape : float or None, optional
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` that sets the shape (``mdot``); ``None`` uses
        ``agn_log_lbol``. The float32 AGN path passes the true value here and a low reference
        in ``agn_log_lbol``.
    dtype : dtype, optional
        dtype of the wavelength arrays the state will be evaluated on. It selects the float32
        branch of the mdot solve (:math:`\dot m` from the luminosity in :math:`L_\odot`); the
        normalization powers are log-space and dtype-independent (#2783). Default: the
        canonical float dtype of the process (float64 under x64, float32 otherwise).

    Returns
    -------
    AdafState
        Scalar fields (shape ``()``); evaluate with :func:`adaf_spectrum_from_state`.

    Notes
    -----
    **Precision**: the bracket of the x_M solve, the peak luminosity and the normalization
    powers are formed as sums of logs (:func:`_adaf_x_m`,
    :func:`_segment_log_power`), so pure float32 gives the same shape as float64 over the
    declared box to 1e-4 pointwise (#2783; ``tests/physics/agn/test_adaf_float32_shape.py``).

    **JIT/grad/vmap-compatible**: yes; pure ``jnp``, with the ``mdot`` and ``T_e`` solves as
    :func:`jax.lax.while_loop` secants inside :func:`jax.lax.custom_root` (data-dependent trip
    counts, never differentiated through). The runner builds it once per composition and every
    disc evaluation (caller grid, anchors, 5100 A, budget grids) reads the same state, so the
    solves are traced once. ``T_e`` is continuous, with no regime junction, and the derivatives
    through both nested solves are the exact implicit-function derivatives (first and second
    order tested), except at the kinks of the :math:`\theta_e = 1` band of Eq. 28's fit (see
    :func:`_adaf_electron_temperature`). The derivative is discontinuous where the derived
    ``mdot`` meets its clip at
    ``mdot_crit`` (Eq. 52) or at the ``1e-8`` floor.

    Implements the same model as Mahadevan (1997 [1]_); the normalization quadrature is
    tengri's own.

    References
    ----------
    .. [1] Mahadevan, R. 1997, "Scaling Laws for Advection-dominated Flows: Applications to
       Low-Luminosity Galactic Nuclei", ApJ, 477, 585. arXiv:astro-ph/9609107.
       https://doi.org/10.1086/303727
    """
    dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
    _f32 = dtype == jnp.float32
    # Shape luminosity (mdot -> whole spectrum) vs normalization magnitude. They
    # coincide by default (float64). On float32 the AGN component passes the true
    # L_bol for the SHAPE while normalizing MAGNITUDE to a low reference, so the
    # runner's ~1e40 L_lambda arithmetic stays in range (#1206).
    _lbol_shape = agn_log_lbol if agn_log_lbol_shape is None else agn_log_lbol_shape
    m = 10.0**agn_log_mbh
    alpha, beta, delta = agn_adaf_alpha, agn_adaf_beta, agn_adaf_delta

    # mdot from the SHAPE luminosity (float32: pass it in L_sun so the ~1e44 erg/s
    # l_bol_erg never forms).
    mdot = _adaf_mdot_of_log_lbol(_lbol_shape, m, alpha, beta, delta, _f32)
    t_e = _adaf_electron_temperature(m, mdot, alpha, beta, delta)
    x_m = _adaf_x_m(t_e, m, mdot, alpha, beta)
    alpha_c = _adaf_alpha_c(_adaf_tau_es(mdot, alpha), t_e)
    nu_p = _adaf_nu_peak(t_e, x_m, m, mdot, alpha, beta)
    l_nu_p = _adaf_lnu_peak(t_e, nu_p, m)
    l_brems0 = _adaf_lbrems0(t_e, m, mdot, alpha)
    log_l_nu_p = _LN10 * _adaf_log10_lnu_peak(t_e, nu_p, m)
    log_l_brems0 = _adaf_log_lbrems0(t_e, m, mdot, alpha)

    # Low cutoff at the largest-radius synchrotron frequency (nu ~ r^{-5/4});
    # high cutoff at the Comptonization ceiling 3 k T_e / h.
    nu_min = nu_p * (_R_MIN / _R_MAX) ** 1.25
    nu_max_c = 3.0 * _K_BOLTZ * t_e / _H_PLANCK
    log_nu_max_c = _LOG_3K_OVER_H + jnp.log(t_e)
    shape = AdafState(
        t_e,
        alpha_c,
        nu_p,
        nu_min,
        nu_max_c,
        l_nu_p,
        l_brems0,
        log_l_nu_p,
        log_l_brems0,
        log_nu_max_c,
        0.0,
        0.0,
        1.0,
    )

    # Renormalize to the canonical L_bol (magnitude from agn_log_lbol: the
    # reference on the float32 path). The integral uses Gauss-Legendre quadrature
    # on segments defined by the spectrum's break frequencies (nu_p, Compton cutoff),
    # NOT the caller's wavelength array. The spectrum support is [0.02 nu_min, 100 k T_e / h].
    # Grid invariance: a caller grid that stops short of the X-ray (e.g., 912 A at z=0)
    # would renormalize the truncated spectrum by ~1e3 if we integrated over the caller's grid.
    nu_lo = 0.02 * nu_min
    nu_hi = 100.0 * _K_BOLTZ * t_e / _H_PLANCK

    # ln of the shape's power over the model's whole support [erg/s], summed in log space
    # (#2783). The synchrotron + Compton segments are Gauss-Legendre on the spectrum's own
    # breaks (_segment_log_power). The bremsstrahlung power is closed form,
    #   int l_brems0 exp(-h nu / k T_e) d nu = l_brems0 (k T_e / h) [exp(-a) - exp(-b)],
    #   a = h nu_lo / k T_e, b = h nu_hi / k T_e,
    # with exp(-a) - exp(-b) = exp(-a) [1 - exp(-(b - a))] formed as a log1p.
    breaks = (nu_lo, nu_min, nu_p, nu_max_c, nu_hi)
    log_segments = [
        _segment_log_power(lambda nu_: _adaf_log_synch_compton(nu_, shape), lo, hi, dtype)
        for lo, hi in pairwise(breaks)
    ]
    log_brems = _adaf_log_brems_power(log_l_brems0, t_e, nu_lo)
    log_integral = logsumexp(jnp.stack([*log_segments, log_brems]))

    # L_bol power 10**agn_log_lbol * L_sun [erg/s] as a log. The ratio stays linear so a
    # zero agn_lum_ratio gives an exact zero spectrum with the correct derivative (#2783).
    log_numer = agn_log_lbol * _LN10 + _LOG_LSUN_ERG
    return shape._replace(log_numer=log_numer, log_integral=log_integral, lum_ratio=agn_lum_ratio)


def adaf_spectrum_from_state(wavelength: jnp.ndarray, state: AdafState) -> jnp.ndarray:
    r""":func:`adaf_spectrum` on ``wavelength`` from a solved :class:`AdafState`.

    .. math::

        L_\nu(\nu) = r\,\exp\!\left[\ln P_{\rm bol} - \ln P_{\rm shape}
        + \ln\left(S(\nu) + B(\nu)\right)\right]

    with :math:`r` = ``state.lum_ratio`` (linear), :math:`P_{\rm bol}` = ``exp(state.log_numer)``
    and :math:`P_{\rm shape}` =
    ``exp(state.log_integral)`` [erg/s]. The normalization is formed in log space, so no
    erg/s-scale linear product enters the float32 gradient (#2783).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    state : AdafState
        From :func:`adaf_scalar_state`, built with the dtype of ``wavelength``.

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz]. The wavelength grid enters only through :math:`\nu`; the
        normalization does not depend on it.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.

    References
    ----------
    .. [1] Mahadevan, R. 1997, ApJ, 477, 585. arXiv:astro-ph/9609107.
       https://doi.org/10.1086/303727
    """
    nu = _wavelength_to_nu(wavelength)
    log_spec = state.log_numer - state.log_integral + _adaf_log_total(nu, state)
    return state.lum_ratio * jnp.exp(log_spec)


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
    rate ``mdot`` is derived from it (Sect. 5.2, :func:`_adaf_mdot_from_lbol`: the radiated
    power is :math:`Q_{ie} + \delta Q^+`, of which Eq. 49 is the high-``mdot`` limit) and
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
        :math:`\int L_\nu\,d\nu = \mathrm{agn\_frac}\times L_{\rm bol}` over the
        model's whole support (radio to X-ray), whatever ``wavelength`` covers.

    Notes
    -----
    **Normalization**: the power of the shape is computed from the spectrum's own breaks, never
    from ``wavelength``: closed-form bremsstrahlung plus a 30-point Gauss-Legendre rule in
    :math:`\ln\nu` on each of the four segments between :math:`0.02\,\nu_{\min}`,
    :math:`\nu_{\min}`, :math:`\nu_p`, :math:`3kT_e/h` and :math:`100\,kT_e/h` (see
    :func:`adaf_scalar_state` for the equation), then divided out so
    :math:`\int L_\nu d\nu = L_{\rm bol}\times` ``agn_lum_ratio`` over the model's whole
    support. Composed as :func:`adaf_scalar_state` + :func:`adaf_spectrum_from_state`.

    **JIT-compatible**: yes, all-``jnp`` with the ``while_loop`` solves of
    :func:`adaf_scalar_state`. ``T_e`` has no regime junction; the derivative is discontinuous
    where the derived ``mdot`` meets its clip.
    Valid in the ADAF regime ``mdot < mdot_crit ~ 0.28 alpha^2``; the derived
    ``mdot`` is clipped there.

    References
    ----------
    .. [1] R. Mahadevan, ApJ, 477, 585 (1997). arXiv:astro-ph/9609107.
    """
    state = adaf_scalar_state(
        agn_log_lbol,
        agn_lum_ratio,
        agn_log_mbh,
        agn_adaf_alpha,
        agn_adaf_beta,
        agn_adaf_delta,
        agn_log_lbol_shape,
        dtype=wavelength.dtype,
    )
    return adaf_spectrum_from_state(wavelength, state)
