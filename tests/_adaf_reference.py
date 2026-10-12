# SPDX-License-Identifier: BSD-3-Clause
"""Independent numpy / scipy reference for the ADAF electron energy balance (Mahadevan 1997).

Paper: R. Mahadevan, ApJ 477, 585 (1997), arXiv:astro-ph/9609107.

Every function here is computed from the quoted equations with exact K_2 (scipy ``kve``),
CODATA 2018 constants and its own Eq. 20 solve; none of it calls the package. The electron
temperature is the root of the full balance (Eq. 8, Sect. 5.1.1)

    Q_ie + delta Q+ = P_synch + P_Compton + P_brems,

and the radiated power of the flow is, by Sect. 5.2, ``L_ADAF = Q- = Q_e+ = Q_ie + delta Q+``.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import kve

# CODATA 2018 constants (cgs) and the fiducial constants of Mahadevan 1997.
_K_B = 1.380649e-16
_M_E = 9.1093837015e-28
_C = 2.99792458e10
_H = 6.62607015e-27
_C1, _C3, _RMIN, _RMAX = 0.5, 0.3, 3.0, 1000.0


# ── Independent reference (numpy / scipy) ────────────────────────────────


def _theta(t_e):
    return _K_B * t_e / (_M_E * _C**2)


def _ref_g(t_e):
    """g(theta_e), Eq. 11."""
    th = _theta(t_e)
    return (2.0 + 2.0 * th + 1.0 / th) / kve(2, 1.0 / th)


def _ref_f_theta(t_e):
    """Bremsstrahlung factor F(theta_e), Eq. 28."""
    th = _theta(t_e)
    if th < 1.0:
        return 4.0 * (2.0 * th / np.pi**3) ** 0.5 * (1.0 + 1.781 * th**1.34) + 1.73 * th**1.5 * (
            1.0 + 1.1 * th + th**2 - 1.25 * th**2.5
        )
    return (9.0 * th / (2.0 * np.pi)) * (np.log(1.123 * th + 0.48) + 1.5) + 2.30 * th * (
        np.log(1.123 * th) + 1.28
    )


def _ref_tau(mdot, alpha):
    """Electron-scattering optical depth, Eq. 31."""
    return 6.2 / alpha / _C1 * mdot * _RMIN**-0.5


def _ref_x_m(t_e, m, mdot, alpha, beta):
    """Self-absorption parameter x_M, Eq. 20, with n_e and B from Eq. 5 at r_min."""
    n_e = 3.16e19 / alpha / _C1 / m * mdot * _RMIN**-1.5
    s1 = 1.42e9 * alpha**-0.5 * (1.0 - beta) ** 0.5 * _C1**-0.5 * _C3**0.5
    b_field = s1 * m**-0.5 * mdot**0.5 * _RMIN**-1.25
    r_cm = _RMIN * 2.95e5 * m
    th = _theta(t_e)
    ln_c = (
        np.log(2.49e-10 * 4.0 * np.pi * n_e * r_cm / b_field)
        - 3.0 * np.log(th)
        - (np.log(kve(2, 1.0 / th)) - 1.0 / th)
    )

    def f(x):
        h = x ** (-7 / 6) + 0.40 * x ** (-17 / 12) + 0.5316 * x ** (-5 / 3)
        return 1.8899 * x ** (1 / 3) - ln_c - np.log(h)

    if f(1.0) > 0:
        return 1.0
    if f(1e6) < 0:
        return 1e6
    return brentq(f, 1.0, 1e6, xtol=1e-14, rtol=1e-15)


def _ref_parts(t_e, m, mdot, alpha, beta, delta):
    """All terms of the Eq. 8 balance [erg/s], plus the intermediates."""
    x_m = _ref_x_m(t_e, m, mdot, alpha, beta)
    s1 = 1.42e9 * alpha**-0.5 * (1.0 - beta) ** 0.5 * _C1**-0.5 * _C3**0.5
    s2 = 1.19e-13 * x_m
    nu_p = s1 * s2 * m**-0.5 * mdot**0.5 * t_e**2 * _RMIN**-1.25  # Eqs. 21-22
    nu_l_nu_p = s1**3 * s2**3 * 1.05e-24 * _RMIN**-1.75 * m**0.5 * mdot**1.5 * t_e**7  # Eq. 24
    th = _theta(t_e)
    alpha_c = -np.log(_ref_tau(mdot, alpha)) / np.log(1.0 + 4.0 * th + 16.0 * th**2)  # Eq. 34
    c_f = 3.0 * _K_B * t_e / _H / nu_p  # ceiling 3 k T_e / h over nu_p, Eq. 35
    p_compton = nu_l_nu_p / (1.0 - alpha_c) * (c_f ** (1.0 - alpha_c) - 1.0)  # Eq. 35
    q_ie = 1.2e38 * _ref_g(t_e) * _C1**-2 * _C3 * beta * m * mdot**2 / alpha**2 / _RMIN  # Eq. 13
    dq_plus = delta * 9.39e38 * (1.0 - beta) / 1.0 * _C3 * m * mdot / _RMIN  # Eqs. 9, 14
    p_synch = 0.71 * nu_l_nu_p  # Eq. 26
    p_brems = (
        4.78e34 * alpha**-2 * _C1**-2 * np.log(_RMAX / _RMIN) * _ref_f_theta(t_e) * m * mdot**2
    )  # Eq. 29
    return {
        "x_m": x_m,
        "nu_p": nu_p,
        "nu_l_nu_p": nu_l_nu_p,
        "alpha_c": alpha_c,
        "c_f": c_f,
        "q_ie": q_ie,
        "dq_plus": dq_plus,
        "p_synch": p_synch,
        "p_compton": p_compton,
        "p_brems": p_brems,
    }


def _ref_residual(log_te, m, mdot, alpha, beta, delta):
    """ln(Q_ie + delta Q+) - ln(P_synch + P_Compton + P_brems), Eq. 8."""
    p = _ref_parts(np.exp(log_te), m, mdot, alpha, beta, delta)
    return np.log(p["q_ie"] + p["dq_plus"]) - np.log(p["p_synch"] + p["p_compton"] + p["p_brems"])


def _ref_root(m, mdot, alpha, beta, delta):
    """Bracketed brentq root of Eq. 8 in T_e [K], on [1e8, 1e13] K."""
    return float(
        np.exp(
            brentq(
                _ref_residual,
                np.log(1e8),
                np.log(1e13),
                args=(m, mdot, alpha, beta, delta),
                xtol=1e-15,
                rtol=1e-15,
            )
        )
    )


def ref_heating(t_e, m, mdot, alpha, beta, delta):
    """Q_e+ = Q_ie + delta Q+ [erg/s] at ``t_e`` (Eqs. 13, 9, 14)."""
    p = _ref_parts(t_e, m, mdot, alpha, beta, delta)
    return p["q_ie"] + p["dq_plus"]


def ref_luminosity(mdot, m, alpha, beta, delta):
    """L_ADAF = Q_e+ at the balance temperature [erg/s] (Sect. 5.2), and that temperature."""
    t_e = _ref_root(m, mdot, alpha, beta, delta)
    return ref_heating(t_e, m, mdot, alpha, beta, delta), t_e


def ref_mdot_from_luminosity(lum, m, alpha, beta, delta):
    """Invert L_ADAF(mdot) = ``lum`` [erg/s] for mdot with a brentq in ln mdot.

    Returns ``(mdot, T_e, clipped)``; ``clipped`` is True where ``lum`` exceeds the luminosity
    at ``mdot_crit = 0.28 alpha^2`` (Eq. 52) or falls below that at mdot = 1e-8, and the
    returned mdot is then the clip value.
    """
    lo, hi = 1e-8, 0.28 * alpha * alpha

    def f(ln_md):
        return np.log(ref_luminosity(np.exp(ln_md), m, alpha, beta, delta)[0]) - np.log(lum)

    if f(np.log(hi)) <= 0.0:
        return hi, _ref_root(m, hi, alpha, beta, delta), True
    if f(np.log(lo)) >= 0.0:
        return lo, _ref_root(m, lo, alpha, beta, delta), True
    md = float(np.exp(brentq(f, np.log(lo), np.log(hi), xtol=1e-14, rtol=1e-14)))
    return md, _ref_root(m, md, alpha, beta, delta), False
