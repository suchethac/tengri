# SPDX-License-Identifier: BSD-3-Clause
"""Electron energy balance of the ADAF Compton branch (Mahadevan 1997, Sect. 4-5).

Paper: R. Mahadevan, ApJ 477, 585 (1997), arXiv:astro-ph/9609107.

For ``alpha_c < 1`` the paper takes synchrotron and bremsstrahlung cooling as
negligible and heating as Coulomb ion-electron heating, so the electron
temperature is the root of

    Q_ie(T_e) = P_Compton(T_e)                                  (Sect. 5.1.2)

with Q_ie from Eq. 13, P_Compton from Eq. 35, the Compton slope alpha_c from
Eq. 34 (optical depth Eq. 31, amplification Eq. 32), and nu_p L_nu_p from
Eqs. 21-23. Eq. 43 is the closed-form inverse of Eq. 34 for a given alpha_c and
is an approximation to this root, not an independent equation.

The reference values below are computed in this file from the paper's equations,
with ``x_M`` (Eq. 20) taken from the solver that is tested on its own. They do
not call the functions under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import kve

from tengri.components.agn.adaf import (
    _adaf_balance_residual,
    _adaf_electron_temperature,
    _adaf_x_m,
)

pytestmark = pytest.mark.regression_paper

# Fiducial constants of Mahadevan 1997 (Sect. 2, Eqs. 5, 31 and 13).
_C1 = 0.5
_C3 = 0.3
_R = 3.0
_BOLTZ = 1.380649e-16  # erg/K
_M_E = 9.1093837e-28  # g
_C_LIGHT = 2.99792458e10  # cm/s
_THETA_PER_K = _BOLTZ / (_M_E * _C_LIGHT**2)


def _ref_g(t_e):
    """g(theta_e), Eq. 11."""
    th = _THETA_PER_K * t_e
    return (2.0 + 2.0 * th + 1.0 / th) / kve(2, 1.0 / th)


def _ref_tau(mdot, alpha):
    """Electron-scattering optical depth, Eq. 31."""
    return 6.2 * alpha**-1.0 * _C1**-1.0 * mdot * _R**-0.5


def _ref_alpha_c(t_e, mdot, alpha):
    """Compton slope, Eq. 34 with the amplification factor of Eq. 32."""
    th = _THETA_PER_K * t_e
    return -np.log(_ref_tau(mdot, alpha)) / np.log(1.0 + 4.0 * th + 16.0 * th**2)


def _ref_q_ie(t_e, m, mdot, alpha, beta):
    """Coulomb ion-electron heating, Eq. 13 (erg/s)."""
    return 1.2e38 * _C1**-2.0 * _C3 * beta * _R**-1.0 * alpha**-2.0 * m * mdot**2 * _ref_g(t_e)


def _ref_p_compton(t_e, m, mdot, alpha, beta):
    """Total Compton power, Eq. 35 (erg/s), with nu_p L_nu_p from Eqs. 21-23."""
    x_m = float(_adaf_x_m(jnp.asarray(t_e), m, mdot, alpha, beta))
    s1 = 1.42e9 * alpha**-0.5 * (1.0 - beta) ** 0.5 * _C1**-0.5 * _C3**0.5
    nu_p = s1 * 1.19e-13 * x_m * m**-0.5 * mdot**0.5 * t_e**2 * _R**-1.25  # Eqs. 21-22
    l_nu_p = 1.05e-24 * t_e * nu_p**2 * m**2 * _R**2  # Eq. 23
    a_c = _ref_alpha_c(t_e, mdot, alpha)
    x_ratio = 6.2e7 * (t_e / 1e9) / (nu_p / 1e12)
    return nu_p * l_nu_p / (1.0 - a_c) * (x_ratio ** (1.0 - a_c) - 1.0)


def _ref_residual(t_e, m, mdot, alpha, beta):
    """log Q_ie - log P_Compton; zero at the energy balance."""
    return np.log(_ref_q_ie(t_e, m, mdot, alpha, beta)) - np.log(
        _ref_p_compton(t_e, m, mdot, alpha, beta)
    )


def _ref_root(m, mdot, alpha, beta):
    """Bracketed reference root of the energy balance in T_e [K]."""
    return brentq(
        lambda t: _ref_residual(t, m, mdot, alpha, beta),
        1e8,
        5e11,
        xtol=1e-6,
        rtol=1e-14,
    )


def _te(m, mdot, alpha, beta=0.5, delta=1e-3):
    return float(_adaf_electron_temperature(m, mdot, alpha, beta, delta))


# ── (a) The converged root does not depend on the starting point ─────────


def test_balance_root_independent_of_bracket():
    """The bisection-Newton solve converges to the same root from different brackets.

    Two brackets, both containing the Compton-branch root for (m=1e8, mdot=0.01,
    alpha=0.3). The solved T_e must agree to 1e-6 relative. The solver takes no
    initial guess; the bracket is the only start-dependent input.
    """
    from tengri.components.agn.adaf import _adaf_solve_te_balance

    wide = float(_adaf_solve_te_balance(1e8, 0.01, 0.3, 0.5, t_lo=1e8, t_hi=5e11))
    narrow = float(_adaf_solve_te_balance(1e8, 0.01, 0.3, 0.5, t_lo=5e8, t_hi=1e10))
    assert narrow == pytest.approx(wide, rel=1e-6)


# ── (b) The solved state satisfies the energy balance ────────────────────


@pytest.mark.parametrize("mdot", [0.005, 0.01, 0.015, 0.02])
def test_energy_balance_residual_vanishes(mdot):
    """Q_ie(T_e) - P_Compton(T_e) = 0 at the returned T_e, in log space, to solver precision.

    Evaluated with the solver's own residual, so the bound is the solver floor. The
    bisection brackets ln T_e to ln(5e3) * 2**-40 ~ 8e-12, the Newton step removes the
    linear part of that error, and the residual is O(10) per unit ln T_e, so the
    leftover is quadratic in the step plus float64 rounding of ln Q ~ 99 (~2e-14).
    Tolerance 1e-10 sits three orders above that floor.
    """
    t_e = _te(1e8, mdot, 0.3)
    res = float(_adaf_balance_residual(jnp.log(t_e), 1e8, mdot, 0.3, 0.5))
    assert abs(res) <= 1e-10


@pytest.mark.parametrize("mdot", [0.005, 0.01, 0.02])
def test_residual_vanishes_against_exact_bessel_reference(mdot):
    """The same balance evaluated with exact K_2 (scipy) agrees to the K_2 approximation.

    The reference uses ``scipy.special.kve``. The package evaluates K_2 by the
    Abramowitz & Stegun rational forms, which differ from the exact g(theta_e) by
    about 2e-6 relative at T_e ~ 2e9 K (measured). That shifts the balance root by
    ~1e-6 in ln T_e, and the reference residual at the package root is ~1.6e-5 in log.
    Tolerance 1e-4 is set from that measured Bessel mismatch with margin.
    """
    t_e = _te(1e8, mdot, 0.3)
    assert abs(_ref_residual(t_e, 1e8, mdot, 0.3, 0.5)) <= 1e-4


def test_public_root_matches_reference_root():
    """The returned T_e equals the exact-Bessel reference root to 1e-5 relative.

    The 1e-5 bound is the A&S K_2 mismatch of the package (see the residual test above),
    which moves the root by ~1.2e-6 relative at this point (measured), with margin.
    """
    t_e = _te(1e8, 0.01, 0.3)
    assert t_e == pytest.approx(_ref_root(1e8, 0.01, 0.3, 0.5), rel=1e-5)


# ── (c) alpha_c lies in the paper's range where the regime applies ───────


@pytest.mark.parametrize("mdot", [0.005, 0.01, 0.015, 0.02])
def test_alpha_c_in_paper_range_at_high_mdot(mdot):
    """0.5 <= alpha_c <= 1 at high accretion rate, where the Compton branch applies.

    The paper states "0.5 <= alpha_c <= 1.0 for all cases of interest" (Sect.
    5.1.2, below Eq. 43). Measured at (m=1e8, alpha=0.3, beta=0.5) the balance
    root gives alpha_c in roughly [0.56, 0.94] over this mdot range, while the
    current code gives 0.09-0.5 outside it.
    """
    a_c = _ref_alpha_c(_te(1e8, mdot, 0.3), mdot, 0.3)
    assert 0.5 <= a_c <= 1.0


# ── (d) T_e is continuous across the declared alpha box at mdot_crit ─────


def test_no_jump_across_box_at_critical_mdot():
    """T_e varies smoothly in alpha on mdot = mdot_crit = 0.28 alpha^2.

    The old branch dropped from 2.06e9 K at alpha = 0.498 to 1e8 K at alpha = 0.499,
    a 20x jump (|d ln T_e| ~ 3) inside Uniform(0.05, 0.5). The balance root falls
    smoothly across the tau_es = 1 crossing near alpha = 0.4988, with the steepest
    step at the box edge (|d ln T_e| ~ 0.05 per 0.0025 in alpha, measured). The bound
    0.1 per 0.0025 is set between that smooth profile and the 3.0 of the jump.
    """
    alphas = np.arange(0.45, 0.5 + 1e-12, 0.0025)
    t_e = np.array([_te(1e8, 0.28 * a * a, a) for a in alphas])
    steps = np.abs(np.diff(np.log(t_e)))
    assert steps.max() < 0.1


# ── (e) High-mdot limit matches the paper's stated temperature ───────────


@pytest.mark.parametrize("m", [1e6, 1e8, 1e10])
def test_high_mdot_limit_within_paper_eq44_window(m):
    """At mdot ~ 1e-2 (alpha = 0.3) T_e lies in Eq. 44's window 2.15e9 - 3.12e9 K.

    Eq. 44 (Mahadevan 1997, p. 17) states "2.15e9 <= T_e <= 3.12e9 whether the
    systems are 1 M_sun or 1e9 M_sun black holes" for mdot ~ 1e-2. The
    accuracy claimed is that window. Sect. 5.1.1 also quotes T_e ~ 2.0e9 K at high
    mdot as an approximate constant. The mass independence stated in Eq. 44 holds
    to about 15% over the mass range here.
    """
    t_e = _te(m, 0.01, 0.3)
    assert 2.15e9 <= t_e <= 3.12e9


# ── (f) Gradients are finite at the box corners ──────────────────────────


@pytest.mark.gradient
@pytest.mark.parametrize("alpha", [0.05, 0.5])
@pytest.mark.parametrize("beta", [0.05, 0.95])
@pytest.mark.parametrize("frac", [0.5, 1.0])
def test_gradients_finite_at_box_corners_float64(alpha, beta, frac):
    """d T_e / d(mdot, alpha, beta, m) is finite at the corners of the declared box.

    ``frac`` sets mdot as a fraction of mdot_crit = 0.28 alpha^2, so the corners
    include the critical-rate edge, where tau_es approaches 1.
    """
    mdot_crit = 0.28 * alpha * alpha
    mdot = frac * mdot_crit

    def t_of(params):
        m_, md, a_, b_ = params
        return _adaf_electron_temperature(m_, md, a_, b_, 1e-3)

    grads = jax.grad(lambda p: jnp.log(t_of(p)))(jnp.asarray([1e8, mdot, alpha, beta]))
    assert bool(jnp.all(jnp.isfinite(grads)))


@pytest.mark.gradient
@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_gradients_finite_at_box_corners_float32(alpha):
    """The same finiteness holds on the pure float32 path (JAX_ENABLE_X64 off)."""
    mdot = 0.28 * alpha * alpha

    def t_of(params):
        m_, md, a_, b_ = params
        return _adaf_electron_temperature(m_, md, a_, b_, 1e-3)

    with jax.enable_x64(False):
        p = jnp.asarray([1e8, mdot, alpha, 0.5], dtype=jnp.float32)
        grads = jax.grad(lambda q: jnp.log(t_of(q)))(p)
        assert bool(jnp.all(jnp.isfinite(grads)))
