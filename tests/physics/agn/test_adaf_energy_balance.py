# SPDX-License-Identifier: BSD-3-Clause
"""Electron energy balance of the ADAF (Mahadevan 1997, Sect. 3, 4 and 5.1).

Paper: R. Mahadevan, ApJ 477, 585 (1997), arXiv:astro-ph/9609107.

The electron temperature is the root of the paper's full balance (Eq. 8, Sect. 5.1.1)

    Q_ie + delta Q+ = P_synch + P_Compton + P_brems

with Q_ie from Eq. 13, delta Q+ from Eqs. 9 and 14 (eps' = (1 - beta)/f, f = 1; the
substitution reproduces Eq. 15), P_synch = 0.71 nu_p L_nu_p (Eq. 26), P_Compton from Eq. 35
(slope alpha_c from Eq. 34, optical depth Eq. 31, amplification Eq. 32), and P_brems from
Eq. 29 (F(theta_e) from Eq. 28); nu_p L_nu_p is Eqs. 21-24 and x_M is Eq. 20.

Every reference value below is computed in this file, in numpy/scipy, from the quoted
equations. The functions under test are used only to produce the value being tested (and
the package's own ``x_M`` is never used: the reference solves Eq. 20 itself).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import kve

from tengri.components.agn import adaf as adaf_mod
from tengri.components.agn.adaf import (
    _adaf_balance_residual,
    _adaf_electron_temperature,
    _adaf_mdot_from_lbol,
    _adaf_solve_te_balance,
)
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_paper

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
    """Bracketed brentq root of Eq. 8 in T_e [K]."""
    return float(
        np.exp(
            brentq(
                _ref_residual,
                np.log(1e8),
                np.log(5e11),
                args=(m, mdot, alpha, beta, delta),
                xtol=1e-15,
                rtol=1e-15,
            )
        )
    )


_TE_BATCH = jax.jit(jax.vmap(_adaf_electron_temperature))


def _te(m, mdot, alpha, beta=0.5, delta=0.1):
    return float(_adaf_electron_temperature(m, mdot, alpha, beta, delta))


def _box_points(n, seed=2825):
    """Random points of the declared box, plus the corners that matter (m, mdot, a, b, d)."""
    rng = np.random.default_rng(seed)
    log_m = rng.uniform(6.0, 10.0, n)
    alpha = rng.uniform(0.05, 0.5, n)
    beta = rng.uniform(0.1, 0.9, n)
    delta = 10.0 ** rng.uniform(-3.0, np.log10(0.5), n)
    frac = 10.0 ** rng.uniform(-4.0, 0.0, n)
    pts = np.stack([10.0**log_m, frac * 0.28 * alpha**2, alpha, beta, delta], axis=1)
    corners = [
        (1e6, 0.28 * 0.05**2, 0.05, 0.1, 1e-3),
        (1e10, 0.28 * 0.05**2, 0.05, 0.9, 0.5),
        (1e6, 0.28 * 0.5**2, 0.5, 0.1, 0.5),
        (1e10, 0.28 * 0.5**2, 0.5, 0.9, 1e-3),
        (1e8, 1e-8, 0.3, 0.5, 0.1),
        (1e8, 1e-8, 0.05, 0.9, 0.5),
    ]
    return np.concatenate([pts, np.asarray(corners)], axis=0)


# ── (iv) Agreement with an independent solve of the same balance ─────────


def test_full_balance_root_matches_independent_brentq():
    """T_e equals an independent scipy brentq root of Eq. 8 at 30 points of the box, to 1e-6.

    The reference uses exact K_2 (scipy ``kve``), CODATA constants, its own Eq. 20 solve and
    the exact ceiling 3 k T_e / h. The remaining difference is the package's
    Abramowitz & Stegun K_2 approximation (about 1e-7 in g), which moves the root by less
    than the 1e-6 bound (measured, see the report).
    """
    pts = _box_points(24)
    t_code = np.asarray(_TE_BATCH(*[jnp.asarray(pts[:, i]) for i in range(5)]))
    t_ref = np.array([_ref_root(*p) for p in pts])
    rel = np.abs(t_code / t_ref - 1.0)
    assert rel.max() <= 1e-6, (rel.max(), pts[rel.argmax()])


def test_independent_balance_residual_vanishes_at_returned_root():
    """The reference Eq. 8 residual (exact K_2, CODATA) vanishes at the returned T_e, to 1e-5.

    The bound is set by the package's Abramowitz & Stegun K_2 approximation (about 1e-7 in
    g), which moves the residual by that amount per unit of its O(10) slope.
    """
    pts = _box_points(24, seed=11)
    t_code = np.asarray(_TE_BATCH(*[jnp.asarray(pts[:, i]) for i in range(5)]))
    ref = np.array([_ref_residual(np.log(t), *p) for t, p in zip(t_code, pts)])
    assert np.abs(ref).max() <= 1e-5, np.abs(ref).max()


def test_package_balance_residual_vanishes_at_returned_root():
    """The package residual is zero at the returned T_e to the solver tolerance (1e-10).

    The secant iterates to the float64 floor: the residual is O(10) per unit ln T_e and
    ln Q ~ 90 carries rounding of ~2e-14, so 1e-10 sits three orders above the floor.
    """
    pts = _box_points(24, seed=11)
    args = [jnp.asarray(pts[:, i]) for i in range(5)]
    t_code = _TE_BATCH(*args)
    resid = jax.jit(jax.vmap(_adaf_balance_residual))(jnp.log(t_code), *args)
    assert float(jnp.max(jnp.abs(resid))) <= 1e-10


def test_bracket_encloses_the_root_over_the_box():
    """The residual is positive at 1e8 K and negative at 5e11 K at 600 random box points.

    This is the condition under which the fixed-count bracketed solve returns a root, so the
    box behavior is explicit: inside the declared prior box the balance always has its root
    on [1e8, 5e11] K.
    """
    pts = _box_points(600, seed=7)
    args = [jnp.asarray(pts[:, i]) for i in range(5)]
    f = jax.jit(jax.vmap(_adaf_balance_residual))
    r_lo = f(jnp.full(len(pts), np.log(1e8)), *args)
    r_hi = f(jnp.full(len(pts), np.log(5e11)), *args)
    assert bool(jnp.all(r_lo > 0.0)) and bool(jnp.all(r_hi < 0.0))
    t_e = _TE_BATCH(*args)
    res = jax.jit(jax.vmap(_adaf_balance_residual))(jnp.log(t_e), *args)
    assert float(jnp.max(jnp.abs(res))) <= 1e-10


# ── (a) The converged root does not depend on the starting point ─────────


@pytest.mark.parametrize(
    "point",
    [(1e8, 0.01, 0.3, 0.5, 0.1), (1e6, 0.07, 0.5, 0.5, 0.1), (1e10, 2e-4, 0.1, 0.3, 0.5)],
)
def test_balance_root_independent_of_start(point):
    """The solve converges to the same root from different starting points, to 1e-6 relative.

    Starts 2e9, 3e8 and 2e11 K, all inside the bracket.
    """
    ref = float(_adaf_solve_te_balance(*point, t_start=2e9))
    for start in (3e8, 2e11):
        t_e = float(_adaf_solve_te_balance(*point, t_start=start))
        assert t_e == pytest.approx(ref, rel=1e-6), start


# ── (i) T_e is continuous along fine scans, through the former junction ──


def _ln_te_of_lbol(log_lbol, alpha, delta, log_mbh, beta=0.5):
    m = 10.0**log_mbh
    mdot = _adaf_mdot_from_lbol(10.0**log_lbol * L_SUN, m, alpha, beta, delta)
    return jnp.log(_adaf_electron_temperature(m, mdot, alpha, beta, delta))


def _max_step_over_bound(fun, grid):
    """Largest |d ln T_e| over one grid step, divided by the bound set by the local slope.

    The bound is 2 max(|g_i|, |g_{i+1}|) dx, with g the autodiff slope at the two ends of the
    step: for a function whose slope varies by less than a factor 2 across one step a step
    larger than the bound is a jump.
    """
    val_and_slope = jax.jit(jax.vmap(lambda x: jax.jvp(fun, (x,), (jnp.ones_like(x),))))
    val, slope = val_and_slope(jnp.asarray(grid))
    val, slope = np.asarray(val), np.asarray(slope)
    step = np.abs(np.diff(val))
    bound = 2.0 * np.maximum(np.abs(slope[:-1]), np.abs(slope[1:])) * np.diff(grid) + 1e-9
    return float(np.max(step / bound)), int(np.argmax(step / bound))


# (log M, log L_bol on the former alpha_c = 1 junction at delta = 0.1, alpha = 0.3, beta = 0.5):
# the former solve stepped T_e 8.58e9 -> 3.62e9 K across log L_bol 6.96 -> 6.97 at log M = 7
# (the package default) and 1.00e10 -> 4.00e9 K across 7.75 -> 7.76 at log M = 8.
_JUNCTIONS = [(7.0, 6.965), (8.0, 7.755)]


@pytest.mark.parametrize("delta", [1e-3, 0.1, 0.5])
@pytest.mark.parametrize("log_mbh", [7.0, 8.0])
def test_te_continuous_along_log_lbol(log_mbh, delta):
    """No step of ln T_e along a 0.005 dex scan of log L_bol exceeds twice the local slope.

    alpha = 0.3, beta = 0.5. The scan crosses the alpha_c = 1 junction of the former
    two-branch solve (a step 40 or more times the bound at delta = 0.1) and the mdot_crit
    clip at high luminosity.
    """
    grid = np.arange(2.0, 9.0, 0.005)
    ratio, _ = _max_step_over_bound(lambda x: _ln_te_of_lbol(x, 0.3, delta, log_mbh), grid)
    assert ratio <= 1.0, ratio


@pytest.mark.parametrize("delta", [1e-3, 0.1, 0.5])
@pytest.mark.parametrize(("log_mbh", "log_lbol"), _JUNCTIONS)
def test_te_continuous_along_alpha(log_mbh, log_lbol, delta):
    """No step of ln T_e along a 0.0005 scan of alpha exceeds twice the local slope.

    The luminosity sits on the former junction at alpha = 0.3; the scan runs over the whole
    declared alpha prior [0.05, 0.5].
    """
    grid = np.arange(0.05, 0.5, 0.0005)
    ratio, _ = _max_step_over_bound(lambda a: _ln_te_of_lbol(log_lbol, a, delta, log_mbh), grid)
    assert ratio <= 1.0, ratio


@pytest.mark.parametrize(("log_mbh", "log_lbol"), _JUNCTIONS)
def test_te_continuous_along_delta(log_mbh, log_lbol):
    """No step of ln T_e along a scan of delta exceeds twice the local slope.

    alpha = 0.3, beta = 0.5, the luminosity on the former junction, over the declared delta
    prior [1e-3, 0.5].
    """
    grid = np.arange(1e-3, 0.5, 0.0005)
    ratio, _ = _max_step_over_bound(lambda d: _ln_te_of_lbol(log_lbol, 0.3, d, log_mbh), grid)
    assert ratio <= 1.0, ratio


@pytest.mark.parametrize(("log_mbh", "log_lbol"), _JUNCTIONS)
def test_public_state_te_has_no_step_through_former_junction(log_mbh, log_lbol):
    """``adaf_scalar_state`` T_e at log L_bol -0.005 and +0.005 around the former junction.

    The step must be within twice the autodiff slope times the 0.01 dex width; the former
    solve returned 8.578e9 K and 3.624e9 K across it at the default mass (alpha = 0.3,
    beta = 0.5, delta = 0.1).
    """

    def ln_te(x):
        s = adaf_mod.adaf_scalar_state(x, 1.0, log_mbh, 0.3, 0.5, 0.1)
        return jnp.log(s.t_e)

    g = jax.grad(ln_te)
    a, b = log_lbol - 0.005, log_lbol + 0.005
    step = abs(float(ln_te(b)) - float(ln_te(a)))
    bound = 2.0 * max(abs(float(g(a))), abs(float(g(b)))) * (b - a)
    assert step <= bound, (step, bound)


# ── (iii) The paper's closed forms are limits of the full balance ────────


def _eq40_scalings(t_e, m, mdot, alpha, beta, delta, a_c, coeff):
    """Eq. 40 with x_M from Eq. 20 at ``t_e``, fitting factor ``a_c`` and prefactor ``coeff``."""
    x_m = _ref_x_m(t_e, m, mdot, alpha, beta)
    return (
        coeff
        / a_c ** (1 / 7)
        * (delta * 2000.0) ** (1 / 7)
        * (x_m / 300.0) ** (-3 / 7)
        * (alpha / 0.3) ** (3 / 14)
        * ((1.0 - beta) / 0.5) ** (-1 / 14)
        * (_C1 / 0.5) ** (3 / 14)
        * (_C3 / 0.3) ** (-1 / 14)
        * (_RMIN / 3.0) ** (3 / 28)
        * m ** (1 / 14)
        * mdot ** (-1 / 14)
    )


def _eq40_rederived_coefficient():
    """Prefactor of Eq. 40 at the fiducial point from Eqs. 9, 14 and 24 with A_c = 1 [K].

    delta Q+ = A_c nu_p L_nu_p at delta = 1/2000, x_M = 300, alpha = 0.3, beta = 0.5, c1 = 0.5,
    c3 = 0.3, r_min = 3, m = mdot = 1: T_e^7 = delta Q+ / (s1^3 s2^3 s3 r_min^(-7/4)).
    This is 1.023e9 K. The prefactor printed in Eq. 40 is 1.1e9, 7.5% larger (a factor 1.66 in
    delta Q+ / nu_p L_nu_p); the value from Eqs. 9, 14, 24 is used for the tight check.
    """
    s1 = 1.42e9 * 0.3**-0.5 * 0.5**0.5 * 0.5**-0.5 * 0.3**0.5
    s2 = 1.19e-13 * 300.0
    nu_l = s1**3 * s2**3 * 1.05e-24 * 3.0**-1.75
    d_q = (1.0 / 2000.0) * 9.39e38 * 0.5 * 0.3 / 3.0
    return (d_q / nu_l) ** (1 / 7)


@pytest.mark.parametrize("m", [1e6, 1e8])
@pytest.mark.parametrize("mdot", [1e-6, 1e-5, 3e-5, 9e-5])
def test_eq40_stated_accuracy_in_its_regime(m, mdot):
    """The full balance lies within Eq. 40's stated accuracy where the paper derives Eq. 40.

    Regime of Eq. 40: alpha_c > 1 and mdot < 1e-3 alpha^2 (= 9e-5 at alpha = 0.3), with
    delta = 1e-3 (the paper's fiducial is 1/2000), beta = 0.5. Eq. 40 as printed carries
    A_c^(-1/7) with "A_c^(1/7) varies from 0.95 to 1.4" (Sect. 5.1.2), so
    T_e / T_40(A_c = 1) must lie in [1/1.4, 1/0.95].
    """
    alpha, beta, delta = 0.3, 0.5, 1e-3
    t_e = _te(m, mdot, alpha, beta, delta)
    assert _ref_parts(t_e, m, mdot, alpha, beta, delta)["alpha_c"] > 1.0
    ratio = t_e / _eq40_scalings(t_e, m, mdot, alpha, beta, delta, 1.0, 1.1e9)
    assert 1.0 / 1.4 <= ratio <= 1.0 / 0.95, ratio


@pytest.mark.parametrize("m", [1e6, 1e8])
@pytest.mark.parametrize("mdot", [1e-7, 1e-6, 1e-5])
def test_eq40_limit_to_the_size_of_the_dropped_terms(m, mdot):
    """At low mdot the full balance reproduces Eq. 40 to the size of the terms Eq. 40 drops.

    Eq. 40 solves delta Q+ = A_c nu_p L_nu_p (Eq. 39) with A_c = 0.71 + 1/(alpha_c - 1)
    taken at the solved alpha_c and x_M at the solved T_e. The full balance differs from that
    equation by the terms Eq. 40 drops: Q_ie, P_brems, and the Compton power beyond
    nu_p L_nu_p / (alpha_c - 1) (Eq. 35 against Eq. 39). At fixed x_M and alpha_c,
    nu_p L_nu_p goes as T_e^7 (Eq. 24), so T_40 / T_e lies within (1 -/+ eps)^(1/7) of 1, with
    eps the dropped power over A_c nu_p L_nu_p, computed here from the reference terms. The
    prefactor is the one re-derived from Eqs. 9, 14 and 24 (see
    ``_eq40_rederived_coefficient``); the printed 1.1e9 is checked separately above.
    """
    alpha, beta, delta = 0.3, 0.5, 1e-3
    t_e = _te(m, mdot, alpha, beta, delta)
    p = _ref_parts(t_e, m, mdot, alpha, beta, delta)
    assert p["alpha_c"] > 1.0
    a_c = 0.71 + 1.0 / (p["alpha_c"] - 1.0)
    dropped = (
        p["q_ie"] + p["p_brems"] + abs(p["p_compton"] - p["nu_l_nu_p"] / (p["alpha_c"] - 1.0))
    ) / (a_c * p["nu_l_nu_p"])
    assert dropped < 0.1, dropped
    t_40 = _eq40_scalings(t_e, m, mdot, alpha, beta, delta, a_c, _eq40_rederived_coefficient())
    bound = max((1.0 + dropped) ** (1 / 7) - 1.0, 1.0 - (1.0 - dropped) ** (1 / 7))
    assert abs(t_40 / t_e - 1.0) <= bound + 1e-6


@pytest.mark.parametrize("frac", [0.3, 0.5, 1.0])
def test_compton_only_limit_where_delta_q_plus_is_small(frac):
    """The Compton-only balance Q_ie = P_Compton is the limit where delta Q+ << Q_ie.

    delta = 1e-3, M = 1e8, alpha = 0.3, beta = 0.5, mdot = frac * mdot_crit. The shift of the
    root from the neglected terms is bounded by the residual of the full balance at the
    Compton-only root divided by the slope d R / d ln T_e: ln T_full - ln T_c =
    R_full(T_c) / |R'|, with R_full(T_c) = ln(1 + delta Q+/Q_ie) - ln(1 + (P_synch +
    P_brems)/P_Compton) and the slope taken from the reference at T_c (the residual is
    monotone and nearly linear in ln T_e across the shift, which is a few percent at most).
    """
    m, alpha, beta, delta = 1e8, 0.3, 0.5, 1e-3
    mdot = frac * 0.28 * alpha**2

    def compton_only(log_te):
        p = _ref_parts(np.exp(log_te), m, mdot, alpha, beta, 0.0)
        return np.log(p["q_ie"]) - np.log(p["p_compton"])

    t_c = float(np.exp(brentq(compton_only, np.log(1e8), np.log(5e11), xtol=1e-15)))
    p = _ref_parts(t_c, m, mdot, alpha, beta, delta)
    eps_heat = p["dq_plus"] / p["q_ie"]
    eps_cool = (p["p_synch"] + p["p_brems"]) / p["p_compton"]
    assert eps_heat < 0.1 and eps_cool < 0.1, (eps_heat, eps_cool)
    h = 1e-4
    slope = abs(compton_only(np.log(t_c) + h) - compton_only(np.log(t_c) - h)) / (2 * h)
    shift = abs(np.log1p(eps_heat) - np.log1p(eps_cool)) / slope
    assert abs(np.log(_te(m, mdot, alpha, beta, delta) / t_c)) <= 1.1 * shift + 1e-6


@pytest.mark.parametrize("frac", [0.3, 0.5, 0.8])
def test_alpha_c_satisfies_paper_eq45_where_compton_cooling_dominates(frac):
    """The solved state satisfies Eq. 45 of the paper (the Q_ie = P_Compton form).

    1 - alpha_c = log[(Q_ie / nu_p L_nu_p)(1 - alpha_c) + 1] / log C_F.

    Eq. 45 is Q_ie = P_Compton rewritten with Eq. 41 (Sect. 5.1.2), the equation the paper
    iterates for "accurate determination of both T_e and alpha_c", valid where delta Q+ << Q_ie
    and synchrotron plus bremsstrahlung cooling are negligible. alpha_c is Eq. 34 at the
    returned T_e, and Q_ie, nu_p L_nu_p and C_F are the reference values at that T_e. The
    right-hand side uses Q_ie alone, as the paper does. The difference from the exact
    relation (P_Compton in place of Q_ie) is the effect of the dropped delta Q+, P_synch and
    P_brems, computed here term by term; the test asserts the solved state differs from
    Eq. 45 by no more than that, and by less than 5e-3 in absolute terms. A solve that
    returns a temperature unrelated to the balance (the identity iteration of Eq. 43, which
    returns the initial guess times 1.03) violates it by order 0.1.
    """
    m, alpha, beta, delta = 1e8, 0.3, 0.5, 1e-3
    mdot = frac * 0.28 * alpha**2
    t_e = _te(m, mdot, alpha, beta, delta)
    p = _ref_parts(t_e, m, mdot, alpha, beta, delta)
    a_c, nl = p["alpha_c"], p["nu_l_nu_p"]
    lhs = 1.0 - a_c
    rhs = np.log(p["q_ie"] / nl * (1.0 - a_c) + 1.0) / np.log(p["c_f"])
    # Exact size of the dropped terms: with P_Compton = Q_ie + delta Q+ - P_synch - P_brems,
    # Eq. 41 gives 1 - alpha_c = ln[(P_Compton / nu_p L_nu_p)(1 - alpha_c) + 1] / ln C_F.
    w = (1.0 - a_c) / nl
    q_eff = p["q_ie"] + p["dq_plus"] - p["p_synch"] - p["p_brems"]
    dropped = abs(np.log((1.0 + w * p["q_ie"]) / (1.0 + w * q_eff))) / np.log(p["c_f"])
    assert a_c < 1.0
    assert abs(lhs - rhs) <= dropped * (1.0 + 1e-3) + 1e-9, (lhs, rhs, dropped)
    assert abs(lhs - rhs) < 5e-3, (lhs, rhs)


# ── (e) High-mdot limit matches the paper's stated temperature ───────────


@pytest.mark.parametrize("m", [1e6, 1e8, 1e10])
def test_high_mdot_limit_within_paper_eq44_window(m):
    """At mdot ~ 1e-2 (alpha = 0.3, delta = 1/2000) T_e lies in Eq. 44's window 2.15e9 - 3.12e9 K.

    Eq. 44 (Mahadevan 1997, p. 17) states "2.15e9 <= T_e <= 3.12e9 whether the systems are
    1 M_sun or 1e9 M_sun black holes" for mdot ~ 1e-2, derived from the Q_ie = P_Compton
    balance of Sect. 5.1.2, which the full balance reduces to at the paper's delta ~ 1/2000.
    The accuracy claimed is that window. Sect. 5.1.1 also quotes T_e ~ 2.0e9 K at high mdot
    as an approximate constant.

    Measured inconsistency: Eq. 44's printed bounds do not follow from Eq. 43 with the tau_es
    of Eq. 31. Evaluated by hand, Eq. 43 at mdot = 0.01, alpha = 0.3 gives about 2.0e9 K at
    alpha_c = 1 and 5.4e9 K at alpha_c = 0.5, where Eq. 44 gives 2.15e9 and 3.12e9; the
    printed bounds correspond to tau_es about twice the Eq. 31 value. The test keeps the
    paper's stated window.
    """
    t_e = _te(m, 0.01, 0.3, 0.5, 1.0 / 2000.0)
    assert 2.15e9 <= t_e <= 3.12e9


# ── Behavior at tau_es >= 1 ──────────────────────────────────────────────


def test_te_smooth_through_tau_es_equal_one():
    """On mdot = mdot_crit, T_e is smooth in alpha through tau_es = 1 (alpha ~ 0.4988).

    The paper does not treat tau_es >= 1 (alpha_c <= 0); the balance evaluates Eq. 35 as
    written there. Steps of ln T_e along a 0.0005 scan of alpha stay within twice the local
    slope, so there is no jump at the box edge.
    """
    grid = np.arange(0.45, 0.5 + 1e-12, 0.0005)

    def ln_te(a):
        return jnp.log(_adaf_electron_temperature(1e8, 0.28 * a * a, a, 0.5, 0.1))

    ratio, _ = _max_step_over_bound(ln_te, grid)
    assert ratio <= 1.0, ratio


# ── Gradients and Hessians ───────────────────────────────────────────────


@pytest.mark.gradient
@pytest.mark.parametrize("alpha", [0.05, 0.5])
@pytest.mark.parametrize("beta", [0.1, 0.9])
@pytest.mark.parametrize("delta", [1e-3, 0.5])
@pytest.mark.parametrize("frac", [1e-3, 1.0])
def test_gradients_finite_at_box_corners_float64(alpha, beta, delta, frac):
    """d ln T_e / d(m, mdot, alpha, beta, delta) is finite and nonzero at the box corners.

    ``frac`` sets mdot as a fraction of mdot_crit = 0.28 alpha^2, so the corners include the
    critical-rate edge, where tau_es approaches 1.
    """
    mdot = frac * 0.28 * alpha * alpha

    def ln_t(p):
        return jnp.log(_adaf_electron_temperature(p[0], p[1], p[2], p[3], p[4]))

    grads = jax.grad(ln_t)(jnp.asarray([1e8, mdot, alpha, beta, delta]))
    assert bool(jnp.all(jnp.isfinite(grads))), f"non-finite T_e gradient {grads}"
    assert bool(jnp.any(grads != 0.0)), f"identically zero T_e gradient {grads}"


@pytest.mark.gradient
@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_gradients_finite_at_box_corners_float32(alpha):
    """The same finiteness holds on the pure float32 path (JAX_ENABLE_X64 off)."""
    mdot = 0.28 * alpha * alpha

    def ln_t(p):
        return jnp.log(_adaf_electron_temperature(p[0], p[1], p[2], p[3], 0.1))

    with jax.enable_x64(False):
        p = jnp.asarray([1e8, mdot, alpha, 0.5], dtype=jnp.float32)
        grads = jax.grad(ln_t)(p)
        assert bool(jnp.all(jnp.isfinite(grads))), f"non-finite T_e gradient {grads}"
        assert bool(jnp.any(grads != 0.0)), f"identically zero T_e gradient {grads}"


@pytest.mark.gradient
@pytest.mark.regression_bug
def test_float32_gradient_of_balance_temperature_matches_float64_fd():
    """Float32 autodiff of ln T_e agrees with a float64 central difference, all four inputs.

    At (m=1e8, mdot=0.0187, alpha=0.3, beta=0.5) the float32 beta derivative was 46% low
    (0.173 against 0.323). The cause was ``ln(4.8e37 beta)`` differentiated as a whole: its
    cotangent, of order 1e-39, is subnormal in float32 and was flushed to zero. Tolerance 1e-3
    relative to the largest component, as in the float32 shape test.
    """
    p64 = np.array([1e8, 0.0187, 0.3, 0.5])

    def log_te(p):
        return jnp.log(_adaf_electron_temperature(p[0], p[1], p[2], p[3], 0.1))

    fd = np.zeros(4)
    for i in range(4):
        h = 1e-6 * p64[i]
        pp, pm = p64.copy(), p64.copy()
        pp[i] += h
        pm[i] -= h
        fd[i] = (float(log_te(jnp.asarray(pp))) - float(log_te(jnp.asarray(pm)))) / (2 * h)
    with jax.enable_x64(False):
        g32 = np.asarray(jax.grad(log_te)(jnp.asarray(p64, dtype=jnp.float32))).astype(np.float64)
    scale = float(np.max(np.abs(fd)))
    assert np.max(np.abs(g32 - fd)) <= 1e-3 * scale, f"f32 AD {g32} vs f64 FD {fd}"


def _hessian_vs_fd_of_grad(fun, x, steps):
    """Return (autodiff Hessian, central FD of the autodiff gradient, max |H|)."""
    grad = jax.grad(fun)
    x = jnp.asarray(x, dtype=jnp.float64)
    h_ad = np.asarray(jax.hessian(fun)(x))
    h_fd = np.zeros_like(h_ad)
    for j, step in enumerate(steps):
        e = jnp.zeros_like(x).at[j].set(step)
        h_fd[:, j] = np.asarray(grad(x + e) - grad(x - e)) / (2.0 * step)
    return h_ad, h_fd, float(np.max(np.abs(h_ad)))


# (m, mdot, alpha, beta, delta): mid Compton regime, the tau_es ~ 1 corner, the former junction.
_HESSIAN_POINTS = [
    (1e8, 0.0126, 0.3, 0.5, 0.1),
    (1e6, 0.07, 0.5, 0.5, 0.1),
    (1e8, 3.82e-3, 0.3, 0.5, 0.1),
]


@pytest.mark.gradient
@pytest.mark.parametrize("point", _HESSIAN_POINTS)
def test_hessian_of_te_matches_finite_difference_of_gradient(point):
    """jax.hessian of ln T_e w.r.t. (ln m, ln mdot, alpha, delta) matches a central difference
    of the (exact) gradient to 1e-5 of the largest entry.

    The implicit-function derivative is exact at every order only if the second-order terms
    (R_uu, R_up) are kept; a solve whose tangent step stops the slope gradient returns
    first derivatives that are right and second derivatives that are wrong by O(1)
    (676 against a largest entry of 715 at the (1e6, 0.07, 0.5) corner).
    """
    m, mdot, alpha, beta, delta = point

    def fun(x):
        return jnp.log(_adaf_electron_temperature(jnp.exp(x[0]), jnp.exp(x[1]), x[2], beta, x[3]))

    x0 = [np.log(m), np.log(mdot), alpha, delta]
    h_ad, h_fd, scale = _hessian_vs_fd_of_grad(fun, x0, [1e-5, 1e-5, 1e-6, 1e-6])
    assert np.max(np.abs(h_ad - h_fd)) <= 1e-5 * scale, (np.max(np.abs(h_ad - h_fd)), scale)


@pytest.mark.gradient
@pytest.mark.parametrize(
    "point", [(7.5, 8.0, 0.3, 0.1), (6.965, 7.0, 0.3, 0.1), (8.2, 6.0, 0.4, 0.3)]
)
def test_hessian_of_te_through_lbol_matches_finite_difference_of_gradient(point):
    """The same check on the public parameters (log L_bol, log M, alpha, delta).

    ``mdot`` comes from the L_bol inversion (Eq. 49), which calls the solve three times, so
    this exercises the nested implicit derivatives.
    """

    def fun(x):
        return _ln_te_of_lbol(x[0], x[2], x[3], x[1])

    h_ad, h_fd, scale = _hessian_vs_fd_of_grad(fun, point, [1e-5, 1e-5, 1e-6, 1e-6])
    assert np.max(np.abs(h_ad - h_fd)) <= 1e-5 * scale, (np.max(np.abs(h_ad - h_fd)), scale)


@pytest.mark.gradient
@pytest.mark.parametrize("point", [(7.5, 8.0, 0.3, 0.1), (8.2, 6.0, 0.4, 0.3)])
def test_hessian_of_summed_spectrum_matches_finite_difference_of_gradient(point):
    """jax.hessian of ln(sum L_nu) w.r.t. (log L_bol, log M, alpha, delta) matches a central
    difference of its gradient to 1e-5 of the largest entry.

    The summed spectrum on a 40-point rest-frame grid is the quantity a Laplace fit reads:
    it carries T_e through nu_p, nu_min, the Compton ceiling and the bremsstrahlung level.
    """
    wave = jnp.asarray(np.logspace(-2, 8, 40))

    def fun(x):
        s = adaf_mod.adaf_scalar_state(x[0], 1.0, x[1], x[2], 0.5, x[3])
        return jnp.log(jnp.sum(adaf_mod.adaf_spectrum_from_state(wave, s)))

    h_ad, h_fd, scale = _hessian_vs_fd_of_grad(fun, point, [1e-4, 1e-4, 1e-5, 1e-5])
    assert np.max(np.abs(h_ad - h_fd)) <= 1e-4 * scale, (np.max(np.abs(h_ad - h_fd)), scale)


# ── Bremsstrahlung closed form ───────────────────────────────────────────


def test_brems_power_log_form_equals_linear_form_in_float64():
    """The log-space bremsstrahlung power equals the linear closed form in float64.

    int_{nu_lo}^{nu_hi} L0 exp(-h nu / k T) d nu = L0 (k T / h) [exp(-a) - exp(-b)], with
    a = h nu_lo / k T and b = h nu_hi / k T = 100. The float32-safe form
    ln L0 + ln t - a + log1p(-exp(-(100 - a))) is an algebraic identity of it.
    """
    for t_e, log_l0, nu_lo_over_t in [(2e9, 50.0, 1e-12), (4e8, 55.0, 3e-9), (1e10, 48.0, 1e-5)]:
        t_scale = _K_B * t_e / _H
        nu_lo = nu_lo_over_t * t_scale
        linear = np.log(np.exp(log_l0) * t_scale * (np.exp(-nu_lo / t_scale) - np.exp(-100.0)))
        got = float(
            adaf_mod._adaf_log_brems_power(
                jnp.asarray(log_l0), jnp.asarray(t_e), jnp.asarray(nu_lo)
            )
        )
        assert got == pytest.approx(linear, rel=1e-12)
