# SPDX-License-Identifier: BSD-3-Clause
"""Accretion rate for a given ADAF luminosity (Mahadevan 1997, Sect. 5.2).

Paper: R. Mahadevan, ApJ 477, 585 (1997), arXiv:astro-ph/9609107.

Sect. 5.2 defines the total luminosity as the electron heating,

    L_ADAF = Q- = Q_e+ = Q_ie + delta Q+                       (Eqs. 8, 13, 14),

evaluated at the balance temperature of Sect. 5.1.1. Eq. 49 is its limit for
mdot > 1e-3 alpha^2 (Q_ie dominates, which "also gives the luminosity for the case
delta = 0") and Eq. 50 its limit for mdot < 1e-3 alpha^2 (delta Q+ dominates). Inverting
L_ADAF for mdot therefore solves the full expression; the tests below hold it to the
independent numpy / scipy solver in ``tests/_adaf_reference.py``, which uses exact K_2, CODATA
constants and its own Eq. 20 solve, and to the two limits with tolerances set by the neglected
term.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests import _adaf_reference as ref

from tengri.components.agn import adaf as adaf_mod
from tengri.components.agn.adaf import _adaf_electron_temperature, _adaf_mdot_from_lbol
from tengri.utils.physics_constants import L_SUN

_MDOT = jax.jit(jax.vmap(_adaf_mdot_from_lbol))
_TE = jax.jit(jax.vmap(_adaf_electron_temperature))

# 1.2e38 c1^-2 c3 r_min^-1 (Eq. 13 / Eq. 49) and 9.39e38 c3 r_min^-1 (Eqs. 9, 14), f = 1.
_COEF_QIE = 1.2e38 * ref._C1**-2 * ref._C3 / ref._RMIN
_COEF_DQ = 9.39e38 * ref._C3 / ref._RMIN


def _box_draws(n_candidates, seed):
    """Random draws of (L_bol [erg/s], m, alpha, beta, delta) over the declared box."""
    rng = np.random.default_rng(seed)
    log_l = rng.uniform(8.0, 14.0, n_candidates)
    log_m = rng.uniform(6.0, 10.0, n_candidates)
    alpha = rng.uniform(0.05, 0.5, n_candidates)
    beta = rng.uniform(0.1, 0.9, n_candidates)
    delta = 10.0 ** rng.uniform(-3.0, np.log10(0.5), n_candidates)
    return np.stack([10.0**log_l * L_SUN, 10.0**log_m, alpha, beta, delta], axis=1)


def _unclipped_draws(n_wanted, seed):
    """The first ``n_wanted`` box draws whose mdot is strictly inside (1e-8, mdot_crit)."""
    pts = _box_draws(8 * n_wanted, seed)
    mdot = np.asarray(_MDOT(*[jnp.asarray(pts[:, i]) for i in range(5)]))
    crit = 0.28 * pts[:, 2] ** 2
    inside = (mdot < crit * (1.0 - 1e-6)) & (mdot > 1e-8 * (1.0 + 1e-6))
    assert inside.sum() >= n_wanted, (inside.sum(), n_wanted)
    return pts[inside][:n_wanted], mdot[inside][:n_wanted]


# ── (i) The returned state radiates the requested luminosity ─────────────


@pytest.mark.conservation
@pytest.mark.regression_paper
def test_returned_state_radiates_the_requested_luminosity():
    """(Q_ie + delta Q+) / L = 1 at the returned (mdot, T_e) over 240 unclipped box draws.

    Q_ie (Eq. 13) and delta Q+ (Eqs. 9, 14) are evaluated by the independent reference at
    the package's T_e. The bound 2e-6 is the package's Abramowitz and Stegun K_2 (about 1e-7
    in g) plus the solver tolerance.
    """
    pts, mdot = _unclipped_draws(240, seed=2825)
    t_e = np.asarray(_TE(*[jnp.asarray(c) for c in (pts[:, 1], mdot, *pts[:, 2:].T)]))
    ratio = np.array(
        [
            ref.ref_heating(t, m, md, a, b, d) / lum
            for t, md, (lum, m, a, b, d) in zip(t_e, mdot, pts)
        ]
    )
    assert np.max(np.abs(ratio - 1.0)) <= 2e-6, (np.max(np.abs(ratio - 1.0)), np.median(ratio))


@pytest.mark.conservation
@pytest.mark.regression_paper
def test_returned_state_cooling_equals_the_requested_luminosity():
    """The reference cooling P_synch + P_Compton + P_brems over L is 1 at the returned state.

    The balance (Eq. 8) makes the cooling equal the heating at the returned T_e, so with the
    heating equal to L (above) the cooling radiates L: Q- = L_ADAF (Sect. 5.2). Forty draws,
    tolerance 1e-5 (the reference solves Eq. 20 itself).
    """
    pts, mdot = _unclipped_draws(40, seed=77)
    t_e = np.asarray(_TE(*[jnp.asarray(c) for c in (pts[:, 1], mdot, *pts[:, 2:].T)]))
    out = []
    for t, md, (lum, m, a, b, d) in zip(t_e, mdot, pts):
        p = ref._ref_parts(t, m, md, a, b, d)
        out.append((p["p_synch"] + p["p_compton"] + p["p_brems"]) / lum)
    assert np.max(np.abs(np.asarray(out) - 1.0)) <= 1e-5, np.max(np.abs(np.asarray(out) - 1.0))


@pytest.mark.conservation
def test_default_point_radiates_the_requested_luminosity():
    """Default delta = 0.1, M = 1e7, alpha = 0.3, beta = 0.5, log L = 8: Q- / L = 1.

    The Eq. 49 inversion returned a state with Q-/L = 1.985 here.
    """
    lum, m, a, b, d = 10.0**8 * L_SUN, 1e7, 0.3, 0.5, 0.1
    mdot = float(_adaf_mdot_from_lbol(lum, m, a, b, d))
    t_e = float(_adaf_electron_temperature(m, mdot, a, b, d))
    assert ref.ref_heating(t_e, m, mdot, a, b, d) / lum == pytest.approx(1.0, abs=2e-6)


# ── (ii) Agreement with an independent inversion ─────────────────────────


@pytest.mark.regression_paper
def test_mdot_and_te_match_an_independent_inversion_of_the_full_luminosity():
    """mdot and T_e equal a scipy brentq inversion of L_ADAF = Q_ie + delta Q+ at 32 points, 1e-6.

    The reference nests its own brentq in T_e (Eq. 8) inside a brentq in ln mdot. The bound is
    the K_2 approximation of the package (about 1e-7 in g) carried through the sqrt.
    """
    pts, mdot = _unclipped_draws(32, seed=5)
    t_e = np.asarray(_TE(*[jnp.asarray(c) for c in (pts[:, 1], mdot, *pts[:, 2:].T)]))
    worst = 0.0
    for t, md, row in zip(t_e, mdot, pts):
        md_ref, t_ref, clipped = ref.ref_mdot_from_luminosity(*row)
        assert not clipped
        worst = max(worst, abs(md / md_ref - 1.0), abs(t / t_ref - 1.0))
    assert worst <= 1e-6, worst


@pytest.mark.regression_paper
@pytest.mark.parametrize(
    ("m", "alpha", "beta", "delta", "log_l"),
    [
        (1e7, 0.3, 0.5, 0.1, 8.0),
        (1e7, 0.3, 0.5, 0.1, 9.0),
        (1e10, 0.5, 0.1, 0.5, 8.0),
        (1e10, 0.05, 0.9, 1e-3, 8.0),
    ],
)
def test_published_corner_points_match_the_independent_inversion(m, alpha, beta, delta, log_l):
    """The default point and three box-edge points agree with the reference to 1e-6."""
    lum = 10.0**log_l * L_SUN
    mdot = float(_adaf_mdot_from_lbol(lum, m, alpha, beta, delta))
    t_e = float(_adaf_electron_temperature(m, mdot, alpha, beta, delta))
    md_ref, t_ref, clipped = ref.ref_mdot_from_luminosity(lum, m, alpha, beta, delta)
    assert not clipped
    assert mdot == pytest.approx(md_ref, rel=1e-6)
    assert t_e == pytest.approx(t_ref, rel=1e-6)


@pytest.mark.bounds
def test_clips_at_mdot_crit_and_at_the_floor_with_the_clip_derivative():
    """Above the luminosity at mdot_crit the result is 0.28 alpha^2 exactly; below that at
    mdot = 1e-8 it is 1e-8 (Eq. 52 and the floor).

    The derivative with respect to log L is zero on a clip, and the derivative with respect to
    alpha at the critical rate is 0.56 alpha.
    """
    m, alpha, beta, delta = 1e8, 0.3, 0.5, 0.1
    lum_crit, _ = ref.ref_luminosity(0.28 * alpha**2, m, alpha, beta, delta)
    over = float(_adaf_mdot_from_lbol(3.0 * lum_crit, m, alpha, beta, delta))
    assert over == 0.28 * alpha**2
    lum_floor, _ = ref.ref_luminosity(1e-8, m, alpha, beta, delta)
    under = float(_adaf_mdot_from_lbol(0.3 * lum_floor, m, alpha, beta, delta))
    assert under == 1e-8
    d_lum = jax.grad(lambda x: _adaf_mdot_from_lbol(10.0**x * L_SUN, m, alpha, beta, delta))
    assert float(d_lum(np.log10(3.0 * lum_crit / L_SUN))) == 0.0
    d_alpha = jax.grad(lambda a: _adaf_mdot_from_lbol(3.0 * lum_crit, m, a, beta, delta))
    assert float(d_alpha(alpha)) == pytest.approx(0.56 * alpha, rel=1e-12)


# ── (iii) The paper's closed forms are the two limits ────────────────────


@pytest.mark.limit
@pytest.mark.parametrize("frac", [0.3, 0.6, 0.95])
def test_eq49_is_the_limit_where_delta_q_plus_is_small(frac):
    """For mdot > 1e-3 alpha^2 and small delta the solution approaches Eq. 49.

    Eq. 49 is L = 1.2e38 g c1^-2 c3 beta r_min^-1 alpha^-2 m mdot^2 (Q_ie alone). With
    L = Q_ie (1 + eps), eps = delta Q+ / Q_ie, the Eq. 49 rate is mdot (1 + eps)^(1/2), so the
    difference is eps/2; the bound is eps itself (twice the neglected term), and eps is
    required to be below 3e-2 so that the point is in the limit.
    """
    m, alpha, beta, delta = 1e8, 0.3, 0.5, 1e-3
    mdot_true = frac * 0.28 * alpha**2
    lum, _ = ref.ref_luminosity(mdot_true, m, alpha, beta, delta)
    mdot = float(_adaf_mdot_from_lbol(lum, m, alpha, beta, delta))
    t_e = float(_adaf_electron_temperature(m, mdot, alpha, beta, delta))
    parts = ref._ref_parts(t_e, m, mdot, alpha, beta, delta)
    eps = parts["dq_plus"] / parts["q_ie"]
    assert eps < 3e-2
    mdot_49 = np.sqrt(lum / (_COEF_QIE * ref._ref_g(t_e) * beta * m / alpha**2))
    assert abs(mdot / mdot_49 - 1.0) <= eps, (mdot / mdot_49, eps)
    assert mdot == pytest.approx(mdot_true, rel=1e-6)


@pytest.mark.limit
@pytest.mark.parametrize("mdot_true", [1e-6, 1e-5])
def test_eq50_is_the_limit_where_delta_q_plus_dominates(mdot_true):
    """For mdot < 1e-3 alpha^2 the solution approaches Eq. 50, L = delta Q+ = B mdot.

    B = delta 9.39e38 ((1 - beta)/f) c3 r_min^-1 m (Eqs. 9, 14; f = 1). With
    L = delta Q+ (1 + eps'), eps' = Q_ie / delta Q+, the Eq. 50 rate L/B is mdot (1 + eps'),
    so the difference is eps'; eps' is required to be below 3e-2 so that the point is in the
    limit.
    """
    m, alpha, beta, delta = 1e8, 0.3, 0.5, 0.5
    lum, _ = ref.ref_luminosity(mdot_true, m, alpha, beta, delta)
    mdot = float(_adaf_mdot_from_lbol(lum, m, alpha, beta, delta))
    t_e = float(_adaf_electron_temperature(m, mdot, alpha, beta, delta))
    parts = ref._ref_parts(t_e, m, mdot, alpha, beta, delta)
    eps = parts["q_ie"] / parts["dq_plus"]
    assert eps < 3e-2
    mdot_50 = lum / (delta * _COEF_DQ * (1.0 - beta) * m)
    assert abs(mdot / mdot_50 - 1.0) <= 1.01 * eps / (1.0 + eps), (mdot / mdot_50, eps)
    assert mdot == pytest.approx(mdot_true, rel=1e-6)


# ── (iv) Exact second derivatives through the nested implicit solves ─────


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


# (log L_bol [Lsun], log M, alpha, beta, delta): the default, a high-mass point, a low-mass point.
_PUBLIC_POINTS = [
    (8.0, 7.0, 0.3, 0.5, 0.1),
    (9.0, 8.5, 0.4, 0.3, 0.02),
    (7.5, 6.5, 0.2, 0.7, 0.3),
]


def _mdot_of(x):
    return _adaf_mdot_from_lbol(10.0 ** x[0] * L_SUN, 10.0 ** x[1], x[2], x[3], x[4])


def _ln_mdot(x):
    return jnp.log(_mdot_of(x))


def _ln_te(x):
    return jnp.log(_adaf_electron_temperature(10.0 ** x[1], _mdot_of(x), x[2], x[3], x[4]))


@pytest.mark.gradient
@pytest.mark.parametrize("point", _PUBLIC_POINTS)
@pytest.mark.parametrize("fun", [_ln_mdot, _ln_te], ids=["ln_mdot", "ln_te"])
def test_hessian_matches_finite_difference_of_the_gradient(fun, point):
    """jax.hessian of ln mdot and of ln T_e w.r.t. (log L, log M, alpha, beta, delta) matches
    a central difference of the autodiff gradient to 1e-5 of the largest entry.

    The nested implicit-function derivatives (mdot from the luminosity, T_e from the balance)
    are exact at every order only if the second-order terms are kept at both levels.
    """
    x = jnp.asarray(point)
    assert float(_mdot_of(x)) < 0.28 * point[2] ** 2 * (1.0 - 1e-3)  # unclipped
    h_ad, h_fd, scale = _hessian_vs_fd_of_grad(fun, point, [1e-5, 1e-5, 1e-6, 1e-6, 1e-6])
    assert np.max(np.abs(h_ad - h_fd)) <= 1e-5 * scale, (np.max(np.abs(h_ad - h_fd)), scale)


@pytest.mark.gradient
def test_gradient_of_ln_mdot_matches_finite_difference_of_the_independent_inversion():
    """d ln mdot / d(log L, log M, alpha, beta, delta) equals a central difference of the
    independent brentq inversion at the default point, to 1e-6 of the largest entry."""
    point = np.asarray(_PUBLIC_POINTS[0])
    g_ad = np.asarray(jax.grad(_ln_mdot)(jnp.asarray(point)))
    steps = np.array([1e-4, 1e-4, 1e-5, 1e-5, 1e-5])

    def ln_md_ref(x):
        lum = 10.0 ** x[0] * L_SUN
        return np.log(ref.ref_mdot_from_luminosity(lum, 10.0 ** x[1], x[2], x[3], x[4])[0])

    g_fd = np.zeros(5)
    for j in range(5):
        e = np.zeros(5)
        e[j] = steps[j]
        g_fd[j] = (ln_md_ref(point + e) - ln_md_ref(point - e)) / (2.0 * steps[j])
    assert np.max(np.abs(g_ad - g_fd)) <= 1e-6 * np.max(np.abs(g_ad)), (g_ad, g_fd)


# ── The closed form at fixed g and the monotone luminosity ───────────────


@pytest.mark.limit
def test_fixed_g_closed_form_solves_the_quadratic_and_reduces_to_both_limits():
    """At fixed g, L = A g mdot^2 + B mdot has the root 2L / [B + sqrt(B^2 + 4 A g L)].

    The residual of the quadratic is zero to 1e-12, the B -> 0 limit is Eq. 49
    (mdot = sqrt(L / A g)) and the A -> 0 limit is Eq. 50 (mdot = L / B).
    """
    ln_root = adaf_mod._adaf_log_mdot_fixed_g
    for lum, a_g, b in [
        (1e44, 3e45, 1e44),
        (1e40, 3e45, 1e44),
        (1e46, 2e44, 5e43),
        (1e38, 1e42, 1e36),
    ]:
        md = np.exp(float(ln_root(np.log(lum), np.log(a_g), np.log(b))))
        assert (a_g * md**2 + b * md) / lum == pytest.approx(1.0, rel=1e-12)
    assert float(np.exp(ln_root(np.log(1e44), np.log(3e45), np.log(1e30)))) == pytest.approx(
        np.sqrt(1e44 / 3e45), rel=1e-9
    )
    assert float(np.exp(ln_root(np.log(1e44), np.log(1e30), np.log(5e43)))) == pytest.approx(
        1e44 / 5e43, rel=1e-9
    )


@pytest.mark.limit
def test_fixed_g_closed_form_is_float32_safe():
    """The quadratic root forms without overflow in float32 for B ~ 1e45 and L ~ 1e47 erg/s."""
    with jax.enable_x64(False):
        ln_l, ln_a, ln_b = (jnp.float32(np.log(v)) for v in (1e47, 5e47, 1e45))
        got = float(jnp.exp(adaf_mod._adaf_log_mdot_fixed_g(ln_l, ln_a, ln_b)))
    ref_md = 2e47 / (1e45 + np.sqrt(1e90 + 4 * 5e47 * 1e47))
    assert np.isfinite(got)
    assert got == pytest.approx(ref_md, rel=1e-5)


@pytest.mark.bounds
def test_luminosity_is_strictly_increasing_in_mdot_with_log_slope_at_least_one():
    """d ln L_ADAF / d ln mdot >= 1 over 60 box draws (Q_ie ~ mdot^2 g, delta Q+ ~ mdot).

    The slope is the autodiff derivative of ln(Q_ie + delta Q+) at the balance T_e. Being
    strictly increasing it makes the inversion root unique.
    """
    pts = _box_draws(60, seed=9)
    rng = np.random.default_rng(10)
    for _lum, m, a, b, d in pts:
        md = 10.0 ** rng.uniform(-7.0, np.log10(0.28 * a * a))

        def ln_lum(w, m=m, a=a, b=b, d=d):
            return adaf_mod._adaf_log_radiated_power(w, m, a, b, d)

        slope = float(jax.grad(ln_lum)(jnp.log(md)))
        assert slope >= 1.0 - 1e-9, (slope, m, md, a, b, d)
