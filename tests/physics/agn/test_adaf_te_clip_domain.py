# SPDX-License-Identifier: BSD-3-Clause
"""The electron-temperature solve on the whole clip domain (Mahadevan 1997, Eq. 8, Sect. 5.1.1).

Paper: R. Mahadevan, ApJ 477, 585 (1997), arXiv:astro-ph/9609107.

The accretion rate derived from a luminosity is clipped to [1e-8, 0.28 alpha^2] (Eq. 52), so
the T_e solve must converge for every mdot in that range, not only for the rates reachable
from the declared (log L, log M) box (mdot >= 8e-5). The box is the declared prior box of
(m, alpha, beta, delta). At mdot = 1e-8 the root climbs to 6.3e11 K at
(m, alpha, beta, delta) = (1e10, 0.5, 0.9, 0.5).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import kve
from tests import _adaf_reference as ref

from tengri.components.agn import adaf as adaf_mod
from tengri.components.agn.adaf import _adaf_balance_residual, _adaf_electron_temperature

pytestmark = [pytest.mark.regression_paper, pytest.mark.bounds]

_TE = jax.jit(jax.vmap(_adaf_electron_temperature))
_RESID = jax.jit(jax.vmap(_adaf_balance_residual))
_CORNERS = [
    (m, 1e-8, a, b, d)
    for m, a, b, d in itertools.product([1e6, 1e10], [0.05, 0.5], [0.1, 0.9], [1e-3, 0.5])
]


def _clip_domain_points(n, seed=3):
    """mdot log-uniform on [1e-8, 0.28 alpha^2], the other parameters over the box."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.05, 0.5, n)
    pts = np.stack(
        [
            10.0 ** rng.uniform(6.0, 10.0, n),
            np.exp(rng.uniform(np.log(1e-8), np.log(0.28 * alpha**2))),
            alpha,
            rng.uniform(0.1, 0.9, n),
            10.0 ** rng.uniform(-3.0, np.log10(0.5), n),
        ],
        axis=1,
    )
    pts[: 2 * n // 5, 1] = 10.0 ** rng.uniform(-8.0, -6.0, 2 * n // 5)  # the slow end
    pts[:, 1] = np.minimum(pts[:, 1], 0.28 * alpha**2)
    return np.concatenate([pts, np.asarray(_CORNERS)], axis=0)


def _args(pts):
    return [jnp.asarray(pts[:, i]) for i in range(5)]


def test_residual_vanishes_at_the_returned_root_over_the_clip_domain():
    """max |R| <= twice the solver tolerance at 2000 clip-domain draws and the 16 corners.

    The solver stops at |R| <= tol on its own iterate; the residual is recomputed here at
    ln(exp(u)), whose rounding adds a few 1e-14 (tol is 2.2e-13 in float64).
    """
    pts = _clip_domain_points(2000)
    t_e = _TE(*_args(pts))
    res = _RESID(jnp.log(t_e), *_args(pts))
    tol = adaf_mod._adaf_solver_tol(res.dtype)
    assert float(jnp.max(jnp.abs(res))) <= 2.0 * tol, (float(jnp.max(jnp.abs(res))), tol)


def test_bracket_encloses_the_root_over_the_clip_domain():
    """The residual is positive at the bracket floor and negative at its ceiling at every point,
    and the returned root is strictly inside, not on an end."""
    pts = _clip_domain_points(2000, seed=8)
    args = _args(pts)
    r_lo = _RESID(jnp.full(len(pts), np.log(adaf_mod._T_LO)), *args)
    r_hi = _RESID(jnp.full(len(pts), np.log(adaf_mod._T_HI)), *args)
    assert bool(jnp.all(r_lo > 0.0)) and bool(jnp.all(r_hi < 0.0))
    t_e = np.asarray(_TE(*args))
    assert t_e.min() > 1.001 * adaf_mod._T_LO and t_e.max() < 0.999 * adaf_mod._T_HI


def test_root_matches_the_independent_solver_over_the_clip_domain():
    """T_e equals the scipy brentq root (bracket [1e8, 1e13] K) at the 16 corners and at 60 draws,
    1e-6 relative, including the worst corner (6.27e11 K)."""
    pts = np.concatenate([np.asarray(_CORNERS), _clip_domain_points(60, seed=21)[:60]])
    t_code = np.asarray(_TE(*_args(pts)))
    t_ref = np.array([ref._ref_root(*p) for p in pts])
    rel = np.abs(t_code / t_ref - 1.0)
    assert rel.max() <= 1e-6, (rel.max(), pts[rel.argmax()])
    assert t_ref.max() > 6.0e11  # the corner that broke the former 5e11 K ceiling


def test_bracket_ceiling_is_a_derived_bound():
    """The ceiling exceeds the synchrotron-only temperature bound at every corner of the box.

    With x_M >= 1 (Eq. 20 floor), P_synch >= 0.71 s1^3 (1.19e-13)^3 s3 r_min^(-7/4) m^(1/2)
    mdot^(3/2) T_e^7 (Eqs. 21-24, 26), and Q_ie + delta Q+ <= 1.2e38 g_max c1^-2 c3 beta
    r_min^-1 alpha^-2 m mdot^2 + delta 9.39e38 (1 - beta) c3 r_min^-1 m mdot (Eqs. 13, 9, 14),
    with g_max the largest g(theta_e) of Eq. 11 for T_e >= 1e8 K. Where the first exceeds the
    second no root exists, so the root is below the temperature where they are equal.
    """
    th = np.logspace(-3, 4, 40000)
    th = th[th >= ref._K_B * adaf_mod._T_LO / (ref._M_E * ref._C**2)]
    g_max = float(np.max((2 + 2 * th + 1 / th) / kve(2, 1 / th)))
    c1, c3, r = ref._C1, ref._C3, ref._RMIN
    worst = 0.0
    for m, a, b, d in itertools.product([1e6, 1e10], [0.05, 0.5], [0.1, 0.9], [1e-3, 0.5]):
        for md in np.logspace(-8, np.log10(0.28 * a * a), 60):
            s1_cubed = (1.42e9 * a**-0.5 * (1 - b) ** 0.5 * c1**-0.5 * c3**0.5) ** 3
            coef = 0.71 * s1_cubed * (1.19e-13) ** 3 * 1.05e-24 * r**-1.75 * m**0.5 * md**1.5
            heat = 1.2e38 * g_max * c1**-2 * c3 * b * m * md**2 / a**2 / r
            heat += d * 9.39e38 * (1 - b) * c3 * m * md / r
            worst = max(worst, (heat / coef) ** (1 / 7))
    assert worst < adaf_mod._T_HI, (worst, adaf_mod._T_HI)
