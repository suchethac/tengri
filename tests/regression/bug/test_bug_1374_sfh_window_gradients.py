# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SFH window bounds must be fittable, and normalized exactly (#1374).

Seven freeable SFH parameters had **exactly zero** autodiff gradient for every
valid configuration, so any gradient-based method (NUTS, gradient MAP, VI)
explored them on prior signal alone with the likelihood contributing nothing,
and nothing warned:

    constant.start   constant.end   exponential.start   constant_then_exponential.age
    psb_wild2020.age   sfh2exp.age_yr   periodic.age_yr

Every one entered its model **only** through a hard boolean window such as
``(t_lookback >= 0) & (t_lookback <= age)``. Point-sampling a step function makes
the result a staircase — it changes only when a grid node crosses the boundary —
whose derivative is zero almost everywhere. ``sfhdelayed.age`` was the control
that proved this was not a generic autodiff limitation: it uses the *same* mask
but ``age`` also survives in a ``dt`` prefactor, and its gradient was alive.

The fix replaces the point-sampled step with :func:`window_weight`, the exact
cell average of the window indicator. That is not a smoothing hack, and it is not
only a gradient fix:

* ``trapezoid`` weights node ``i`` by exactly the cell width used here (including
  the half-width end nodes), so the window integrates to its true length and a
  renormalized top-hat carries its requested mass **exactly**, at any resolution.
  The old hard mask over-counted by up to one cell: measured **1.97 %** error at
  the production default ``n_grid=256`` and 4.02 % at 128, converging away only
  as the grid refines.
* it introduces no free parameter — a sigmoid window would need a width, and the
  age grid is log-spaced so no single width in years is right across the range.

The one cell straddling a boundary carries the fraction of its star formation
that falls inside the window. That is the same defect #964 fixed for the DSPS
histogram kernel, where annihilating a straddling segment re-attributed 3.8 % of
the mass and biased the optical CSP by +1.2 %.

References
----------
.. [1] A. C. Carnall et al., "Inferring the star formation histories of massive
   quiescent galaxies with Bagpipes," MNRAS, 480, 4379 (2018).
   https://doi.org/10.1093/mnras/sty2169
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.mean_sfh import (
    constant,
    constant_then_exponential,
    exponential,
    periodic,
    psb_wild2020,
    sfh2exp,
    sfhdelayed,
    window_weight,
)

pytestmark = pytest.mark.regression_bug

_T = jnp.logspace(5.0, 10.14, 256)


def _mass_weighted(sfr):
    """A read-out that responds to SHAPE.

    Total mass is pinned to ``10**log_total_mass`` by ``_renormalize_to_mass``, so
    ``d(mass)/d(anything)`` is zero even for a healthy model — it cannot
    discriminate a dead gradient from a working one.
    """
    return jnp.trapezoid(sfr * jnp.exp(-_T / 1e9), _T)


#: (label, fn, kwargs, name of the window parameter, value to probe at).
#: Every entry was measured GRADIENT-DEAD before the fix.
_WINDOW_PARAMS = [
    ("constant.start", constant, {"log_total_mass": 10.0, "end": 8e9}, "start", 1e9),
    ("constant.end", constant, {"log_total_mass": 10.0, "start": 1e9}, "end", 8e9),
    ("exponential.start", exponential, {"log_total_mass": 10.0, "tau": 1e9}, "start", 1e9),
    (
        "constant_then_exponential.age",
        constant_then_exponential,
        {"log_total_mass": 10.0, "tau": 1e9, "quench_age": 2e9},
        "age",
        6e9,
    ),
    (
        "psb_wild2020.age",
        psb_wild2020,
        {
            "log_total_mass": 10.0,
            "tau": 1e9,
            "burstage": 5e8,
            "alpha": 3.0,
            "beta": 3.0,
            "fburst": 0.3,
        },
        "age",
        6e9,
    ),
    (
        "sfh2exp.age_yr",
        sfh2exp,
        {
            "log_total_mass": 10.0,
            "tau_main_yr": 4e9,
            "tau_burst_yr": 3e8,
            "f_burst": 0.1,
            "burst_age_yr": 5e8,
        },
        "age_yr",
        6e9,
    ),
    (
        "periodic.age_yr",
        periodic,
        {
            "log_total_mass": 10.0,
            "delta_bursts_yr": 5e8,
            "tau_bursts_yr": 2e8,
            "burst_type": 0,
        },
        "age_yr",
        6e9,
    ),
]


@pytest.mark.parametrize(
    ("label", "fn", "kwargs", "param", "value"),
    _WINDOW_PARAMS,
    ids=[c[0] for c in _WINDOW_PARAMS],
)
def test_window_parameter_has_a_correct_gradient(label, fn, kwargs, param, value):
    """LOAD-BEARING. Neuter: restore any hard mask and its entry here fails.

    Asserts the gradient is not merely nonzero but **right**: it must match a
    central finite difference. A nonzero-but-wrong gradient would still mislead a
    sampler, so ``!= 0`` alone is not the invariant worth pinning.
    """

    def scalar(x):
        return _mass_weighted(fn(_T, **{**kwargs, param: x}))

    g = float(jax.grad(scalar)(value))
    h = value * 1e-6
    fd = (float(scalar(value + h)) - float(scalar(value - h))) / (2.0 * h)

    assert np.isfinite(g)
    assert abs(fd) > 1e-12, f"probe setup failed: {label} does not move the read-out here"
    assert g != 0.0, (
        f"{label} has EXACTLY zero gradient while the finite difference is {fd:.4g} — "
        "the window is being point-sampled again"
    )
    assert abs(g - fd) <= 1e-4 * abs(fd), (
        f"{label} gradient is wrong, not just present: autodiff {g:.6g} vs "
        f"finite difference {fd:.6g}"
    )


def test_the_live_control_is_still_live():
    """``sfhdelayed.age`` was never dead — it must not regress either.

    It shares the mask but keeps a smooth ``dt`` prefactor, which is what proved
    the original defect was specific to mask-only parameters rather than a
    limitation of autodiff through ``jnp.where``.
    """

    def scalar(x):
        return _mass_weighted(sfhdelayed(_T, log_total_mass=10.0, tau=1e9, age=x))

    g = float(jax.grad(scalar)(6e9))
    fd = (float(scalar(6e9 + 6e3)) - float(scalar(6e9 - 6e3))) / 1.2e4
    assert abs(g - fd) <= 1e-4 * abs(fd)


@pytest.mark.parametrize("n_grid", [64, 128, 256, 1024])
def test_constant_sfh_matches_the_analytic_top_hat_at_every_resolution(n_grid):
    """The normalization is exact, not merely convergent.

    A top-hat of width ``end - start`` carrying mass ``M`` has
    ``SFR = M / (end - start)`` — an analytic ground truth that owes nothing to
    this implementation. The point-sampled mask was 1.97 % out at ``n_grid=256``
    and 4.02 % at 64-128, converging only as the grid refined.
    """
    t = jnp.logspace(5.0, 10.14, n_grid)
    start, end, log_m = 5e8, 5e9, 10.0
    sfr = np.asarray(constant(t, log_total_mass=log_m, start=start, end=end))

    interior = (np.asarray(t) > start * 1.05) & (np.asarray(t) < end * 0.95)
    assert interior.sum() > 5, "probe setup failed: too few interior nodes"
    analytic = 10.0**log_m / (end - start)
    plateau = float(sfr[interior].mean())
    assert abs(plateau - analytic) <= 1e-12 * analytic, (
        f"n_grid={n_grid}: plateau {plateau:.8f} vs analytic {analytic:.8f} — the "
        "window is not being integrated exactly"
    )
    # Flat, not merely correct on average.
    assert float(np.std(sfr[interior])) <= 1e-12 * analytic


class TestWindowWeightItself:
    """Direct contract of the helper the fix is built on."""

    @pytest.mark.parametrize(
        ("lo", "hi", "label"),
        [
            (5e8, 5e9, "interior"),
            (-1e30, 1e30, "covers the whole grid"),
            (1e5, 1e10, "touches both grid edges"),
        ],
    )
    def test_integrates_to_the_window_length(self, lo, hi, label):
        """LOAD-BEARING: ``trapezoid``-consistency is the whole point.

        Neuter: mirror the end cells outward (full width instead of half) and the
        'touches both grid edges' case fails — ``trapezoid`` weights the end nodes
        by half a spacing, so a full-width end cell is inconsistent with the very
        quadrature this is meant to be exact under.
        """
        got = float(jnp.trapezoid(window_weight(_T, lo, hi), _T))
        want = min(hi, float(_T[-1])) - max(lo, float(_T[0]))
        assert abs(got - want) <= 1e-12 * abs(want), f"{label}: {got:.6e} vs {want:.6e}"

    def test_bounded_and_interior_cells_are_exactly_one(self):
        w = np.asarray(window_weight(_T, 5e8, 5e9))
        assert w.min() >= 0.0 and w.max() <= 1.0
        t = np.asarray(_T)
        interior = (t > 5e8 * 1.2) & (t < 5e9 * 0.8)
        assert np.all(w[interior] == 1.0), "interior cells must be untouched by the fix"
        assert np.all(w[(t < 5e8 * 0.8) | (t > 5e9 * 1.2)] == 0.0)

    def test_only_the_straddling_cells_differ_from_a_hard_mask(self):
        """The change is local: at most one cell per boundary."""
        t = np.asarray(_T)
        w = np.asarray(window_weight(_T, 5e8, 5e9))
        hard = np.where((t >= 5e8) & (t <= 5e9), 1.0, 0.0)
        assert int(np.sum(np.abs(w - hard) > 1e-12)) <= 2

    def test_descending_grid_mirrors_the_ascending_answer(self):
        w_asc = np.asarray(window_weight(_T, 5e8, 5e9))
        w_desc = np.asarray(window_weight(_T[::-1], 5e8, 5e9))
        assert np.allclose(w_desc, w_asc[::-1])

    def test_single_node_grid_does_not_crash(self):
        """Degenerate grid: no neighbor exists, so the point test is the answer."""
        w = np.asarray(window_weight(jnp.array([1e9]), 5e8, 5e9))
        assert w.shape == (1,) and w[0] == 1.0


def test_total_mass_is_preserved_for_every_patched_model():
    """The renormalization contract still holds after the window change."""
    for label, fn, kwargs, param, value in _WINDOW_PARAMS:
        sfr = fn(_T, **{**kwargs, param: value})
        mass = float(jnp.trapezoid(sfr, _T))
        assert abs(mass - 1e10) <= 1e-6 * 1e10, f"{label}: mass {mass:.6e} != 1e10"


def test_jit_and_vmap_safe():
    """The window sits on the traced path for every patched model."""
    ages = jnp.array([4e9, 6e9, 8e9])

    def one(age):
        return jnp.sum(sfh2exp(_T, 10.0, 4e9, 3e8, 0.1, age, 5e8))

    out = jax.jit(jax.vmap(one))(ages)
    assert out.shape == (3,) and bool(jnp.all(jnp.isfinite(out)))
