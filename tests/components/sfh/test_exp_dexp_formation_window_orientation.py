# SPDX-License-Identifier: BSD-3-Clause
"""``exp``/``dexp`` form stars in ``[0, age(z)]``, not ``[start, infinity)`` (#2521, F-new-1).

``exponential`` and ``delayed_exponential`` windowed their shape to
``[start, inf)`` -- mass sits at lookback ``>= start``, unbounded toward the
oldest SSP template age, with zero SFR between ``start`` and the present.
That is the mirror image of both their own docstrings ("declining
exponential *from* start") and their verified-correct sibling
``declining_exponential``, which uses ``T = age - t_lookback`` and confines
its mass to ``[0, age]``. Even the smallest legal ``start`` (well inside
``age(z)``) puts virtually all mass past ``start``, which the #683 runtime
clamp then discards -- ``exp``/``dexp`` were the only two "z-capped" families
that still lost mass at onset multiplier 0.5 (sweep_S3_report.md Sec. 3,
F-new-1).

The fix reinterprets ``start`` as the formation lookback time (the
description already reads "when did SF start?") and gives both functions the
same ``[0, start]`` formation window as ``declining_exponential``, with
``T = start - t_lookback``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.conservation

from tengri.components.stellar.sfh.mean_sfh import (
    declining_exponential,
    delayed_exponential,
    exponential,
)

LB = jnp.linspace(1e5, 14.0e9, 20_001)  # fine lookback grid [yr]


def _pdf(sfr):
    sfr = np.maximum(np.asarray(sfr, dtype=float), 0.0)
    mass = np.trapezoid(sfr, np.asarray(LB))
    return sfr / mass


def _l1(p, q):
    """Total-variation-like L1 distance between two PDFs on the shared LB grid."""
    return 0.5 * float(np.trapezoid(np.abs(p - q), np.asarray(LB)))


def _old_buggy_exponential(t_lookback, log_total_mass, tau, start):
    """The pre-fix ``exponential`` formula: window ``[start, inf)`` (F-new-1)."""
    from tengri.components.stellar.sfh.mean_sfh import _renormalize_to_mass, window_weight

    dt = jnp.maximum(t_lookback - start, 0.0)
    shape = jnp.exp(-dt / tau) * window_weight(t_lookback, start, jnp.inf)
    return _renormalize_to_mass(shape, t_lookback, log_total_mass)


def _old_buggy_delayed_exponential(t_lookback, log_total_mass, tau, start):
    """The pre-fix ``delayed_exponential`` formula: window ``[start, inf)`` (F-new-1)."""
    from tengri.components.stellar.sfh.mean_sfh import _renormalize_to_mass, window_weight

    dt = jnp.maximum(t_lookback - start, 0.0)
    ratio = dt / tau
    raw = ratio * jnp.exp(-ratio + 1.0)
    shape = jnp.maximum(raw, 0.0) * window_weight(t_lookback, start, jnp.inf)
    return _renormalize_to_mass(shape, t_lookback, log_total_mass)


@pytest.mark.parametrize("tau_gyr,anchor_gyr", [(2.0, 5.0), (1.0, 3.0), (3.0, 8.0)])
def test_exponential_matches_declining_exponential_orientation(tau_gyr, anchor_gyr):
    """Fixed ``exponential(start=anchor)`` == ``declining_exponential(age=anchor)``.

    Same tau, same anchor: the two must now be the SAME shape (mirror-distance
    to the correct orientation < 1e-3), and far from the old, wrong orientation
    (mirror-distance to the OLD buggy formula > 0.05).
    """
    tau, anchor = tau_gyr * 1e9, anchor_gyr * 1e9
    p = _pdf(exponential(LB, log_total_mass=10.0, tau=tau, start=anchor))
    q_correct = _pdf(declining_exponential(LB, log_total_mass=10.0, tau=tau, age=anchor))
    q_old_buggy = _pdf(_old_buggy_exponential(LB, log_total_mass=10.0, tau=tau, start=anchor))

    d_correct = _l1(p, q_correct)
    d_old = _l1(p, q_old_buggy)
    assert d_correct < 1e-3, f"tau={tau_gyr} anchor={anchor_gyr}: D(correct)={d_correct}"
    assert d_old > 0.05, f"tau={tau_gyr} anchor={anchor_gyr}: D(old buggy)={d_old}"


@pytest.mark.parametrize("tau_gyr,anchor_gyr", [(2.0, 5.0), (1.0, 3.0), (3.0, 8.0)])
def test_delayed_exponential_orientation_flipped_from_old_buggy_formula(tau_gyr, anchor_gyr):
    """Fixed ``delayed_exponential`` confines its mass to ``[0, start]``, not ``[start, inf)``.

    ``delayed_exponential`` has no verified-correct sibling with the identical
    shape (its rise-then-decline is its own convention), so this pins the
    orientation directly: mass at lookback > start must vanish (the window
    fix), and the new shape must be far from the old buggy one, which put
    (virtually) all its mass there instead.
    """
    tau, anchor = tau_gyr * 1e9, anchor_gyr * 1e9
    p = _pdf(delayed_exponential(LB, log_total_mass=10.0, tau=tau, start=anchor))
    q_old_buggy = _pdf(
        _old_buggy_delayed_exponential(LB, log_total_mass=10.0, tau=tau, start=anchor)
    )

    d_old = _l1(p, q_old_buggy)
    assert d_old > 0.05, f"tau={tau_gyr} anchor={anchor_gyr}: D(old buggy)={d_old}"

    lb_np = np.asarray(LB)
    mass_after_start = np.trapezoid(np.where(lb_np > anchor, p, 0.0), lb_np)
    assert mass_after_start < 1e-6, (
        f"tau={tau_gyr} anchor={anchor_gyr}: {mass_after_start:.3e} of the normalized "
        f"PDF still sits at lookback > start (the old [start, inf) window)"
    )


@pytest.mark.parametrize("family_fn,anchor_gyr", [(exponential, 5.0), (delayed_exponential, 3.0)])
def test_zero_sfr_before_formation(family_fn, anchor_gyr):
    """No star formation at lookback > start (the formation time) -- window is [0, start].

    Excludes the single grid cell straddling ``start`` itself: ``window_weight``
    is a cell-AVERAGED window (#1374, a differentiable soft edge, not a hard
    step), so that one cell legitimately carries a partial, nonzero weight.
    """
    anchor = anchor_gyr * 1e9
    sfr = np.asarray(family_fn(LB, log_total_mass=10.0, tau=1e9, start=anchor))
    lb_np = np.asarray(LB)
    margin = 3.0 * float(np.mean(np.diff(lb_np)))  # a few grid cells past the soft edge
    np.testing.assert_array_equal(sfr[lb_np > anchor + margin], 0.0)
