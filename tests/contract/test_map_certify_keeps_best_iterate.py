# SPDX-License-Identifier: BSD-3-Clause
"""MAP certification returns the best iterate, never the last one.

A continuation round can raise the loss. ``_certify_map_point`` must return the lowest
finite loss among the scipy stop and every round's end, with the stationarity measured
at that returned point.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends import map_dispatch
from tengri.inference.map_stationarity import Stationarity

pytestmark = pytest.mark.contract


def _quad(v):
    return jnp.sum((v - 1.0) ** 2)


def test_certify_returns_best_iterate_when_a_round_raises_the_loss(monkeypatch):
    x0 = jnp.zeros(3)
    x_low = jnp.full(3, 0.9)  # round 1: loss 0.03, lower than the start (3.0)
    x_high = jnp.full(3, 5.0)  # round 2: loss 48.0, higher than round 1
    outputs = iter([x_low, x_high])

    def _fake_lbfgs(flat_loss, x, *, n_steps, gtol):
        return next(outputs), 0.0, None

    monkeypatch.setattr(map_dispatch, "lbfgs_to_gtol", _fake_lbfgs)
    x, nlp, _, _, stationary, attempts, reason = map_dispatch._certify_map_point(
        _quad, x0, scipy_success=True, n_steps=10, gtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(x), np.asarray(x_low))
    assert nlp == pytest.approx(0.03, rel=1e-6)
    assert attempts == 3
    assert reason is None
    assert stationary is False


def _stat(nlp, decrement):
    return Stationarity(
        nlp=nlp, grad_norm=1.0, decrement=decrement, positive_definite=True, hessian_finite=True
    )


def _script_certify(monkeypatch, stats, points):
    """Feed ``_certify_map_point`` scripted stationarity measures and round outputs."""
    stat_seq = list(stats)
    point_seq = list(points)
    calls = []

    def _next(seq):
        # Past the script's end the last entry repeats, so an unexpected extra round
        # fails on its assertion instead of on StopIteration.
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def _fake_stationarity(loss, x):
        return _next(stat_seq)

    def _fake_lbfgs(flat_loss, x, *, n_steps, gtol):
        calls.append(x)
        return _next(point_seq), 0.0, None

    monkeypatch.setattr(map_dispatch, "stationarity", _fake_stationarity)
    monkeypatch.setattr(map_dispatch, "lbfgs_to_gtol", _fake_lbfgs)
    return calls


def test_map_certify_accepts_stationary_within_round_off(monkeypatch):
    nlp0 = 163.24656257987317
    eps = float(jnp.finfo(jnp.float64).eps)
    x0, x_c = jnp.zeros(3), jnp.full(3, 0.5)
    # Round 1 reaches a stationary point whose loss is 3 eps |nlp| above the start's.
    calls = _script_certify(
        monkeypatch,
        [_stat(nlp0, 1e-3), _stat(nlp0 + 3 * eps * abs(nlp0), 1e-12)],
        [x_c],
    )
    x, nlp, _, _, stationary, attempts, reason = map_dispatch._certify_map_point(
        _quad, x0, scipy_success=True, n_steps=10, gtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(x), np.asarray(x_c))
    assert nlp == pytest.approx(nlp0 + 3 * eps * abs(nlp0), rel=1e-12)
    assert stationary is True
    assert attempts == 2
    assert reason is None
    assert len(calls) == 1


def test_map_certify_rejects_stationary_with_real_rise(monkeypatch):
    nlp0 = 161.77
    x0, x_c = jnp.zeros(3), jnp.full(3, 0.5)
    # Round 1 reaches a stationary point 48 nats above the start: a different, higher
    # minimum. It is not the lowest-loss iterate, so the start is kept and not converged.
    _script_certify(monkeypatch, [_stat(nlp0, 1e-3), _stat(210.65, 1e-12)], [x_c])
    x, nlp, _, _, stationary, attempts, reason = map_dispatch._certify_map_point(
        _quad, x0, scipy_success=True, n_steps=10, gtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(x), np.asarray(x0))
    assert nlp == pytest.approx(nlp0)
    assert stationary is False
    assert attempts == 2
    assert reason is None


def test_map_certify_stops_when_a_round_does_not_improve(monkeypatch):
    nlp0 = 161.77
    x0, x_high = jnp.zeros(3), jnp.full(3, 5.0)
    # Round 1 is rejected (a real rise). Another round from the same best point would
    # replay it, so no second round runs.
    calls = _script_certify(monkeypatch, [_stat(nlp0, 1e-3), _stat(210.65, 1e-3)], [x_high])
    _, _, _, _, _, attempts, _ = map_dispatch._certify_map_point(
        _quad, x0, scipy_success=True, n_steps=10, gtol=1e-12
    )
    assert len(calls) == 1
    assert attempts == 2
