# SPDX-License-Identifier: BSD-3-Clause
"""A non-finite Hessian is a distinct MAP status, never a zero decrement.

Before this contract a float32 Hessian with non-finite entries gave a Newton
decrement of 0 and ``positive_definite=False``, which the certificate read as an
ordinary non-positive-definite point. It then ran every continuation round, about
75 s per restart on the DESI-style fixture (#3205 follow-up). The rule pinned here:

* ``stationarity`` reports ``hessian_finite=False`` with a NaN decrement;
* ``_certify_map_point`` does not continue the point, so it runs no L-BFGS round;
* the point is reported with ``stationary=False`` and
  ``reason="hessian_not_finite"``, so ``map_converged`` is False and says why.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from tengri.inference.backends import map_dispatch
from tengri.inference.map_stationarity import stationarity

pytestmark = pytest.mark.contract


def _nonfinite_hessian_loss(vec):
    # f = 1e38 * x**3 at x = 1: f and the gradient 3e38 are representable in float32,
    # the curvature 6e38 overflows to inf.
    return jnp.float32(1e38) * jnp.sum(vec**3)


def _quadratic_loss(vec):
    return jnp.sum((vec - 1.0) ** 2)


def test_nonfinite_hessian_is_a_distinct_status():
    x = jnp.asarray([1.0], dtype=jnp.float32)
    stat = stationarity(_nonfinite_hessian_loss, x)
    assert stat.hessian_finite is False
    assert math.isnan(stat.decrement)
    assert stat.positive_definite is False
    assert stat.stationary is False


def test_finite_hessian_keeps_its_decrement():
    x = jnp.zeros(3, dtype=jnp.float32)
    stat = stationarity(_quadratic_loss, x)
    assert stat.hessian_finite is True
    assert stat.positive_definite is True
    assert math.isfinite(stat.decrement)


def test_certify_does_not_continue_a_nonfinite_point(monkeypatch):
    calls = []

    def _forbidden_lbfgs(*args, **kwargs):
        calls.append(1)
        raise AssertionError("continuation must not run for a non-finite Hessian")

    monkeypatch.setattr(map_dispatch, "lbfgs_to_gtol", _forbidden_lbfgs)
    x = jnp.asarray([1.0], dtype=jnp.float32)
    *_, decrement, stationary, attempts, reason = map_dispatch._certify_map_point(
        _nonfinite_hessian_loss,
        x,
        scipy_success=True,
        n_steps=50,
        gtol=1e-12,
    )
    assert calls == []
    assert stationary is False
    assert reason == "hessian_not_finite"
    assert attempts == 1
    assert math.isnan(decrement)


def test_finite_point_has_no_reason_and_continues(monkeypatch):
    seen = []
    original = map_dispatch.lbfgs_to_gtol

    def _recording_lbfgs(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(map_dispatch, "lbfgs_to_gtol", _recording_lbfgs)
    x = jnp.zeros(2, dtype=jnp.float64)
    *_, reason = map_dispatch._certify_map_point(
        _quadratic_loss,
        x,
        scipy_success=False,
        n_steps=200,
        gtol=1e-10,
    )
    assert reason is None
    assert len(seen) >= 1
