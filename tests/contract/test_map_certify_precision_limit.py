# SPDX-License-Identifier: BSD-3-Clause
"""Float32 certification stops at scipy's point; float64 keeps its continuation rounds.

A float32 loss cannot resolve the Newton-decrement bar (1e-8 nats), so the continuation
rounds could only replay round-off. The contract pinned here:

* float32: no L-BFGS round runs, and the point is reported with ``map_reason`` set to
  ``"precision_limit"`` and ``map_converged`` False;
* float64: the continuation rounds still run from a non-stationary point.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.inference.backends import map_dispatch
from tengri.inference.map_stationarity import Stationarity

pytestmark = pytest.mark.contract


def _quadratic(dtype):
    """A 4-D positive-definite quadratic with a gradient away from its minimum."""
    curvature = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=dtype))

    def loss(vec):
        return 0.5 * vec @ curvature @ vec + jnp.sum(vec) * 0.25

    x0 = jnp.full((4,), 3.0, dtype=dtype)
    return loss, x0


def _spy_lbfgs(monkeypatch):
    """Wrap ``lbfgs_to_gtol`` so each continuation round is counted, not changed."""
    real = map_dispatch.lbfgs_to_gtol
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(map_dispatch, "lbfgs_to_gtol", spy)
    return calls


def test_float32_certification_runs_no_round_and_reports_precision_limit(monkeypatch):
    loss, x0 = _quadratic(jnp.float32)
    calls = _spy_lbfgs(monkeypatch)
    x, _nlp, _gn, _dec, stationary, attempts, reason = map_dispatch._certify_map_point(
        loss, x0, scipy_success=False, n_steps=50, gtol=1e-6
    )
    assert calls == [], "float32 certification ran a continuation round it cannot certify"
    assert reason == "precision_limit"
    assert stationary is False
    assert attempts == 1
    assert x.dtype == jnp.float32


def test_float32_stationary_point_is_converged_even_when_scipy_failed(monkeypatch):
    loss, x0 = _quadratic(jnp.float32)
    stationary = Stationarity(
        nlp=1.0, grad_norm=0.0, decrement=1e-12, positive_definite=True, hessian_finite=True
    )
    monkeypatch.setattr(map_dispatch, "stationarity", lambda flat_loss, x: stationary)
    _, _, _, _, converged, attempts, reason = map_dispatch._certify_map_point(
        loss, x0, scipy_success=False, n_steps=50, gtol=1e-6
    )
    assert converged is True
    assert reason is None
    assert attempts == 1


def test_float64_certification_keeps_its_continuation_rounds(monkeypatch):
    loss, x0 = _quadratic(jnp.float64)
    calls = _spy_lbfgs(monkeypatch)
    map_dispatch._certify_map_point(loss, x0, scipy_success=False, n_steps=50, gtol=1e-6)
    assert len(calls) >= 1, "float64 certification skipped its continuation rounds"
