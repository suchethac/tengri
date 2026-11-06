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
