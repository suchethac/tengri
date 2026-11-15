# SPDX-License-Identifier: BSD-3-Clause
"""The certificate's programs compile once per objective, not once per round (#3530).

A continuation round of ``_certify_map_point`` runs the shared L-BFGS driver. The driver
used to run eagerly, so every round re-traced and recompiled its while loop (about 0.7 s
per round on a warm quickstart fit). The contract pinned here:

* a second certificate on the same objective compiles nothing, including its
  continuation rounds;
* the rounds of one certificate compile nothing after the first round;
* the data is a traced argument: a new dataset of the same shape reuses the programs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.backends import map_dispatch
from tengri.inference.map_stationarity import stationarity

pytestmark = pytest.mark.contract

_COMPILES = [0]


def _count_compile(event, duration, **kwargs):
    if "backend_compile" in event:
        _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count_compile)


def _compiles() -> int:
    return _COMPILES[0]


def _quadratic_loss(vec, data):
    curvature = jnp.arange(1.0, vec.size + 1.0, dtype=vec.dtype)
    return 0.5 * jnp.sum(curvature * vec**2) + jnp.sum(data * vec)


def _loss_for(data):
    return lambda vec: _quadratic_loss(vec, data)


@pytest.fixture
def round_compiles(monkeypatch):
    """Record ``(compiles_at_entry, compiles_at_exit)`` for each continuation round."""
    real = map_dispatch.lbfgs_to_gtol
    rounds = []

    def recording(*args, **kwargs):
        entry = _compiles()
        out = real(*args, **kwargs)
        rounds.append((entry, _compiles()))
        return out

    monkeypatch.setattr(map_dispatch, "lbfgs_to_gtol", recording)
    return rounds


def test_second_stationarity_on_the_same_loss_compiles_nothing():
    loss = _loss_for(jnp.zeros(4, dtype=jnp.float64))
    x = jnp.full((4,), 0.5, dtype=jnp.float64)
    stationarity(loss, x)
    before = _compiles()
    stationarity(loss, x)
    assert _compiles() - before == 0


def test_continuation_rounds_compile_nothing_after_the_first(round_compiles):
    loss = _loss_for(jnp.zeros(4, dtype=jnp.float64))
    x0 = jnp.full((4,), 3.0, dtype=jnp.float64)
    map_dispatch._certify_map_point(loss, x0, scipy_success=False, n_steps=3, gtol=1e-14)
    assert len(round_compiles) == 2, "the certificate should run both continuation rounds"
    second_entry, second_exit = round_compiles[1]
    assert second_exit - second_entry == 0, "the second round recompiled"


def test_second_certificate_with_continuation_compiles_nothing(round_compiles):
    loss = _loss_for(jnp.zeros(4, dtype=jnp.float64))
    x0 = jnp.full((4,), 3.0, dtype=jnp.float64)
    map_dispatch._certify_map_point(loss, x0, scipy_success=False, n_steps=3, gtol=1e-14)
    before = _compiles()
    map_dispatch._certify_map_point(loss, x0, scipy_success=False, n_steps=3, gtol=1e-14)
    assert _compiles() - before == 0


def test_new_data_of_the_same_shape_reuses_the_continuation_programs(round_compiles):
    """The data is traced: the second dataset compiles nothing and is the one certified."""
    x0 = jnp.full((4,), 3.0, dtype=jnp.float64)
    data_a = jnp.ones(4, dtype=jnp.float64)
    data_b = jnp.full((4,), -2.0, dtype=jnp.float64)
    map_dispatch._certify_map_point(
        _quadratic_loss, x0, scipy_success=False, n_steps=3, gtol=1e-14, data=data_a
    )
    before = _compiles()
    x_b, nlp_b, *_ = map_dispatch._certify_map_point(
        _quadratic_loss, x0, scipy_success=False, n_steps=3, gtol=1e-14, data=data_b
    )
    assert _compiles() - before == 0, "a new dataset of the same shape recompiled"
    np_expected = float(_quadratic_loss(jnp.asarray(x_b), data_b))
    assert nlp_b == pytest.approx(np_expected, rel=1e-10)
