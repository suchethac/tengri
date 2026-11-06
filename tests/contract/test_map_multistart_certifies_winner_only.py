# SPDX-License-Identifier: BSD-3-Clause
"""A multistart MAP certifies only its winning restart.

Certifying every restart ran up to ``_MAP_CONTINUATION_ROUNDS`` L-BFGS rounds per restart,
so a float32 NUTS fit went from 21 s to 611 s (#3233). The rule pinned here:

* the restarts run scipy's optimizer with no certificate;
* the winner is the lowest finite loss at scipy's stop, so continuation cannot change it;
* ``_certify_map_point`` runs once, on that winner, and the returned point and its
  ``map_converged``, ``map_decrement`` and ``map_reason`` are what certifying that restart
  alone gives.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends import map_dispatch
from tengri.inference.context import InferenceContext
from tests.contract.test_map_multistart_kernel_reuse import fitter  # noqa: F401

pytestmark = pytest.mark.contract

_N_RESTARTS = 3
_N_STEPS = 60
_TOL = 1e-5


def _restarts(context, key):
    """Each restart's scipy stop, with no certificate, from the same keys the multistart uses."""
    return [
        map_dispatch._scipy_restart(
            context,
            init_params=context.initial_params(k, init_from=None),
            grad_fn=context.grad_fn,
            loss_fn=context.neg_log_posterior_fn,
            data_args=context.data_args,
            optimizer="lbfgs",
            n_steps=_N_STEPS,
            tol=_TOL,
        )
        for k in jax.random.split(key, _N_RESTARTS)
    ]


def test_multistart_certifies_only_the_lowest_loss_restart(fitter, monkeypatch):  # noqa: F811
    calls = []
    original = map_dispatch._certify_map_point

    def _spy(flat_loss, x, **kwargs):
        out = original(flat_loss, x, **kwargs)
        calls.append((jnp.asarray(x), out))
        return out

    monkeypatch.setattr(map_dispatch, "_certify_map_point", _spy)
    key = jax.random.PRNGKey(0)
    multi = fitter.run("map", key=key, n_restarts=_N_RESTARTS, n_steps=_N_STEPS, verbose=False)

    assert len(calls) == 1, f"_certify_map_point ran {len(calls)} times, expected once"

    context = InferenceContext.from_target(fitter)
    restarts = _restarts(context, key)
    losses = np.asarray([r.final_loss for r in restarts])
    best = int(np.argmin(losses))
    assert multi.diagnostics["restart"] == best

    winner = restarts[best]
    alone_loss = map_dispatch._flat_map_loss(
        context.neg_log_posterior_fn, context.data_args, winner.unravel_fn
    )
    alone = map_dispatch._certify_map_point(
        alone_loss,
        jnp.asarray(winner.x),
        scipy_success=winner.success,
        n_steps=_N_STEPS,
        gtol=_TOL,
    )
    spy_x, spy_out = calls[0]
    np.testing.assert_array_equal(spy_x, jnp.asarray(winner.x))
    assert spy_out[4:] == alone[4:], "the spied certificate differs from the winner alone"
    assert multi.diagnostics["map_converged"] == alone[4]
    assert multi.diagnostics["map_decrement"] == alone[3] or (
        np.isnan(multi.diagnostics["map_decrement"]) and np.isnan(alone[3])
    )
    expected_params = context.to_physical(winner.unravel_fn(alone[0]))
    for name, value in expected_params.items():
        np.testing.assert_array_equal(np.asarray(multi.params[name]), np.asarray(value))
