# SPDX-License-Identifier: BSD-3-Clause
"""run_laplace refuses a non-finite explicit init_from before it reaches the Hessian.

An explicit start point skips run_map, and with it the non-finite check that every MAP
result passes through (#1397). The Laplace path must apply the same check to the
unbounded point it actually uses.
"""

import jax.numpy as jnp
import pytest

from tengri.inference.backends import map_dispatch
from tengri.inference.context import InferenceContext


class _FakeContext:
    """Minimal stand-in exposing the one method run_laplace calls on the init point."""

    neg_log_posterior_fn = None
    data_args = None
    grad_fn = None
    model = None

    def __init__(self, unbounded):
        self._unbounded = unbounded

    def unbounded_from_posterior(self, posterior):
        return self._unbounded

    def to_physical(self, params):
        return params


@pytest.fixture
def spy_laplace(monkeypatch):
    """Record calls into the Laplace solver instead of running a Hessian."""
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return "laplace-result"

    import tengri.inference.backends.laplace as laplace_mod

    monkeypatch.setattr(laplace_mod, "run_laplace", _spy)
    monkeypatch.setattr(InferenceContext, "from_target", staticmethod(lambda c: c))
    return calls


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf])
def test_laplace_rejects_non_finite_init_from(spy_laplace, bad):
    """A NaN or inf coordinate in init_from raises before the Laplace solver runs."""
    ctx = _FakeContext({"sfh_dpl_alpha": jnp.array(bad), "dust_tau": jnp.array(0.5)})
    with pytest.raises(ValueError, match="sfh_dpl_alpha"):
        map_dispatch.run_laplace(ctx, key=None, init_from=object())
    assert spy_laplace == []


def test_laplace_accepts_finite_init_from(spy_laplace):
    """A finite explicit start reaches the Laplace solver unchanged."""
    point = {"sfh_dpl_alpha": jnp.array(1.0), "dust_tau": jnp.array(0.5)}
    ctx = _FakeContext(point)
    out = map_dispatch.run_laplace(ctx, key=None, init_from=object())
    assert out == "laplace-result"
    assert spy_laplace[0]["map_params_unbounded"] is point
