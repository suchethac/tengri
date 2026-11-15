# SPDX-License-Identifier: BSD-3-Clause
"""The certificate's decisions and reasons are those of the reference implementation (#3530).

``stationarity`` was made cheaper (one fused program, one host transfer). The contract is
unchanged, and it is pinned here against a copy of the previous implementation:

* ``hessian_finite``, ``positive_definite`` and ``stationary`` are identical on every case;
* ``nlp``, ``grad_norm`` and ``decrement`` agree to round-off, and a non-finite Hessian keeps
  a NaN decrement.

The cases cover a converged minimum, a point with gradient above the tolerance, a saddle
(not positive definite), a non-finite Hessian, a float32 quadratic and a loss that takes
its data as a traced argument.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.map_stationarity import MAP_DECREMENT_BAR, Stationarity, stationarity

pytestmark = pytest.mark.contract


def _reference_stationarity(loss, x, data=None) -> Stationarity:
    """The previous ``stationarity``: separate jitted value-and-gradient, Hessian, then eigh."""

    def live(vec, dat):
        return loss(vec) if dat is None else loss(vec, dat)

    value_and_grad = jax.jit(jax.value_and_grad(live))
    hessian_fn = jax.jit(jax.hessian(live))
    x = jnp.asarray(x)
    value, grad = value_and_grad(x, data)
    hessian = hessian_fn(x, data)
    grad_norm = float(jnp.linalg.norm(grad))
    if not bool(jnp.all(jnp.isfinite(hessian))):
        return Stationarity(
            nlp=float(value),
            grad_norm=grad_norm,
            decrement=float("nan"),
            positive_definite=False,
            hessian_finite=False,
        )
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    positive = eigenvalues > 0
    safe = jnp.where(positive, eigenvalues, 1.0)
    projected = (eigenvectors.T @ grad) ** 2
    decrement = 0.5 * jnp.sum(jnp.where(positive, projected / safe, 0.0))
    return Stationarity(
        nlp=float(value),
        grad_norm=grad_norm,
        decrement=float(decrement),
        positive_definite=bool(jnp.all(positive)),
        hessian_finite=True,
    )


def _quadratic(vec):
    return 0.5 * jnp.sum(jnp.arange(1.0, vec.size + 1.0) * vec**2)


def _shifted_quadratic(vec):
    return jnp.sum((vec - 1.0) ** 2) + 0.1 * jnp.sum(jnp.sin(vec))


def _saddle(vec):
    return 0.5 * (vec[0] ** 2 - vec[1] ** 2)


def _nonfinite_hessian(vec):
    # Gradient 3e38 is representable in float32, the curvature 6e38 overflows to inf.
    return jnp.float32(1e38) * jnp.sum(vec**3)


def _data_loss(vec, data):
    return jnp.sum(data * (vec - 1.0) ** 2) + 0.05 * jnp.sum(vec**4)


CASES = [
    # name, loss, x, data
    ("converged_minimum", _quadratic, jnp.zeros(4), None),
    ("gradient_above_bar", _shifted_quadratic, jnp.full((3,), 3.0), None),
    ("nonconverged_point", _shifted_quadratic, jnp.full((3,), 1.0001), None),
    ("saddle_not_pd", _saddle, jnp.asarray([0.0, 0.0]), None),
    ("nonfinite_hessian", _nonfinite_hessian, jnp.asarray([1.0], dtype=jnp.float32), None),
    (
        "float32_quadratic",
        _shifted_quadratic,
        jnp.full((4,), 2.0, dtype=jnp.float32),
        None,
    ),
    ("traced_data", _data_loss, jnp.full((3,), 0.5), jnp.asarray([1.0, 2.0, 3.0])),
]


def _assert_same(new: Stationarity, ref: Stationarity, rtol: float) -> None:
    assert new.hessian_finite is ref.hessian_finite
    assert new.positive_definite is ref.positive_definite
    assert new.stationary is ref.stationary
    for field in ("nlp", "grad_norm", "decrement"):
        a, b = getattr(new, field), getattr(ref, field)
        if math.isnan(b):
            assert math.isnan(a), f"{field}: reference NaN, new {a}"
        else:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-30, err_msg=field)


@pytest.mark.parametrize(("name", "loss", "x", "data"), CASES, ids=[c[0] for c in CASES])
def test_decisions_and_reasons_match_the_reference(name, loss, x, data):
    rtol = 1e-5 if jnp.asarray(x).dtype == jnp.float32 else 1e-10
    new = stationarity(loss, x, data)
    ref = _reference_stationarity(loss, x, data)
    _assert_same(new, ref, rtol)


@pytest.mark.parametrize("gradient_scale", [0.0, 0.5, 1e3])
def test_decision_boundary_matches_the_reference(gradient_scale):
    """A decrement straddling the bar keeps its verdict: the bar is compared, not rounded."""

    def loss(vec):
        return 0.5 * jnp.sum(vec**2) + gradient_scale * jnp.sum(vec)

    x = jnp.full((2,), -gradient_scale)
    new = stationarity(loss, x)
    ref = _reference_stationarity(loss, x)
    _assert_same(new, ref, 1e-10)
    assert new.stationary is (ref.decrement < MAP_DECREMENT_BAR)
