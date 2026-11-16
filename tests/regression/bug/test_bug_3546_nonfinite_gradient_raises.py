# SPDX-License-Identifier: BSD-3-Clause
"""MAP and Laplace raise on a non-finite gradient instead of returning (#3546).

A model whose value is finite but whose gradient is NaN for one parameter made
scipy's L-BFGS-B stop after 0 iterations, and ``method='map'`` returned that
starting point as a ``Posterior`` with ``grad_norm = nan``. ``method='laplace'``
then formed a NaN Hessian and returned NaN parameters and a NaN log-evidence.

The objectives here are closed forms with a finite value and a NaN derivative in
one named parameter, so no SSP grid is needed. The runners are called directly:
the scipy MAP runner raises before it reads anything from its context.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from tengri.config.exceptions import InferenceError, NonFiniteGradientError
from tengri.inference.backends import laplace, map_dispatch

pytestmark = pytest.mark.regression_bug


def _dead_root(b):
    """Zero for every ``b``, with a NaN first derivative at ``b <= 0``."""
    return jnp.sqrt(jnp.maximum(b, 0.0) * 0.0)


def _loss_nan_gradient(params, data_args):
    return jnp.sum((params["a"] - 1.0) ** 2) + _dead_root(params["b"])


def _loss_finite(params, data_args):
    return jnp.sum((params["a"] - 1.0) ** 2) + 0.5 * params["b"] ** 2


_INIT = {"a": jnp.asarray([0.2, -0.3]), "b": jnp.asarray(0.0)}


def _run_scipy(loss, context=None):
    return map_dispatch._run_map_scipy(
        context if context is not None else types.SimpleNamespace(),
        init_params=_INIT,
        grad_fn=jax.value_and_grad(loss),
        loss_fn=loss,
        data_args=None,
        optimizer="lbfgs",
        n_steps=50,
        tol=1e-8,
        verbose=False,
    )


def test_fixture_value_is_finite_and_gradient_is_not():
    value, grad = jax.value_and_grad(_loss_nan_gradient)(_INIT, None)
    # (0.2 - 1)**2 + (-0.3 - 1)**2 = 2.33, and d/da = 2 (a - 1).
    np.testing.assert_allclose(float(value), 2.33, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(grad["a"]), [-1.6, -2.6], rtol=1e-12)
    assert np.isnan(float(grad["b"]))


def test_map_raises_and_names_the_parameter():
    with pytest.raises(NonFiniteGradientError, match="'b'") as excinfo:
        _run_scipy(_loss_nan_gradient)
    assert excinfo.value.parameters == ("b",)
    assert isinstance(excinfo.value, InferenceError)


def test_multistart_raises_when_every_restart_has_a_nonfinite_gradient():
    context = types.SimpleNamespace(
        initial_params=lambda key, init_from=None: _INIT,
        grad_fn=jax.value_and_grad(_loss_nan_gradient),
    )
    with pytest.raises(NonFiniteGradientError, match="'b'"):
        map_dispatch._run_map_multistart_scipy(
            context,
            key=jax.random.PRNGKey(0),
            n_restarts=3,
            n_steps=50,
            tol=1e-8,
            optimizer="lbfgs",
            verbose=False,
            loss_fn=_loss_nan_gradient,
            data_args=None,
        )


def _run_laplace(loss, position):
    flat, unravel = ravel_pytree(position)
    return laplace.run_laplace(
        key=jax.random.PRNGKey(0),
        logdensity_fn=lambda vec, data_args: -loss(unravel(vec), data_args),
        data_args=None,
        position=flat,
        unravel_fn=unravel,
        to_physical_fn=lambda params: params,
        model=None,
        n_samples=16,
        verbose=False,
    )


def test_laplace_raises_on_a_nonfinite_gradient():
    with pytest.raises(NonFiniteGradientError, match="'b'") as excinfo:
        _run_laplace(_loss_nan_gradient, _INIT)
    assert excinfo.value.parameters == ("b",)


def test_laplace_raises_on_a_nonfinite_hessian_with_a_finite_gradient():
    def loss(params, data_args):
        b = params["b"]
        return jnp.sum((params["a"] - 1.0) ** 2) + jnp.where(b > 0.0, b**1.5, 0.0)

    grad = jax.grad(loss)(_INIT, None)
    # The gradient at b = 0 is exactly zero; only the second derivative is NaN.
    assert float(grad["b"]) == 0.0
    with pytest.raises(NonFiniteGradientError, match="Hessian") as excinfo:
        _run_laplace(loss, _INIT)
    assert excinfo.value.parameters == ("b",)


def test_laplace_on_a_finite_objective_still_returns():
    """Unit Gaussian in three dimensions: mode at (1, 1, 0), unit covariance / 2 and 1."""
    posterior = _run_laplace(_loss_finite, _INIT)
    # The polish reaches the mode, where the loss is 0: log Z = -0.5 ln(1 * 2 * 2).
    np.testing.assert_allclose(
        posterior.diagnostics["log_evidence"], -0.5 * np.log(4.0), rtol=1e-6
    )
    np.testing.assert_allclose(posterior.diagnostics["eigenvalues"], [1.0, 2.0, 2.0], rtol=1e-6)


def test_multistart_keeps_the_restarts_that_have_a_finite_gradient(monkeypatch):
    calls = []

    def one_start(context, **kwargs):
        calls.append(len(calls))
        if calls[-1] == 0:
            raise NonFiniteGradientError("refused", parameters=("b",))
        return types.SimpleNamespace(
            diagnostics={"final_loss": 3.0 - calls[-1]},
            params={"a": jnp.zeros(2), "b": jnp.asarray(float(calls[-1]))},
            loss_history=jnp.asarray([1.0]),
        )

    monkeypatch.setattr(map_dispatch, "_run_map_scipy", one_start)
    context = types.SimpleNamespace(
        initial_params=lambda key, init_from=None: _INIT, grad_fn=None, model=None
    )
    posterior = map_dispatch._run_map_multistart_scipy(
        context,
        key=jax.random.PRNGKey(0),
        n_restarts=3,
        n_steps=50,
        tol=1e-8,
        optimizer="lbfgs",
        verbose=False,
        loss_fn=_loss_finite,
        data_args=None,
    )
    assert len(calls) == 3
    # Restarts 1 and 2 ran (losses 2.0 and 1.0); the lower one wins.
    assert float(posterior.params["b"]) == 2.0
