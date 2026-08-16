# SPDX-License-Identifier: BSD-3-Clause
"""Tests for quasi-Newton optimizers in MAP dispatch.

Verifies that L-BFGS is accessible through the MAP optimizer= kwarg
and converges on simple problems. Also tests jaxopt solver builder
for the batch/vmap path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def quadratic_loss():
    """A simple quadratic loss: ||params - target||^2, with data_args."""

    def loss_fn(params, data_args):
        target = data_args["target"]
        return jnp.sum((params["x"] - target) ** 2)

    return loss_fn


@pytest.fixture()
def rosenbrock_loss():
    """Rosenbrock function — a harder 2-D test surface."""

    def loss_fn(params, data_args):
        x, y = params["x"][0], params["x"][1]
        return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2

    return loss_fn


# ── Builder tests ─────────────────────────────────────────────────


class TestBuildJaxoptSolver:
    def test_builds_lbfgs(self, quadratic_loss):
        from tengri.inference.backends.map_dispatch import _build_jaxopt_solver

        solver, name = _build_jaxopt_solver(
            "lbfgs",
            quadratic_loss,
            maxiter=50,
            tol=1e-6,
        )
        assert name == "L-BFGS"
        assert hasattr(solver, "run")
        assert hasattr(solver, "update")

    def test_unknown_optimizer_error(self):
        from tengri.inference.backends.map_dispatch import _build_optax_optimizer

        with pytest.raises(ValueError, match="Unknown optimizer"):
            _build_optax_optimizer("bogus", 0.01)


# ── Convergence tests (standalone, no Fitter needed) ──────────────


class TestJaxoptConvergence:
    """Verify each solver converges on a simple quadratic."""

    def test_converges_quadratic(self, quadratic_loss):
        from tengri.inference.backends.map_dispatch import _build_jaxopt_solver

        solver, _ = _build_jaxopt_solver(
            "lbfgs",
            quadratic_loss,
            maxiter=100,
            tol=1e-8,
        )
        init_params = {"x": jnp.array([5.0, -3.0, 7.0])}
        data_args = {"target": jnp.array([1.0, 2.0, 3.0])}

        result = solver.run(init_params, data_args)
        assert jnp.allclose(result.params["x"], data_args["target"], atol=1e-4)

    def test_converges_rosenbrock(self, rosenbrock_loss):
        from tengri.inference.backends.map_dispatch import _build_jaxopt_solver

        solver, _ = _build_jaxopt_solver(
            "lbfgs",
            rosenbrock_loss,
            maxiter=200,
            tol=1e-8,
        )
        init_params = {"x": jnp.array([-1.0, 1.0])}
        data_args = {}

        result = solver.run(init_params, data_args)
        assert jnp.allclose(result.params["x"], jnp.array([1.0, 1.0]), atol=1e-3)


# ── Constants / registry tests ────────────────────────────────────


class TestOptimizerRegistry:
    def test_jaxopt_solvers_set(self):
        from tengri.inference.backends.map_dispatch import _JAXOPT_SOLVERS

        assert {"lbfgs", "lbfgs_scipy"} == _JAXOPT_SOLVERS

    def test_all_optimizers_includes_both(self):
        from tengri.inference.backends.map_dispatch import (
            _ALL_OPTIMIZERS,
            _JAXOPT_SOLVERS,
            _OPTAX_OPTIMIZERS,
        )

        assert _ALL_OPTIMIZERS == _OPTAX_OPTIMIZERS | _JAXOPT_SOLVERS

    def test_error_message_lists_all(self):
        from tengri.inference.backends.map_dispatch import _build_optax_optimizer

        with pytest.raises(ValueError, match="lbfgs"):
            _build_optax_optimizer("bogus", 0.01)


# ── vmap batch tests (standalone) ─────────────────────────────────


class TestJaxoptVmap:
    """Verify jaxopt solvers work with jax.vmap for batch optimization."""

    def test_vmap_run_converges(self, quadratic_loss):
        from tengri.inference.backends.map_dispatch import _build_jaxopt_solver

        solver, _ = _build_jaxopt_solver(
            "lbfgs",
            quadratic_loss,
            maxiter=100,
            tol=1e-8,
        )

        batch_init = {"x": jnp.array([[5.0, -3.0], [0.0, 10.0], [-5.0, 5.0]])}
        batch_data = {"target": jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])}

        result = jax.vmap(solver.run)(batch_init, batch_data)

        for i in range(3):
            params_i = result.params["x"][i]
            assert jnp.allclose(params_i, jnp.array([1.0, 2.0]), atol=1e-4), (
                f"Galaxy {i} did not converge: {params_i}"
            )
