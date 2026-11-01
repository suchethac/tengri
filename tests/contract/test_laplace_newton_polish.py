# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the damped Newton polish in tengri.inference.backends.laplace.

Covers the shared helpers in :mod:`tengri.inference.newton` (on log-densities)
and the ``newton_steps`` polish of :func:`run_laplace` end to end.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from tengri.config.exceptions import LaplaceNotAtModeWarning
from tengri.inference.backends.laplace import run_laplace
from tengri.inference.newton import (
    damped_newton_step,
    first_accepted_step,
    newton_trial_steps,
)

pytestmark = pytest.mark.contract


def _quadratic_logdensity(mode, precision):
    """Log-density ``-0.5 (x - mode)^T A (x - mode)`` for SPD ``A``."""

    def logdensity(x, _data_args):
        d = x - mode
        return -0.5 * d @ precision @ d

    return logdensity


def _spd_matrix(n_dim, seed=0):
    rng = np.random.RandomState(seed)
    m = rng.normal(size=(n_dim, n_dim))
    return jnp.asarray(m @ m.T + n_dim * np.eye(n_dim))


def _indefinite_loss(v):
    """``0.5 x^2 - 0.5 y^2 + 0.25 y^4``: negative curvature in y near the origin."""
    return 0.5 * v[0] ** 2 - 0.5 * v[1] ** 2 + 0.25 * v[1] ** 4


def _indefinite_logdensity(x, _data_args):
    return -_indefinite_loss(x)


def test_one_damped_step_lands_on_the_quadratic_mode():
    """Newton is exact on a quadratic: one step from any start reaches the mode."""
    mode = jnp.asarray([0.5, -1.0, 2.0, 0.1])
    precision = _spd_matrix(4)
    logd = _quadratic_logdensity(mode, precision)
    x0 = mode + jnp.asarray([0.7, -0.4, 0.3, 1.2])

    x_new, accepted = damped_newton_step(x0, logd, None)

    assert bool(accepted)
    np.testing.assert_allclose(np.asarray(x_new), np.asarray(mode), atol=1e-10)


def test_the_accepted_step_strictly_raises_an_indefinite_logdensity():
    """At (0.3, 0.2) the y curvature is negative; the step must still ascend."""
    x0 = jnp.asarray([0.3, 0.2])

    x_new, accepted = damped_newton_step(x0, _indefinite_logdensity, None)

    assert bool(accepted)
    assert float(_indefinite_logdensity(x_new, None)) > float(_indefinite_logdensity(x0, None))


def test_undamped_saddle_free_step_is_the_accepted_descent_step_away_from_the_saddle():
    """At (0.3, 0.2) the undamped trial (index 0) is accepted and its y step moves away from 0.

    The expected step is ``-V diag(1/|lambda|) V^T g`` from the loss derivatives, so
    dropping the absolute value flips the y component's sign and fails here.
    """
    x0 = jnp.asarray([0.3, 0.2])
    loss0 = _indefinite_loss(x0)
    loss_grad = jax.grad(_indefinite_loss)(x0)
    loss_hessian = jax.hessian(_indefinite_loss)(x0)

    steps, _ = newton_trial_steps(loss_grad, loss_hessian)
    trial_losses = jax.vmap(_indefinite_loss)(x0 + steps)
    x_new, accepted, index = first_accepted_step(x0, steps, trial_losses, loss0)

    assert bool(accepted)
    assert int(index) == 0
    np.testing.assert_allclose(np.asarray(x_new), np.asarray(x0 + steps[0]))

    g = np.asarray(loss_grad, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(loss_hessian, dtype=float))
    expected = -(eigenvectors @ ((eigenvectors.T @ g) / np.abs(eigenvalues)))
    np.testing.assert_allclose(np.asarray(steps[0]), expected, atol=1e-12)

    step_y = float(steps[0, 1])
    # The loss falls as |y| grows here (y**2 < 1), so the y step must point along
    # sign(y0), i.e. away from the saddle at y = 0, and against the y gradient.
    assert step_y * float(x0[1]) > 0.0
    assert np.sign(step_y) == -np.sign(float(loss_grad[1]))
    assert abs(step_y) == pytest.approx(0.192 / 0.88, rel=1e-6)


def test_at_the_exact_mode_nothing_is_accepted_and_x_is_bit_identical():
    """With zero gradient every trial equals the start, so none is strictly better."""
    mode = jnp.asarray([0.5, -1.0, 2.0, 0.1])
    logd = _quadratic_logdensity(mode, _spd_matrix(4))

    x_new, accepted = damped_newton_step(mode, logd, None)

    assert not bool(accepted)
    assert np.array_equal(np.asarray(x_new), np.asarray(mode))


def test_jit_and_vmap_match_a_python_loop():
    """The step is jit- and vmap-safe and agrees with a per-start Python loop."""
    mode = jnp.asarray([0.5, -1.0, 2.0, 0.1])
    logd = _quadratic_logdensity(mode, _spd_matrix(4))
    starts = jnp.stack(
        [
            mode + jnp.asarray([0.7, -0.4, 0.3, 1.2]),
            mode + jnp.asarray([-0.2, 0.9, -1.1, 0.5]),
            mode + jnp.asarray([0.1, 0.1, 0.1, -0.3]),
        ]
    )

    def step(x):
        return damped_newton_step(x, logd, None)

    loop = [step(x) for x in starts]
    loop_x = np.stack([np.asarray(x) for x, _ in loop])
    loop_acc = np.asarray([bool(a) for _, a in loop])

    jit_x, jit_acc = jax.jit(step)(starts[0])
    vmap_x, vmap_acc = jax.vmap(step)(starts)

    np.testing.assert_allclose(np.asarray(jit_x), loop_x[0], atol=1e-12)
    assert bool(jit_acc) == bool(loop_acc[0])
    np.testing.assert_allclose(np.asarray(vmap_x), loop_x, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(vmap_acc), loop_acc)


def test_first_accepted_step_ignores_non_finite_rows():
    """A NaN step that is not selected must not leak into the result."""
    x = jnp.asarray([1.0, 2.0])
    steps = jnp.asarray([[jnp.nan, 0.0], [0.5, 0.5]])
    trial_losses = jnp.asarray([0.0, 0.0])

    x_new, accepted, index = first_accepted_step(x, steps, trial_losses, 1.0)

    assert bool(accepted)
    assert int(index) == 1
    np.testing.assert_allclose(np.asarray(x_new), [1.5, 2.5])


def _mode_of(loss, start):
    """Minimize ``loss`` numerically to a tight tolerance."""
    grad = jax.jit(jax.grad(loss))
    result = minimize(
        lambda v: float(loss(jnp.asarray(v))),
        np.asarray(start),
        jac=lambda v: np.asarray(grad(jnp.asarray(v))),
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 1000},
    )
    return jnp.asarray(result.x)


def _smooth_loss(v):
    a, b = v[0], v[1]
    return 0.5 * a**2 + 0.5 * (b - 1.0) ** 2 + 0.1 * (a * b) ** 2 + 0.05 * b**4


def _smooth_logdensity(position, _data_args):
    return -_smooth_loss(position)


def _run_smooth(start, newton_steps):
    return run_laplace(
        key=jax.random.PRNGKey(0),
        logdensity_fn=_smooth_logdensity,
        data_args=None,
        position=start,
        unravel_fn=lambda v: {"a": v[0], "b": v[1]},
        to_physical_fn=lambda p: p,
        model=None,
        n_samples=200,
        newton_steps=newton_steps,
        verbose=False,
    )


def test_run_laplace_polish_reaches_the_mode_without_a_warning():
    """From an offset start the polish lands on the mode: decrement ~0, no warning."""
    mode = _mode_of(_smooth_loss, [0.0, 0.5])
    start = mode + jnp.asarray([0.3, -0.3])

    with warnings.catch_warnings():
        warnings.simplefilter("error", LaplaceNotAtModeWarning)
        # Quadratic convergence from a 0.3 offset: 2 steps leave |grad| ~ 7e-4 and
        # 4 steps reach ~5e-15, so 4 is the count that clears the 1e-8 decrement bar.
        post = _run_smooth(start, newton_steps=4)

    assert post.diagnostics["newton_decrement"] < 1e-8
    assert post.diagnostics["newton_steps_accepted"] >= 1


def test_run_laplace_without_the_polish_warns_and_accepts_nothing():
    """With newton_steps=0 the same offset start is reported as off-mode."""
    mode = _mode_of(_smooth_loss, [0.0, 0.5])
    start = mode + jnp.asarray([0.3, -0.3])

    with pytest.warns(LaplaceNotAtModeWarning):
        post = _run_smooth(start, newton_steps=0)

    assert post.diagnostics["newton_steps_accepted"] == 0
    assert post.diagnostics["newton_decrement"] > 0.1


def test_negative_newton_steps_raises():
    with pytest.raises(ValueError, match="newton_steps"):
        _run_smooth(jnp.asarray([0.0, 1.0]), newton_steps=-1)
