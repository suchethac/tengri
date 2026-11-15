# SPDX-License-Identifier: BSD-3-Clause
r"""A non-finite Hessian is reported by name at every consumer, never as NaN draws (#3505).

The Laplace backend used to take a NaN Hessian through its eigenvalue regularizer and
report an indefinite curvature with "most negative nan", then return NaN draws; the
information content and the hierarchical mode search did the same through their
Cholesky. The rule now lives in :mod:`tengri.inference.curvature`: a value, gradient or
Hessian that is not finite raises :class:`NonFiniteCurvatureError`, naming which of the
three, which coordinates and the dtype, in float64 as in float32.

The objectives below have a finite gradient and a NaN second derivative: the Hessian is
the first thing that fails, which is the case a gradient-only check misses.
"""

from __future__ import annotations

import gc
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.curvature import (
    Curvature,
    NonFiniteCurvatureError,
    curvature,
    flat_names,
    require_finite,
)

pytestmark = pytest.mark.regression_bug


@jax.custom_jvp
def _linear(x):
    return x


@_linear.defjvp
def _linear_jvp(primals, tangents):
    return primals[0], jnp.nan * tangents[0]


@jax.custom_jvp
def _bad_quadratic(x):
    """0.5 x^2 with a finite first derivative and a NaN second derivative."""
    return 0.5 * x * x


@_bad_quadratic.defjvp
def _bad_quadratic_jvp(primals, tangents):
    return _bad_quadratic(primals[0]), _linear(primals[0]) * tangents[0]


def _one_row_nan(x, _data=None):
    return -0.5 * jnp.sum(x**2) - _bad_quadratic(x[1])


def _all_nan(x, _data=None):
    return -0.5 * jnp.sum(x**2) - _bad_quadratic(jnp.sum(x))


OBJECTIVES = {"one_row": (_one_row_nan, "rows 1"), "all": (_all_nan, "rows 0, 1, 2")}
DTYPES = [(True, "float64"), (False, "float32")]


def _point(dtype):
    return jnp.asarray([0.3, 0.2, -0.1], dtype=dtype)


def _dtype_of(x64):
    return jnp.float64 if x64 else jnp.float32


@pytest.mark.parametrize("x64,dtype_name", DTYPES)
@pytest.mark.parametrize("which", OBJECTIVES)
def test_helper_names_what_is_non_finite(which, x64, dtype_name):
    fn, _rows = OBJECTIVES[which]
    with jax.enable_x64(x64):
        curv = curvature(fn, _point(_dtype_of(x64)), None)
        assert not bool(curv.finite)
        assert np.all(np.isfinite(np.asarray(curv.grad)))
        with pytest.raises(NonFiniteCurvatureError) as err:
            curv.require_finite(names=["a", "b", "c"], where="probe")
    message = str(err.value)
    assert message.startswith("probe: Hessian has")
    assert dtype_name in message
    assert "gradient" not in message.split("(dtype")[0]
    expected = {"one_row": "rows b)", "all": "rows a, b, c)"}[which]
    assert expected in message


def test_a_finite_curvature_passes_and_returns_itself():
    def f(x, _):
        return -0.5 * jnp.sum(x**2)

    curv = curvature(f, jnp.ones(3), None)
    assert isinstance(curv.require_finite(), Curvature)
    assert bool(curv.finite)


def test_value_and_gradient_failures_are_named_apart_from_the_hessian():
    with pytest.raises(NonFiniteCurvatureError) as err:
        require_finite(value=jnp.asarray(jnp.nan), grad=jnp.asarray([1.0, jnp.inf]))
    text = str(err.value)
    assert "value is nan" in text and "gradient has 1 of 2" in text and "Hessian" not in text


def test_traced_values_skip_the_check():
    @jax.jit
    def g(x):
        require_finite(hess=jnp.full((2, 2), jnp.nan) + 0.0 * x)
        return x

    assert float(g(jnp.asarray(1.0))) == 1.0


def test_flat_names_follow_the_pytree():
    from jax.flatten_util import ravel_pytree

    _, unravel = ravel_pytree({"mass": jnp.zeros(()), "xi": jnp.zeros(2)})
    assert flat_names(unravel, 3) == ("mass", "xi[0]", "xi[1]")


# -- consumers ----------------------------------------------------------------------


@pytest.mark.parametrize("x64,dtype_name", DTYPES)
@pytest.mark.parametrize("which", OBJECTIVES)
def test_preconditioning_metric_and_information_raise(which, x64, dtype_name):
    from tengri.inference.preconditioning import negative_hessian_metric

    fn, _ = OBJECTIVES[which]
    with (
        jax.enable_x64(x64),
        pytest.raises(NonFiniteCurvatureError, match=f"Hessian.*dtype {dtype_name}"),
    ):
        negative_hessian_metric(fn, _point(_dtype_of(x64)), None, floor=0.0, names=["a", "b", "c"])


@pytest.mark.parametrize("which", OBJECTIVES)
def test_newton_polish_raises(which):
    from tengri.inference.newton import newton_polish

    fn, _ = OBJECTIVES[which]
    with pytest.raises(NonFiniteCurvatureError, match="Hessian"):
        newton_polish(_point(jnp.float64), fn, None, max_steps=3, tol=1e-9)


@pytest.mark.parametrize("which", OBJECTIVES)
def test_laplace_raises_instead_of_returning_nan_draws(which):
    from tengri.inference.backends.laplace import run_laplace

    fn, _rows = OBJECTIVES[which]
    with pytest.raises(NonFiniteCurvatureError) as err:
        run_laplace(
            key=jax.random.PRNGKey(0),
            logdensity_fn=fn,
            data_args=None,
            position=_point(jnp.float64),
            unravel_fn=lambda v: {"a": v[0], "b": v[1], "c": v[2]},
            to_physical_fn=None,
            model=None,
        )
    message = str(err.value)
    assert message.startswith("laplace: Hessian has")
    assert "indefinite" not in message.split(".")[0] and "most negative" not in message
    # The Hessian is symmetrized before the check, which spreads a NaN row to its column.
    assert "rows" in message and "b" in message.split("rows")[1].split(")")[0]


@pytest.mark.parametrize("which", OBJECTIVES)
def test_map_stationarity_keeps_its_documented_distinct_status(which):
    """The stationarity certificate reports ``hessian_finite=False``, by design.

    A MAP certificate must say "curvature unknown" rather than raise: the caller decides
    whether an unknown curvature blocks the fit (``map_dispatch`` turns it into the
    ``hessian_not_finite`` reason).
    """
    from tengri.inference.map_stationarity import stationarity

    fn, _ = OBJECTIVES[which]
    status = stationarity(lambda v: -fn(v), _point(jnp.float64))
    assert status.hessian_finite is False
    assert not status.stationary
    assert np.isnan(status.decrement)


@pytest.mark.parametrize("which", OBJECTIVES)
def test_hierarchical_laplace_raises(which, monkeypatch):
    from tengri.inference import _hierarchical_flat as flat

    fn, _ = OBJECTIVES[which]
    x0 = _point(jnp.float64)
    prob = types.SimpleNamespace(
        n_dim=3,
        init_flat=x0,
        log_prob=lambda v: fn(v),
        log_prob_with_data=fn,
        data_args=(),
        unravel=lambda v: {"a": v[0], "b": v[1], "c": v[2]},
    )
    monkeypatch.setattr(flat, "build_flat_problem", lambda *a, **k: prob)
    fitter = types.SimpleNamespace(n_galaxies=1)
    with pytest.raises(NonFiniteCurvatureError, match="Hessian"):
        flat.run_flat_sampler(
            fitter, "laplace", key=jax.random.PRNGKey(0), map_steps=5, verbose=False
        )


# -- program cache identity -----------------------------------------------------------


class _ValueEqual:
    """Equal and same-hash for every instance, yet each bakes its own constant."""

    def __init__(self, scale):
        self.scale = scale

    def __eq__(self, other):
        return isinstance(other, _ValueEqual)

    def __hash__(self):
        return 1

    def __call__(self, x, _data=None):
        return self.scale * jnp.sum(x**2)


def test_value_equal_callables_do_not_share_a_program():
    x = jnp.asarray([1.0, 2.0])
    first, second = _ValueEqual(1.0), _ValueEqual(3.0)
    assert float(curvature(first, x, None, want=("value",)).value) == 5.0
    assert float(curvature(second, x, None, want=("value",)).value) == 15.0
    assert float(curvature(second, x, None).hess[0, 0]) == 6.0
    assert float(curvature(first, x, None).hess[0, 0]) == 2.0


_EVENTS: list[float] = []
_COUNTING = {"on": False}


def _listener(event, duration, **_):
    if _COUNTING["on"] and event == "/jax/core/compile/backend_compile_duration":
        _EVENTS.append(duration)


jax.monitoring.register_event_duration_secs_listener(_listener)


def _compiles(thunk):
    _EVENTS.clear()
    _COUNTING["on"] = True
    try:
        jax.block_until_ready(thunk())
        return len(_EVENTS)
    finally:
        _COUNTING["on"] = False


def test_a_stable_function_compiles_once_and_a_fresh_closure_every_call():
    x = jnp.asarray([1.0, 2.0, 3.0])

    def stable(v, _data=None):
        return jnp.sum(v**4)

    curvature(stable, x, None)
    assert _compiles(lambda: curvature(stable, x, None)) == 0

    def run_fresh():
        return curvature(lambda v, _d=None: jnp.sum(v**4), x, None)

    run_fresh()
    assert _compiles(run_fresh) > 0, "a fresh closure per call recompiles: hoist it"


def test_collected_functions_release_their_programs():
    from tengri.inference import curvature as module

    before = len(module._PROGRAMS)

    def temporary(v, _data=None):
        return jnp.sum(v**2)

    curvature(temporary, jnp.ones(2), None)
    assert len(module._PROGRAMS) == before + 1
    del temporary
    gc.collect()
    assert len(module._PROGRAMS) == before
