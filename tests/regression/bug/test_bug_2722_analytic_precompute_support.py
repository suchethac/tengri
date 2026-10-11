# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2722: the analytic dust precompute never clamps a reachable parameter.

``dust_analytic_precompute.precompute`` built its default node axes over the *declared* free prior
(dust_T 20-80 K, dust_beta_ir 1-2.5, dust_alpha_mir 1-3, dust_lambda_0_um 50-500 um). The lookup
holds the edge value beyond its nodes, with exactly zero gradient, so a widened prior
(``dust_T: Uniform(10, 120)``) or a pinned value outside the declared range (``Fixed(15.0)``) gave
a flat likelihood over the part the nodes did not reach, with nothing raised.

The default axes now span the declared prior extended to what the model can reach, at the declared
node density in the interpolation coordinate (ln for dust_T and dust_lambda_0_um); a supplied axis
that does not cover that reach raises. Every reference below is the exact closure band-averaged
in numpy (never a lookup), and the accuracy class is that pinned in
``tests/contract/test_dust_analytic_precompute_accuracy_2676.py`` (1e-3 in the far-IR bands).
"""

from __future__ import annotations

import functools
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Gaussian, Parameters, Uniform
from tengri.components.dust import dust_analytic_precompute as adapter
from tengri.components.dust.emission import DUST_EMISSION_MODELS as M
from tengri.config.exceptions import GridSupportWarning

pytestmark = pytest.mark.regression_bug

_MODELS = ("modified_blackbody", "casey2012", "graybody")
_WIDE_REST_AA = np.geomspace(1e2, 1e11, 72000)
_BANDS_UM = ((60.0, 90.0), (250.0, 500.0), (750.0, 950.0))
_ACCURACY = 1e-3  # the far-IR class pinned by #2676
_GRADIENT_RTOL = 1e-2
_FD_STEP = 5e-3  # relative central-difference step of the exact closure


def _tophat(lo_um, hi_um):
    inner = np.geomspace(lo_um * 1e4, hi_um * 1e4, 400)
    wave = np.concatenate([[inner[0] * (1 - 1e-6)], inner, [inner[-1] * (1 + 1e-6)]])
    return wave, np.concatenate([[0.0], np.ones_like(inner), [0.0]])


_FILTERS = tuple(_tophat(*band) for band in _BANDS_UM)


def _exact(model, query, band_index):
    """Exact closure band average over ``_BANDS_UM[band_index]`` at ``query`` (name -> value)."""
    fw, ft = _FILTERS[band_index]
    sed = np.asarray(M[model](jnp.asarray(_WIDE_REST_AA), 1.0, redshift=0.0, **query))
    return np.trapezoid(np.interp(fw, _WIDE_REST_AA, sed) * ft / fw, fw) / np.trapezoid(
        ft / fw, fw
    )


def _spec(model, **priors):
    return Parameters(dust_emission=model, **priors)


@functools.cache
def _built(model, name, bounds):
    """``(spec, lookup)`` for ``name`` ~ Uniform(*bounds), built through the public path."""
    spec = _spec(model, **{name: Uniform(*bounds)})
    result = adapter.precompute(
        [f[0] for f in _FILTERS], [f[1] for f in _FILTERS], 0.0, spec, model=model
    )
    return spec, adapter.build_lookup(result, model=model)


def _closure_query(spec, model, name, value):
    fixed = spec.get_fixed_values()
    return {n: value if n == name else fixed[n] for n in adapter.AXIS_PARAMS[model]}


_WIDENED = (
    ("dust_T", (10.0, 120.0), (11.0, 15.0, 100.0, 118.0)),
    ("dust_beta_ir", (0.5, 3.5), (0.6, 3.3)),
)
_WIDENED_CELLS = [
    pytest.param(model, name, bounds, value, id=f"{model}-{name}-{value:g}")
    for model in _MODELS
    for name, bounds, values in _WIDENED
    for value in values
]


@pytest.mark.parametrize(("model", "name", "bounds", "value"), _WIDENED_CELLS)
def test_widened_prior_lookup_equals_exact_closure(model, name, bounds, value):
    """Under a widened free prior the lookup equals the exact closure at every reachable value."""
    spec, lookup = _built(model, name, bounds)
    query = _closure_query(spec, model, name, value)
    got = np.asarray(lookup(1.0, value))
    for i, band in enumerate(_BANDS_UM):
        ratio = got[i] / _exact(model, query, i)
        assert ratio == pytest.approx(1.0, abs=_ACCURACY), (
            f"{model} {name}={value:g} band {band} um: lookup/exact = {ratio:.6f} "
            f"(the lookup holds the declared-range edge value beyond the nodes)"
        )


@pytest.mark.parametrize(("model", "name", "bounds", "value"), _WIDENED_CELLS)
def test_widened_prior_gradient_is_finite_nonzero_and_matches_exact(model, name, bounds, value):
    """The band-flux gradient is finite, non-zero and matches the exact closure's difference."""
    spec, lookup = _built(model, name, bounds)
    for i, band in enumerate(_BANDS_UM):
        grad = float(jax.grad(lambda v, i=i: lookup(1.0, v)[i])(jnp.asarray(value)))
        lo = _exact(model, _closure_query(spec, model, name, value * (1 - _FD_STEP)), i)
        hi = _exact(model, _closure_query(spec, model, name, value * (1 + _FD_STEP)), i)
        reference = (hi - lo) / (2 * _FD_STEP * value)
        assert np.isfinite(grad), f"{model} {name}={value:g} band {band}: gradient {grad}"
        assert grad != 0.0, f"{model} {name}={value:g} band {band}: gradient is exactly zero"
        assert grad == pytest.approx(reference, rel=_GRADIENT_RTOL), (
            f"{model} {name}={value:g} band {band}: grad {grad:.6e} vs exact {reference:.6e}"
        )


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("value", [15.0, 110.0])
def test_fixed_value_outside_declared_range_equals_exact_closure(model, value):
    """``dust_T: Fixed(v)`` with v outside 20-80 K pins the exact closure at v, not the edge."""
    spec = _spec(model, dust_T=Fixed(value))
    result = adapter.precompute(
        [f[0] for f in _FILTERS], [f[1] for f in _FILTERS], 0.0, spec, model=model
    )
    got = np.asarray(result["grid_phot"]).ravel()
    query = _closure_query(spec, model, "dust_T", value)
    free_axes = [n for n in adapter.AXIS_PARAMS[model] if n != "dust_T"]
    assert not free_axes or all(n in spec.get_fixed_values() for n in free_axes)
    for i, band in enumerate(_BANDS_UM):
        ratio = got[i] / _exact(model, query, i)
        assert ratio == pytest.approx(1.0, abs=_ACCURACY), (
            f"{model} Fixed(dust_T={value:g}) band {band} um: grid/exact = {ratio:.6f}"
        )


@pytest.mark.parametrize("model", _MODELS)
def test_user_axis_not_covering_a_widened_prior_raises(model):
    """A supplied 20-80 K axis under ``Uniform(10, 120)`` raises, naming the span and the reach."""
    spec = _spec(model, dust_T=Uniform(10.0, 120.0))
    with pytest.raises(ValueError, match=r"dust_T.*\[20, 80\].*\[10, 120\]"):
        adapter.precompute(
            [f[0] for f in _FILTERS],
            [f[1] for f in _FILTERS],
            0.0,
            spec,
            model=model,
            T_grid=np.geomspace(20.0, 80.0, 41),
        )


def test_user_axis_not_containing_a_fixed_value_raises():
    spec = _spec("modified_blackbody", dust_T=Fixed(15.0))
    with pytest.raises(ValueError, match=r"dust_T.*\[20, 80\].*\[15, 15\]"):
        adapter.precompute(
            [_FILTERS[1][0]],
            [_FILTERS[1][1]],
            0.0,
            spec,
            model="modified_blackbody",
            T_grid=np.geomspace(20.0, 80.0, 41),
        )


def test_user_axis_covering_the_support_is_used_as_given():
    spec = _spec("modified_blackbody", dust_T=Uniform(25.0, 60.0), dust_beta_ir=Uniform(1.2, 2.0))
    grid_t = np.geomspace(20.0, 80.0, 12)
    result = adapter.precompute(
        [_FILTERS[1][0]],
        [_FILTERS[1][1]],
        0.0,
        spec,
        model="modified_blackbody",
        T_grid=grid_t,
    )
    assert np.array_equal(np.asarray(result["_preint"].axes[0]), jnp.asarray(grid_t))


def _free_declared(model):
    return {n: Uniform(*adapter._get_param_bounds(n)) for n in adapter.AXIS_PARAMS[model]}


def _old_axis(name, n):
    lo, hi = adapter._get_param_bounds(name)
    if name in adapter._LOG_AXIS_PARAMS:
        return np.geomspace(lo, hi, n, dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


_AXIS_CELLS = [
    pytest.param(model, name, id=f"{model}-{name}")
    for model in _MODELS
    for name in adapter.AXIS_PARAMS[model]
]


@pytest.mark.parametrize(("model", "name"), _AXIS_CELLS)
def test_default_priors_build_the_declared_axes_bit_for_bit(model, name):
    """With no model, or priors equal to the declared ones, the axes are the old formula's."""
    n = adapter._DEFAULT_NODES[model][name]
    expected = _old_axis(name, n)
    assert np.array_equal(adapter._default_axis(name, n), expected)
    assert np.array_equal(adapter._default_axis(name, n, None), expected)
    lo, hi = adapter._get_param_bounds(name)
    assert np.array_equal(adapter._default_axis(name, n, (lo, hi)), expected)
    inside = (lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo))
    assert np.array_equal(adapter._default_axis(name, n, inside), expected)

    spec = _spec(model, **_free_declared(model))
    result = adapter.precompute([_FILTERS[1][0]], [_FILTERS[1][1]], 0.0, spec, model=model)
    index = adapter.AXIS_PARAMS[model].index(name)
    assert np.array_equal(np.asarray(result["axes"][index]), np.asarray(jnp.asarray(expected)))


def test_widened_temperature_axis_keeps_node_density_and_spans_the_prior():
    """``Uniform(10, 120)``: ceil(49 ln 12 / ln 4) = 88 geometric nodes over exactly 10-120 K."""
    spec = _spec(
        "modified_blackbody",
        **{**_free_declared("modified_blackbody"), "dust_T": Uniform(10.0, 120.0)},
    )
    result = adapter.precompute(
        [_FILTERS[1][0]], [_FILTERS[1][1]], 0.0, spec, model="modified_blackbody"
    )
    axis = np.asarray(result["_preint"].axes[0], dtype=np.float64)
    assert axis.size == 88
    assert axis[0] == 10.0
    assert axis[-1] == pytest.approx(120.0, rel=1e-12)
    ratios = axis[1:] / axis[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-6), "the temperature axis must stay geometric"


def test_fixed_value_beyond_the_declared_range_extends_the_default_axis_at_density():
    """``Fixed(200)``: geometric default axis over 20-200 K, ceil(49 ln 10 / ln 4) = 82 nodes."""
    axis = adapter._default_axis("dust_T", 49, (200.0, 200.0))
    assert axis.size == 82
    assert axis[0] == 20.0
    assert axis[-1] == pytest.approx(200.0, rel=1e-12)
    ratios = axis[1:] / axis[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-12)


@pytest.mark.parametrize(("model", "name"), _AXIS_CELLS)
def test_every_axis_widened_keeps_its_declared_density(model, name):
    """An axis widened to 1.65 times its declared span has ceil(1.65 n) nodes."""
    n = adapter._DEFAULT_NODES[model][name]
    lo, hi = adapter._get_param_bounds(name)
    log_axis = name in adapter._LOG_AXIS_PARAMS
    new_hi = hi * (hi / lo) ** 0.65 if log_axis else hi + 0.65 * (hi - lo)
    axis = adapter._default_axis(name, n, (lo, new_hi))
    assert axis.size == math.ceil(n * 1.65)
    assert axis[0] == lo
    assert axis[-1] == pytest.approx(new_hi, rel=1e-12)


def test_a_log_axis_reaching_zero_raises():
    with pytest.raises(ValueError, match="dust_T"):
        adapter._default_axis("dust_T", 49, (0.0, 80.0))


def test_unbounded_prior_warns_once_and_the_lookup_holds_the_edge_value():
    """A Gaussian dust_alpha_mir is unbounded: one warning, declared nodes, clamped edge."""
    spec = _spec("casey2012", dust_alpha_mir=Gaussian(2.0, 0.3))
    with pytest.warns(GridSupportWarning, match="dust_alpha_mir") as record:
        result = adapter.precompute(
            [_FILTERS[1][0]], [_FILTERS[1][1]], 0.0, spec, model="casey2012"
        )
    advisories = [w for w in record if issubclass(w.category, GridSupportWarning)]
    assert len(advisories) == 1, [str(w.message) for w in record]
    axis = np.asarray(result["axes"][0], dtype=np.float64)
    assert (axis[0], axis[-1]) == (1.0, 3.0)
    lookup = adapter.build_lookup(result, model="casey2012")
    edge = float(lookup(1.0, 3.0)[0])
    assert float(lookup(1.0, 3.5)[0]) == edge
    assert float(jax.grad(lambda v: lookup(1.0, v)[0])(jnp.asarray(3.5))) == 0.0


def test_user_axis_with_fewer_than_four_nodes_warns():
    with pytest.warns(UserWarning, match=r"dust_T has 3 nodes"):
        adapter.precompute(
            [_FILTERS[1][0]],
            [_FILTERS[1][1]],
            0.0,
            None,
            model="modified_blackbody",
            T_grid=np.array([20.0, 40.0, 80.0]),
        )


def test_no_warning_for_default_priors_or_bounded_widening():
    spec = _spec("modified_blackbody", dust_T=Uniform(10.0, 120.0), dust_beta_ir=Fixed(1.8))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter.precompute(
            [_FILTERS[1][0]], [_FILTERS[1][1]], 0.0, spec, model="modified_blackbody"
        )
