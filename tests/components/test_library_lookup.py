# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``library_lookup``, the one entry point for template-library lookups (#2861).

Owner ruling: a library template lookup is node-exact PCHIP. Each query returns
the stored node value exactly, the interpolant is shape-preserving, a numerical
overhang (prior past the node hull by less than half an end spacing) is
extrapolated with a nonzero gradient, and a physical overhang is held at the
edge of that band (never ``jnp.clip``, which halves the boundary gradient).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import interp_nd_pchip, library_lookup

pytestmark = pytest.mark.contract


def _grid_1d():
    x = jnp.asarray([0.0, 1.0, 2.0, 4.0, 8.0])
    y = jnp.asarray([0.0, 1.0, 1.5, 1.6, 4.0])
    return x, y


def _grid_2d():
    x = jnp.asarray([0.0, 0.5, 1.5, 3.0])
    z = jnp.asarray([-1.0, 0.0, 2.0])
    values = jnp.asarray(np.sin(np.add.outer(np.asarray(x), 2.0 * np.asarray(z))) ** 2 + 0.3)
    return (x, z), values


def test_node_values_are_exact_1d():
    x, y = _grid_1d()
    for k in range(x.shape[0]):
        got = library_lookup(y, (x,), (x[k],))
        assert float(got) == float(y[k])


def test_node_values_are_exact_2d():
    (x, z), values = _grid_2d()
    for i in range(x.shape[0]):
        for j in range(z.shape[0]):
            got = library_lookup(values, (x, z), (x[i], z[j]))
            assert float(got) == float(values[i, j])


def test_defaults_to_pchip_between_nodes():
    x, y = _grid_1d()
    point = (jnp.asarray(1.7),)
    np.testing.assert_allclose(
        float(library_lookup(y, (x,), point)),
        float(interp_nd_pchip(y, (x,), point)),
        rtol=0,
        atol=1e-14,
    )


def test_gradient_nonzero_across_numerical_overhang():
    x, y = _grid_1d()
    spacing = float(x[-1] - x[-2])
    overhang = 0.4 * spacing  # less than half an end spacing: numerical gap
    grad = jax.grad(lambda p: library_lookup(y, (x,), (p,))[()])(x[-1] + overhang)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


def test_physical_overhang_is_held_with_zero_gradient():
    x, y = _grid_1d()
    spacing = float(x[-1] - x[-2])
    beyond = float(x[-1]) + 1.0 * spacing  # more than half an end spacing: physical gap
    value = library_lookup(y, (x,), (beyond,))
    grad = jax.grad(lambda p: library_lookup(y, (x,), (p,))[()])(beyond)
    assert float(value) == float(y[-1])
    assert float(grad) == 0.0


def test_boundary_gradient_not_halved():
    x, y = _grid_1d()
    # Derivative at the outermost node equals the one-sided derivative from inside.
    h = 1e-6
    inside = (float(x[-1]) - h, float(x[-1]))
    slope_inside = (
        float(library_lookup(y, (x,), (inside[1],))) - float(library_lookup(y, (x,), (inside[0],)))
    ) / h
    grad = jax.grad(lambda p: library_lookup(y, (x,), (p,))[()])(float(x[-1]))
    np.testing.assert_allclose(float(grad), slope_inside, rtol=1e-4)


def test_extrapolate_flags_length_checked():
    x, y = _grid_1d()
    try:
        library_lookup(y, (x,), (1.0,), extrapolate=(True, False))
    except ValueError:
        return
    raise AssertionError("extrapolate with the wrong length must raise ValueError")
