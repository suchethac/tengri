# SPDX-License-Identifier: BSD-3-Clause
"""CB19 grid-fraction interpolation must differentiate through a duplicate node (#1492).

``_frac_idx`` divides by the bracketing interval ``dx = grid[idx+1] - grid[idx]``.
Where ``dx == 0`` (a repeated grid node at the top end, reached when the value is
clipped onto it), the forward ``jnp.where`` selects 0, but a single ``where`` lets
the untaken ``(val - grid) / 0`` carry a NaN cotangent back to ``val``. The
double-``where`` form keeps the reverse pass finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_frac_idx_forward_value_at_duplicate_top_node():
    """A duplicate top node gives the node value with zero fractional part."""
    from tengri.components.nebular.cloudy_cb19 import _frac_idx

    grid = jnp.array([0.0, 1.0, 2.0, 2.0])
    assert float(_frac_idx(2.0, grid)) == pytest.approx(2.0)


def test_frac_idx_grad_on_regular_bracket_is_unit_slope():
    """Inside a positive-width bracket the fraction is linear: d(idx)/dv = 1/dx = 1."""
    from tengri.components.nebular.cloudy_cb19 import _frac_idx

    grid = jnp.array([0.0, 1.0, 2.0, 2.0])
    g = jax.grad(lambda v: _frac_idx(v, grid))(jnp.float64(0.5))
    assert np.isfinite(float(g))
    assert float(g) != 0.0
    assert float(g) == pytest.approx(1.0)


def test_frac_idx_grad_is_finite_on_duplicate_top_node():
    """The gradient through a zero-width bracket is finite, not NaN.

    The value is clipped onto the duplicated top node, and ``jnp.clip`` passes no
    cotangent past its bound, so the gradient is exactly zero: the value is saturated.
    """
    from tengri.components.nebular.cloudy_cb19 import _frac_idx

    grid = jnp.array([0.0, 1.0, 2.0, 2.0])
    g = jax.grad(lambda v: _frac_idx(v, grid))(jnp.float64(2.0))
    assert np.isfinite(float(g))
    assert float(g) == 0.0
