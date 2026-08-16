# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for ``pchip_interp_1d`` — the single 1-D PCHIP in the package.

Three separate PCHIP implementations used to coexist: the correct one in
``utils/grid_interp``, plus private copies in ``sfh/dense_basis.py`` and
``sfh/mean_sfh.py``. Both copies were defective, and both are now deleted in
favor of this one. The regression tests below pin the three failures so a
re-fork cannot reintroduce them:

1. ``dense_basis`` took the **unweighted** harmonic mean of the bracketing
   secants. That equals the Fritsch-Carlson form only on a uniform grid, and
   the quantile times it was applied to are never uniform — measured 2.7 %
   deviation from SciPy.
2. That same copy guarded the denominator with ``jnp.maximum(d_l + d_r, 1e-30)``,
   which fails **open** on decreasing data: both secants negative, their sum
   negative, clamped up to +1e-30, slope explodes. Measured range ±6.4e28
   against SciPy's [0, 1].
3. ``mean_sfh`` gated only the division *output*, not its *inputs*, so a
   symmetric zig-zag on exactly uniform log-spacing sent the denominator
   through zero. The discarded ``inf`` then formed ``0 * inf = NaN`` in the
   reverse pass (the #892 trap).

References
----------
.. [1] F. N. Fritsch & R. E. Carlson, "Monotone Piecewise Cubic
   Interpolation," SIAM J. Numer. Anal. 17, 238 (1980). DOI: 10.1137/0717021.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import PchipInterpolator

from tengri.utils.grid_interp import pchip_interp_1d
from tests._grad_parity import assert_grad_matches_fd

# ── Regressions for the two deleted copies ────────────────────────


@pytest.mark.regression_bug
def test_non_uniform_nodes_match_scipy():
    """Weighted F-C slopes: exact on a NON-uniform grid, where the bug lived.

    The unweighted harmonic mean agrees with Fritsch-Carlson only when the
    nodes are equally spaced, so a uniform grid cannot detect the defect.
    These are realistic dense_basis quantile times.
    """
    x = np.array([0.0, 0.08, 0.21, 0.39, 0.62, 0.85, 1.0])
    y = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    xq = np.linspace(0.0, 1.0, 501)
    got = np.asarray(pchip_interp_1d(jnp.asarray(x), jnp.asarray(y), jnp.asarray(xq)))
    assert_allclose(got, PchipInterpolator(x, y)(xq), atol=1e-12, rtol=0)


@pytest.mark.regression_bug
def test_decreasing_data_does_not_explode():
    """A fail-open floor sent decreasing data to ±6e28; it must track SciPy."""
    x = np.linspace(0.0, 1.0, 7)
    y = np.array([1.0, 0.80, 0.62, 0.47, 0.30, 0.14, 0.0])  # strictly decreasing
    xq = np.linspace(0.0, 1.0, 501)
    got = np.asarray(pchip_interp_1d(jnp.asarray(x), jnp.asarray(y), jnp.asarray(xq)))
    assert_allclose(got, PchipInterpolator(x, y)(xq), atol=1e-12, rtol=0)
    assert got.max() <= 1.0 + 1e-12
    assert got.min() >= -1e-12


@pytest.mark.gradient
def test_zigzag_on_uniform_spacing_has_finite_vjp():
    """Double-``where``: non-monotonic data on uniform spacing must not NaN.

    Exact decades give ``h`` identically uniform, which is what drove the old
    single-``where`` denominator through zero.
    """
    x = jnp.log10(jnp.array([1e6, 1e7, 1e8, 1e9, 1e10]))
    y = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0])  # symmetric zig-zag
    xq = jnp.linspace(6.0, 10.0, 32)

    g = assert_grad_matches_fd(lambda yv: jnp.sum(pchip_interp_1d(x, yv, xq) ** 2), y)
    chex.assert_tree_all_finite(g)


# ── Core interpolation contract ───────────────────────────────────


@pytest.mark.regression_paper
def test_node_exact():
    """Interpolation reproduces every tabulated node exactly."""
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert_allclose(np.asarray(pchip_interp_1d(x, y, x)), np.asarray(y), atol=1e-14)


@pytest.mark.bounds
def test_monotone_input_gives_monotone_output():
    x = jnp.linspace(0.0, 1.0, 10)
    y = x**3
    out = pchip_interp_1d(x, y, jnp.linspace(0.05, 0.95, 50))
    assert jnp.all(jnp.diff(out) >= -1e-12)


@pytest.mark.limit
def test_linear_data_is_reproduced_exactly():
    x = jnp.linspace(0.0, 1.0, 5)
    y = 2.0 * x + 1.0
    xq = jnp.linspace(0.1, 0.9, 30)
    assert_allclose(np.asarray(pchip_interp_1d(x, y, xq)), np.asarray(2.0 * xq + 1.0), atol=1e-12)


@pytest.mark.limit
def test_constant_data_stays_constant():
    x = jnp.linspace(0.0, 1.0, 6)
    y = jnp.full(6, 3.14)
    out = pchip_interp_1d(x, y, jnp.linspace(0.1, 0.9, 20))
    assert_allclose(np.asarray(out), 3.14, atol=1e-12)


# ── Extrapolation switch ──────────────────────────────────────────


@pytest.mark.bounds
def test_clamps_outside_the_table_by_default():
    """Default holds the boundary node — no cubic runaway off the ends."""
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.asarray(x)
    out = pchip_interp_1d(x, y, jnp.array([-0.5, 0.0, 1.0, 1.5]))
    chex.assert_tree_all_finite(out)
    assert_allclose(np.asarray(out), np.array([0.0, 0.0, 1.0, 1.0]), atol=1e-12)


@pytest.mark.contract
def test_extrapolate_continues_the_edge_cubic():
    """``extrapolate=True`` must leave the table, not clamp — ProSpect needs it."""
    x = jnp.linspace(0.0, 1.0, 5)
    y = 2.0 * x  # slope 2 everywhere
    out = pchip_interp_1d(x, y, jnp.array([-0.5, 1.5]), extrapolate=True)
    # A straight line extrapolates to itself.
    assert_allclose(np.asarray(out), np.array([-1.0, 3.0]), atol=1e-12)


# ── JAX transforms ────────────────────────────────────────────────


@pytest.mark.gradient
def test_gradients_finite_and_jit_vmap_safe():
    x = jnp.linspace(0.0, 1.0, 5)
    xq = jnp.linspace(0.1, 0.9, 10)

    g = jax.grad(lambda y: jnp.sum(pchip_interp_1d(x, y, xq) ** 2))(jnp.linspace(0.0, 1.0, 5))
    chex.assert_tree_all_finite(g)

    y = jnp.linspace(0.0, 1.0, 5)
    eager = pchip_interp_1d(x, y, xq)
    jitted = jax.jit(pchip_interp_1d)(x, y, xq)
    chex.assert_trees_all_close(eager, jitted, rtol=1e-12)

    batched = jax.vmap(lambda yy: pchip_interp_1d(x, yy, xq))(jnp.stack([y, 2.0 * y]))
    chex.assert_shape(batched, (2, xq.shape[0]))
    chex.assert_trees_all_close(batched[0], eager, rtol=1e-12)
