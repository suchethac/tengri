# SPDX-License-Identifier: BSD-3-Clause
"""Mathematical-contract tests for ``interp_nd_pchip``.

The node-exact monotone-cubic primitive (Fritsch & Carlson 1980) backs the
CAT3D-Wind torus interpolation. These tests pin its three guarantees: it
reproduces nodes exactly, it matches SciPy's reference ``PchipInterpolator``
between nodes, it does not overshoot on step data, and it is C¹-differentiable.

References
----------
.. [1] F. N. Fritsch & R. E. Carlson, "Monotone Piecewise Cubic
   Interpolation," SIAM J. Numer. Anal. 17, 238 (1980). DOI: 10.1137/0717021.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tengri.utils.grid_interp import interp_nd_pchip


@pytest.mark.regression_paper
def test_matches_scipy_pchip_1d():
    """1-D interpolation matches SciPy's PchipInterpolator within 1e-9."""
    x = jnp.asarray([0.0, 1.0, 2.5, 4.0, 7.0, 10.0])
    y = jnp.asarray([1.0, 3.0, 2.0, 5.0, 4.5, 6.0])
    ref = PchipInterpolator(np.asarray(x), np.asarray(y))
    for xq in (0.3, 1.0, 2.0, 3.7, 6.2, 9.9):
        got = float(interp_nd_pchip(y[:, None], (x,), (xq,))[0])
        assert abs(got - float(ref(xq))) < 1e-9, f"xq={xq}: {got} vs {ref(xq)}"


@pytest.mark.limit
def test_node_exact_nd():
    """At every tabulated node the interpolant returns the stored value exactly."""
    x0 = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    x1 = jnp.asarray([-2.0, -1.0, 0.5, 2.0])
    rng = np.random.default_rng(0)
    grid = jnp.asarray(rng.normal(size=(4, 4, 5)))  # trailing dim preserved
    for i, a in enumerate(x0):
        for j, b in enumerate(x1):
            got = interp_nd_pchip(grid, (x0, x1), (float(a), float(b)))
            np.testing.assert_allclose(np.asarray(got), np.asarray(grid[i, j]), atol=1e-12)


@pytest.mark.bounds
def test_no_overshoot_on_step():
    """Monotone (shape-preserving): a step in the data is not overshot."""
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0])  # nearest-neighbor-style step
    lo, hi = float(y.min()), float(y.max())
    for xq in np.linspace(0.0, 4.0, 101):
        v = float(interp_nd_pchip(y[:, None], (x,), (float(xq),))[0])
        assert lo - 1e-9 <= v <= hi + 1e-9, f"overshoot at xq={xq}: {v}"


@pytest.mark.gradient
def test_gradient_finite_and_continuous():
    """jax.grad through the interpolant is finite across and at cell boundaries."""
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    y = jnp.asarray([1.0, 4.0, 2.0, 5.0])

    def f(xq):
        return interp_nd_pchip(y[:, None], (x,), (xq,))[0]

    g = jax.jit(jax.grad(f))
    for xq in (0.0, 0.5, 1.0, 1.999, 2.0, 2.5, 3.0):
        assert jnp.isfinite(g(xq)), f"non-finite gradient at xq={xq}"
        assert jnp.any(g(xq) != 0.0), (
            "`g(xq)` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


@pytest.mark.regression_bug
def test_gradient_finite_on_symmetric_peak_node_values():
    """Gradient w.r.t. node values is finite for non-monotonic (symmetric-peak)
    data (#892).

    ``_pchip_slopes`` takes a weighted harmonic mean of the two bracketing
    secants, used only where they share a sign. For a symmetric peak the
    secants are equal and opposite, so the harmonic denominator
    ``w1/d_left + w2/d_right`` passes through 0 and the (discarded) harmonic
    value diverges — the outer ``where`` then forms ``0 * inf = NaN`` in the
    VJP. This is exactly the trap that made the SKIRTOR torus grid-axis params
    (``agn_p/q/tau_skirtor``, whose dust template is non-monotonic in λ)
    non-differentiable inside ``skirtor_disc_dust_ratio``. The forward value is
    unaffected — only the reverse pass was poisoned.
    """
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0])

    def f(y):
        return jnp.sum(interp_nd_pchip(y[:, None], (x,), (1.4,)))

    for y in (
        jnp.asarray([0.0, 1.0, 0.0, 1.0]),  # symmetric peak: secants +1, -1
        jnp.asarray([1.0, 2.0, 1.0, 2.0]),
    ):
        grad = jax.grad(f)(y)
        assert jnp.all(jnp.isfinite(grad)), f"non-finite VJP on non-monotonic data: {grad}"
        assert jnp.any(grad != 0.0), (
            "`grad` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
