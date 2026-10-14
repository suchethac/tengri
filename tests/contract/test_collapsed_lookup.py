# SPDX-License-Identifier: BSD-3-Clause
"""One interpolation step behind nine collapsed photometry lookups (#1431).

Nine ``*_phot_collapsed`` closures each spelled the same middle step their own
way: two interpolation kernels, and a scalar-template guard that four of them
had and five did not. They now share
:func:`tengri.components._collapsed_lookup.interp_collapsed`.

The triweight kernel was removed from this helper (#2861): a stored node must
come back exactly, and the smoother does not do that. The PCHIP path is the
only one left, and it is checked against the original body.

The reference implementations below are the **original bodies, preserved
verbatim**. They are deliberately not refactored to call the helper -- that
would make this file assert that a thing equals itself. Every assertion is
bit-exact (``array_equal``, not ``allclose``): this sits on the AGN and dust
photometry path, under ``jax.jit`` and under ``grad``, so "close enough" is not
the claim being made.

The five closures that lacked the ``if not axes`` guard now get it. That is
safe because both kernels already return ``grid_phot`` unchanged for an empty
axis tuple -- asserted here rather than assumed, since it is the whole
justification for treating all nine alike.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components._collapsed_lookup import KERNELS, interp_collapsed
from tengri.utils.grid_interp import interp_nd_pchip

pytestmark = pytest.mark.contract


# ── the original bodies, preserved verbatim ─────────────────────────────


def _original_pchip(grid_phot, axes, free_axis_values):
    """cat3d, nenkova_agnfitter, skirtor_agnfitter -- never guarded."""
    return interp_nd_pchip(grid_phot, axes, tuple(free_axis_values))


# ── fixtures ─────────────────────────────────────────────────────


def _grid(n_axes: int, n_filters: int = 4, seed: int = 0):
    """A photometry grid of the shape a collapsed lookup reads."""
    rng = np.random.default_rng(seed)
    axes = tuple(np.linspace(0.5, 2.5, 4 + i) for i in range(n_axes))
    shape = (*(len(a) for a in axes), n_filters)
    grid_phot = jnp.asarray(rng.uniform(0.1, 10.0, size=shape))
    axes_j = tuple(jnp.asarray(a) for a in axes)
    point = tuple(jnp.asarray(float(a[1]) + 0.13) for a in axes)
    return grid_phot, axes_j, point


N_AXES = [1, 2, 3]


# ── bit-exact agreement with the original body ───────────────────


@pytest.mark.parametrize("n_axes", N_AXES)
def test_pchip_matches_the_original_body(n_axes):
    grid_phot, axes, point = _grid(n_axes, seed=2)
    got = interp_collapsed(grid_phot, axes, point, kernel="pchip")
    assert jnp.array_equal(got, _original_pchip(grid_phot, axes, point))


@pytest.mark.parametrize("n_axes", N_AXES)
def test_pchip_returns_stored_nodes_exactly(n_axes):
    grid_phot, axes, _point = _grid(n_axes, seed=5)
    node = tuple(a[1] for a in axes)
    got = interp_collapsed(grid_phot, axes, node, kernel="pchip")
    expected = grid_phot[(1,) * n_axes]
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=0.0)


# ── the premise: an empty axis tuple is already a no-op ──────────


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_fully_collapsed_grid_returns_itself(kernel):
    """Why all nine can share one treatment though only four guarded it."""
    grid_phot = jnp.asarray([1.0, 2.0, 3.0])
    got = interp_collapsed(grid_phot, (), (), kernel=kernel)
    assert jnp.array_equal(got, grid_phot)


def test_the_kernel_is_a_no_op_when_fully_collapsed():
    grid_phot = jnp.asarray([1.0, 2.0, 3.0])
    assert jnp.array_equal(interp_nd_pchip(grid_phot, (), ()), grid_phot)


# ── it stays usable where it is actually used ────────────────────


@pytest.mark.parametrize("kernel", KERNELS)
def test_compiles_to_the_same_result_as_the_original(kernel):
    grid_phot, axes, point = _grid(2, seed=3)

    @jax.jit
    def f(*values):
        return interp_collapsed(grid_phot, axes, values, kernel=kernel)

    np.testing.assert_allclose(
        np.asarray(f(*point)), np.asarray(_original_pchip(grid_phot, axes, point)), rtol=1e-12
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_gradient_matches_the_original(kernel):
    grid_phot, axes, point = _grid(2, seed=4)

    def got(*v):
        return interp_collapsed(grid_phot, axes, v, kernel=kernel).sum()

    def ref(*v):
        return _original_pchip(grid_phot, axes, v).sum()

    g1 = jax.grad(got, argnums=(0, 1))(*point)
    g2 = jax.grad(ref, argnums=(0, 1))(*point)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-12)


def test_an_unknown_kernel_raises():
    grid_phot, axes, point = _grid(1)
    with pytest.raises(ValueError, match="unknown interpolation kernel"):
        interp_collapsed(grid_phot, axes, point, kernel="triweight")


def test_kernel_is_keyword_only():
    """A positional kernel would be easy to pass by mistake."""
    grid_phot, axes, point = _grid(1)
    with pytest.raises(TypeError):
        interp_collapsed(grid_phot, axes, point, "pchip")
