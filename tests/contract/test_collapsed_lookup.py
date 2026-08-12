# SPDX-License-Identifier: BSD-3-Clause
"""One interpolation step behind nine collapsed photometry lookups (#1431).

Nine ``*_phot_collapsed`` closures each spelled the same middle step their own
way: two interpolation kernels, and a scalar-template guard that four of them
had and five did not. They now share
:func:`tengri.components._collapsed_lookup.interp_collapsed`.

The reference implementations below are the **original bodies, transcribed
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
from tengri.utils.grid_interp import interp_nd_pchip, interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid

pytestmark = pytest.mark.contract


# ── the original bodies, transcribed ─────────────────────────────


def _original_triweight(grid_phot, axes, edges, free_axis_values):
    """The guarded spelling: disc, qsogen, grahsp, dust."""
    if not axes:
        return grid_phot
    return interp_nd_triweight(grid_phot, axes, edges, tuple(free_axis_values))


def _original_triweight_unguarded(grid_phot, axes, edges, free_axis_values):
    """The unguarded spelling: silva04, skirtor."""
    return interp_nd_triweight(grid_phot, axes, edges, tuple(free_axis_values))


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
    edges = tuple(edges_for_grid(a) for a in axes)
    axes_j = tuple(jnp.asarray(a) for a in axes)
    point = tuple(jnp.asarray(float(a[1]) + 0.13) for a in axes)
    return grid_phot, axes_j, edges, point


N_AXES = [1, 2, 3]


# ── bit-exact agreement with the originals ───────────────────────


@pytest.mark.parametrize("n_axes", N_AXES)
def test_triweight_matches_the_original_body(n_axes):
    grid_phot, axes, edges, point = _grid(n_axes)
    got = interp_collapsed(grid_phot, axes, point, kernel="triweight", edges=edges)
    assert jnp.array_equal(got, _original_triweight(grid_phot, axes, edges, point))


@pytest.mark.parametrize("n_axes", N_AXES)
def test_triweight_matches_the_unguarded_original_body(n_axes):
    """The five that had no guard must be unaffected by gaining one."""
    grid_phot, axes, edges, point = _grid(n_axes, seed=1)
    got = interp_collapsed(grid_phot, axes, point, kernel="triweight", edges=edges)
    assert jnp.array_equal(got, _original_triweight_unguarded(grid_phot, axes, edges, point))


@pytest.mark.parametrize("n_axes", N_AXES)
def test_pchip_matches_the_original_body(n_axes):
    grid_phot, axes, _edges, point = _grid(n_axes, seed=2)
    got = interp_collapsed(grid_phot, axes, point, kernel="pchip")
    assert jnp.array_equal(got, _original_pchip(grid_phot, axes, point))


# ── the premise: an empty axis tuple is already a no-op ──────────


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_fully_collapsed_grid_returns_itself(kernel):
    """Why all nine can share one treatment though only four guarded it."""
    grid_phot = jnp.asarray([1.0, 2.0, 3.0])
    got = interp_collapsed(grid_phot, (), (), kernel=kernel, edges=())
    assert jnp.array_equal(got, grid_phot)


@pytest.mark.parametrize(
    ("kernel", "call"),
    [
        ("triweight", lambda g: interp_nd_triweight(g, (), (), ())),
        ("pchip", lambda g: interp_nd_pchip(g, (), ())),
    ],
)
def test_the_kernels_themselves_are_no_ops_when_fully_collapsed(kernel, call):
    """Pin it upstream too: if a kernel stops being a no-op, the guard matters."""
    grid_phot = jnp.asarray([1.0, 2.0, 3.0])
    assert jnp.array_equal(call(grid_phot), grid_phot)


# ── it stays usable where it is actually used ────────────────────


@pytest.mark.parametrize("kernel", KERNELS)
def test_compiles_to_the_same_result_as_the_original(kernel):
    """Every call site is inside a ``@jax.jit`` closure, so compile both and compare.

    Deliberately *not* jit-versus-eager. XLA fuses and reassociates, so the
    compiled result of this code differs from the eager one by ~1 ulp
    (measured: max_abs 8.9e-16) — with or without this refactor. Asserting
    jit == eager would be a test of XLA, and it would fail on unrefactored
    ``main`` too. The invariant that belongs to this change is
    jit(new) == jit(original), which is exact.
    """
    grid_phot, axes, edges, point = _grid(2, seed=3)
    kw = {"edges": edges} if kernel == "triweight" else {}

    def new(values):
        return interp_collapsed(grid_phot, axes, values, kernel=kernel, **kw)

    def original(values):
        if kernel == "pchip":
            return _original_pchip(grid_phot, axes, values)
        return _original_triweight(grid_phot, axes, edges, values)

    assert jnp.array_equal(jax.jit(new)(point), jax.jit(original)(point))


@pytest.mark.parametrize("kernel", KERNELS)
def test_gradient_matches_the_original(kernel):
    """These lookups sit under ``grad``; a changed derivative is a changed fit."""
    grid_phot, axes, edges, point = _grid(2, seed=4)

    if kernel == "triweight":

        def new(v):
            return interp_collapsed(grid_phot, axes, v, kernel=kernel, edges=edges).sum()

        def old(v):
            return _original_triweight(grid_phot, axes, edges, v).sum()
    else:

        def new(v):
            return interp_collapsed(grid_phot, axes, v, kernel=kernel).sum()

        def old(v):
            return _original_pchip(grid_phot, axes, v).sum()

    g_new = jax.grad(new)(point)
    g_old = jax.grad(old)(point)
    for a, b in zip(g_new, g_old, strict=True):
        assert jnp.array_equal(a, b)


# ── the two ways to misuse it fail loudly ────────────────────────


def test_an_unknown_kernel_raises():
    """Silently falling through to the other kernel would change the physics."""
    grid_phot, axes, edges, point = _grid(1)
    with pytest.raises(ValueError, match="unknown interpolation kernel"):
        interp_collapsed(grid_phot, axes, point, kernel="cubic", edges=edges)


def test_triweight_without_edges_raises():
    """Otherwise this fails deep inside the kernel, far from the cause."""
    grid_phot, axes, _edges, point = _grid(1)
    with pytest.raises(ValueError, match="needs bin edges"):
        interp_collapsed(grid_phot, axes, point, kernel="triweight")


def test_kernel_is_keyword_only():
    """A positional kernel would be easy to pass where ``edges`` belongs."""
    grid_phot, axes, edges, point = _grid(1)
    with pytest.raises(TypeError):
        interp_collapsed(grid_phot, axes, point, "triweight", edges)
