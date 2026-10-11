# SPDX-License-Identifier: BSD-3-Clause
"""The X-ray column on the library sightline is evaluated only where it acts, exactly.

The MM83 photoelectric absorption of the torus sightline was evaluated on the whole
rest-frame grid on every call, and the band response of the X-ray emitter paid for it
(688,180 compiled FLOPs against a 500,000 budget). Only the samples shortward of the
library's grid take the column; every other sample keeps the library transmission. The
restricted evaluation must reproduce the dense one to rounding, values and gradient,
because the dense form is the reference the restriction is checked against.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import torus_library_transmission as tlt

pytestmark = pytest.mark.regression_bug

#: The rest-frame grid spans both sides of the library's 10 A edge and the MM83 table's
#: upper energy edge (1.24 A), so the restriction and the extrapolated tail are both hit.
WAVE = np.geomspace(0.5, 1.0e5, 5994)

#: Relative agreement required of the restricted path against the dense one.
RTOL = 1.0e-14


def _dense(cos_inc):
    """The dense reference: a traced grid takes the whole-grid absorption branch."""
    return jax.jit(
        lambda w: tlt.library_torus_sightline("skirtor", w, cos_inc=cos_inc, params={})
    )(jnp.asarray(WAVE))


@pytest.mark.parametrize("cos_inc", [0.1, 0.5, 0.95])
def test_restricted_absorption_matches_dense_values(cos_inc):
    restricted = tlt.library_torus_sightline("skirtor", WAVE, cos_inc=cos_inc, params={})
    dense = _dense(cos_inc)
    assert restricted is not None
    for got, want in (
        (restricted.transmission, dense.transmission),
        (restricted.disc_factor, dense.disc_factor),
    ):
        got, want = np.asarray(got), np.asarray(want)
        rel = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0e-300))
        assert rel <= RTOL, f"cos_i={cos_inc}: restricted differs from dense by {rel:.3e}"


def test_restricted_absorption_gradient_matches_dense():
    weight = jnp.cos(jnp.arange(WAVE.size) * 1.0e-3)

    def total(cos_inc, dense: bool):
        def inner(c, w):
            s = tlt.library_torus_sightline("skirtor", w, cos_inc=c, params={})
            return jnp.sum(weight * s.transmission + s.disc_factor)

        if dense:
            return jax.jit(inner)(cos_inc, jnp.asarray(WAVE))
        return inner(cos_inc, WAVE)

    for cos_inc in (0.1, 0.5, 0.95):
        g_restricted = jax.jit(jax.grad(lambda c: total(c, dense=False)))(cos_inc)
        g_dense = jax.jit(jax.grad(lambda c: total(c, dense=True)))(cos_inc)
        rel = abs(float(g_restricted) - float(g_dense)) / max(abs(float(g_dense)), 1.0e-300)
        assert rel <= RTOL, f"cos_i={cos_inc}: gradient differs by {rel:.3e}"


def _static(wave=WAVE):
    """The build-time absorption data of the concrete grid, as the component carries it."""
    index = tlt.below_library_index(wave, "skirtor")
    assert index is not None and index.below.any(), "the grid has no samples below the edge"
    return index


def _traced_static(cos_inc, index):
    """The static-index form, on a traced grid as the photometry kernel passes it."""
    return jax.jit(
        lambda w: tlt.library_torus_sightline(
            "skirtor", w, cos_inc=cos_inc, params={}, below_index=index
        )
    )(jnp.asarray(WAVE))


@pytest.mark.parametrize("cos_inc", [0.1, 0.5, 0.95])
def test_static_index_matches_dense_values_on_a_traced_grid(cos_inc):
    """The build-time index form, on a traced grid, reproduces the dense traced form."""
    static = _traced_static(cos_inc, _static())
    dense = _dense(cos_inc)
    for got, want in (
        (static.transmission, dense.transmission),
        (static.disc_factor, dense.disc_factor),
    ):
        got, want = np.asarray(got), np.asarray(want)
        assert np.all(np.isfinite(got))
        rel = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0e-300))
        assert rel <= RTOL, f"cos_i={cos_inc}: static index differs from dense by {rel:.3e}"


def test_static_index_gradient_matches_dense_on_a_traced_grid():
    weight = jnp.cos(jnp.arange(WAVE.size) * 1.0e-3)
    index = _static()

    def total(cos_inc, static: bool):
        def inner(c, w):
            kw = {"below_index": index} if static else {}
            s = tlt.library_torus_sightline("skirtor", w, cos_inc=c, params={}, **kw)
            return jnp.sum(weight * s.transmission + s.disc_factor)

        return jax.jit(inner)(cos_inc, jnp.asarray(WAVE))

    for cos_inc in (0.1, 0.5, 0.95):
        g_static = jax.jit(jax.grad(lambda c: total(c, static=True)))(cos_inc)
        g_dense = jax.jit(jax.grad(lambda c: total(c, static=False)))(cos_inc)
        rel = abs(float(g_static) - float(g_dense)) / max(abs(float(g_dense)), 1.0e-300)
        assert rel <= RTOL, f"cos_i={cos_inc}: static-index gradient differs by {rel:.3e}"


def test_index_for_another_grid_length_is_refused():
    """A build-time index handed to a grid of another length raises; it never falls through."""
    other = np.geomspace(0.5, 1.0e5, WAVE.size + 2)
    index = tlt.below_library_index(other, "skirtor")
    assert index.n_wave != WAVE.size
    with pytest.raises(ValueError, match="build-time below-library index"):
        _traced_static(0.5, index)
