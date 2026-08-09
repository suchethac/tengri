# SPDX-License-Identifier: BSD-3-Clause
"""Threading the nthcomp table must not perturb the value or the gradient.

The nthcomp Comptonization table (~15 MB) used to be read from a module-level
cache inside :func:`nthcomp_lnu_interp`, which froze it into every graph that
touched a Comptonized disc. It is now passed in as a pytree argument.

That required restructuring a :func:`jax.custom_vjp`: the table became primal
argument 0, it rides in the residuals (the backward pass re-evaluates the
interpolation at ``gamma + eps``), and the backward pass returns one extra,
structurally-zero cotangent for it.

Changing the arity and residuals of a gradient-carrying primitive can alter
gradients silently, so this pins **both** the value and ``d/dgamma`` — and
asserts the gradient is non-trivial, since "identical" proves nothing about
two zeros.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.regression_bug]

_GAMMAS = (1.6, 2.0, 2.4, 3.0)


def _paths():
    from tengri.components.agn._nthcomp import (
        _TABLE_AVAILABLE,
        load_nthcomp_table,
        nthcomp_lnu_interp,
    )

    if not _TABLE_AVAILABLE:
        pytest.skip("nthcomp templates not available")

    nu = jnp.logspace(14.0, 18.0, 256)
    table = load_nthcomp_table()

    def cache_path(g):
        return jnp.sum(nthcomp_lnu_interp(nu, g, 100.0, 0.01))

    def threaded_path(g):
        return jnp.sum(nthcomp_lnu_interp(nu, g, 100.0, 0.01, _template=table))

    return cache_path, threaded_path


@pytest.mark.parametrize("gamma", _GAMMAS)
def test_threaded_table_matches_cache_value_and_gradient(gamma):
    """Both paths agree exactly, in value and in d/dgamma."""
    cache_path, threaded_path = _paths()
    g = jnp.asarray(gamma)

    assert float(cache_path(g)) == float(threaded_path(g))
    assert float(jax.grad(cache_path)(g)) == float(jax.grad(threaded_path)(g))


def test_gamma_gradient_is_finite_and_nonzero():
    """Guard the comparison above against being a pair of zeros.

    The custom VJP deliberately returns zero for nu / kTe / kTbb, so a
    gradient check that happened to land on an all-zero result would pass
    while proving nothing.
    """
    _, threaded_path = _paths()
    grads = [float(jax.grad(threaded_path)(jnp.asarray(x))) for x in _GAMMAS]

    assert all(jnp.isfinite(jnp.asarray(x)) for x in grads), grads
    assert any(abs(x) > 0.0 for x in grads), f"all gradients zero: {grads}"
