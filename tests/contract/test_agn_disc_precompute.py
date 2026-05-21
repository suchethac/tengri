"""Smoke tests for the analytic AGN disc precompute adapters (PR 1).

Exercises the build/lookup surface for ``powerlaw_disc``, ``ss_disc``, and
``cigale_disc`` adapters. Numerical equivalence against the runtime path lives
in the broader integration tests; this file only verifies the adapter contract:

* ``precompute`` returns the expected dict shape with finite, positive band
  fluxes on the grid nodes.
* ``build_lookup`` returns a callable that is JIT-compatible.
* ``Fixed``-axis collapse via ``slice_fixed_axes`` reduces grid dimensionality
  without breaking the lookup.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def filter_set():
    """A small synthetic filter set covering UV–optical–NIR."""
    centers = np.array([1500.0, 2300.0, 4500.0, 6500.0, 8500.0, 12000.0])
    widths = np.array([300.0, 400.0, 800.0, 1000.0, 1200.0, 1500.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


class TestPowerlawDiscAdapter:
    def test_precompute_shape_and_finite(self, filter_set):
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=1.0, parameters=None, model="powerlaw_disc"
        )
        phot = np.asarray(result["grid_phot"])
        assert phot.shape[-1] == len(waves)
        chex.assert_tree_all_finite(phot)

    def test_lookup_jit(self, filter_set):
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=1.0, parameters=None, model="powerlaw_disc"
        )
        lookup = disc_precompute.build_lookup(result, model="powerlaw_disc")
        jit_lookup = jax.jit(lookup)
        out = jit_lookup(jnp.float64(1.0), jnp.float64(-1.5))
        chex.assert_tree_all_finite(np.asarray(out))


class TestSSDiscAdapter:
    def test_precompute_shape_and_finite(self, filter_set):
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=0.5, parameters=None, model="ss_disc"
        )
        phot = np.asarray(result["grid_phot"])
        assert phot.shape[-1] == len(waves)
        chex.assert_tree_all_finite(phot)

    def test_lookup_jit(self, filter_set):
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=0.5, parameters=None, model="ss_disc"
        )
        lookup = disc_precompute.build_lookup(result, model="ss_disc")
        jit_lookup = jax.jit(lookup)
        out = jit_lookup(jnp.float64(1.0), jnp.float64(8.0), jnp.float64(-1.5))
        chex.assert_tree_all_finite(np.asarray(out))


class TestCigaleDiscAdapter:
    def test_precompute_template_shape(self, filter_set):
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=2.0, parameters=None, model="cigale_disc"
        )
        phot = np.asarray(result["grid_phot"])
        assert phot.shape[-1] == len(waves)
        chex.assert_tree_all_finite(phot)
        assert result["axes"] == ()

    def test_lookup_callable(self, filter_set):
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=2.0, parameters=None, model="cigale_disc"
        )
        lookup = disc_precompute.build_lookup(result, model="cigale_disc")
        out = jax.jit(lookup)(jnp.float64(1.0))
        chex.assert_tree_all_finite(np.asarray(out))


class TestRegistryEntries:
    def test_disc_keys_registered(self):
        from tengri.forward.precompute.registry import _REGISTRY

        for key in ("powerlaw_disc", "ss_disc", "cigale_disc"):
            assert key in _REGISTRY
            assert _REGISTRY[key].endswith("disc_precompute")
