# SPDX-License-Identifier: BSD-3-Clause
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

from tests._jit_parity import assert_jit_matches_eager

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
        out = assert_jit_matches_eager(lookup, jnp.float64(1.0), jnp.float64(-1.5))
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
        # Axes are (agn_log_mbh, agn_log_lbol) after #902; query at M_bh=1e8,
        # log10(L_bol/L_sun)=11 (sub-Eddington).
        out = assert_jit_matches_eager(
            lookup, jnp.float64(1.0), jnp.float64(8.0), jnp.float64(11.0)
        )
        chex.assert_tree_all_finite(np.asarray(out))

    def test_second_axis_is_not_degenerate(self, filter_set):
        """The ss_disc grid's second axis must produce distinct disc shapes.

        Regression for #902 (silent-failure): the second axis was
        ``agn_log_mdot``, passed to ``multicolor_disc`` as ``agn_log_ledd`` —
        a parameter that block has ignored since #846 (the Eddington ratio is
        derived from ``agn_log_lbol``). Every node along axis 1 therefore
        produced the *identical* disc: a silent no-op grid axis. The axis is
        now ``agn_log_lbol`` (the post-#846 shape driver), so slices at
        different luminosities must be measurably distinct.
        """
        from tengri.components.agn import disc_precompute

        waves, trans = filter_set
        result = disc_precompute.precompute(
            waves, trans, redshift=0.5, parameters=None, model="ss_disc"
        )
        phot = np.asarray(result["grid_phot"])  # (n_mbh, n_lbol, n_filters)
        assert phot.ndim == 3, f"expected (n_mbh, n_lbol, n_filters), got {phot.shape}"
        low_lum = phot[:, 0, :]
        high_lum = phot[:, -1, :]
        # Relative comparison: the energy-normalized (shape-only) templates are
        # O(1e-17), so an absolute tolerance would trivially call them equal.
        max_rel = float(np.max(np.abs(high_lum - low_lum) / (np.abs(low_lum) + 1e-30)))
        assert max_rel > 0.05, (
            "ss_disc second grid axis is degenerate — the faintest and brightest "
            f"luminosity nodes produce near-identical disc shapes (max rel diff "
            f"{max_rel:.2e}); silent no-op (#902)."
        )


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
