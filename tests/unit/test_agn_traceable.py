"""Test JIT-traced array threading for AGN precomputed models.

Verifies that grid arrays are properly threaded as JIT-traced kwargs
instead of being captured in closures, reducing XLA HLO size.
"""

from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp

from tengri.components.agn.skirtor_precompute import (
    build_skirtor_photometry_lookup,
)


class TestSKIRTORTracedArrays:
    """Test SKIRTOR precompute with grid_arrays_traced kwargs."""

    def test_build_lookup_with_traced_arrays(self):
        """Verify build_lookup accepts grid_arrays_traced kwarg."""
        # Create minimal mock precomp dict
        grid_phot_mock = jnp.ones((5, 5, 5, 5, 5, 3))  # 5D grid + 3 filters
        axes_mock = (
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # tau
            jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]),  # p
            jnp.array([0.2, 0.4, 0.6, 0.8, 1.0]),  # q
            jnp.array([10, 30, 50, 70, 90]),  # oa
            jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]),  # cos_inc
        )
        precomp = {
            "grid_phot": grid_phot_mock,
            "axes": axes_mock,
        }

        # Build with traced arrays
        grid_arrays_traced = (grid_phot_mock, axes_mock)
        lookup = build_skirtor_photometry_lookup(
            precomp, grid_arrays_traced=grid_arrays_traced
        )

        # Verify the lookup is callable
        assert callable(lookup)

        # Call it with mock parameters
        result = lookup(
            agn_log_lbol=44.0,
            agn_tau_skirtor=3.5,
            agn_p_skirtor=1.5,
            agn_q_skirtor=0.6,
            agn_oa_skirtor=50.0,
            agn_cos_inc=0.5,
            agn_torus_frac=0.5,
        )

        # Verify output shape (should match n_filters)
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_build_lookup_backward_compat(self):
        """Verify build_lookup still works without grid_arrays_traced (backward compat)."""
        # Create minimal mock precomp dict
        grid_phot_mock = jnp.ones((3, 3, 3, 3, 3, 2))
        axes_mock = (
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([0.5, 1.0, 1.5]),
            jnp.array([0.2, 0.4, 0.6]),
            jnp.array([10, 50, 90]),
            jnp.array([0.0, 0.5, 1.0]),
        )
        precomp = {
            "grid_phot": grid_phot_mock,
            "axes": axes_mock,
        }

        # Build WITHOUT traced arrays (default behavior)
        lookup = build_skirtor_photometry_lookup(precomp)

        # Verify the lookup is callable
        assert callable(lookup)

        # Call it with mock parameters
        result = lookup(
            agn_log_lbol=44.0,
            agn_tau_skirtor=2.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=0.4,
            agn_oa_skirtor=50.0,
            agn_cos_inc=0.5,
            agn_torus_frac=0.5,
        )

        # Verify output shape
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_traced_arrays_match_closure_capture(self):
        """Verify traced arrays produce same results as closure capture."""
        # Create matching mock precomp
        grid_phot_mock = jnp.ones((3, 3, 3, 3, 3, 2)) + 0.1
        axes_mock = (
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([0.5, 1.0, 1.5]),
            jnp.array([0.2, 0.4, 0.6]),
            jnp.array([10, 50, 90]),
            jnp.array([0.0, 0.5, 1.0]),
        )
        precomp = {
            "grid_phot": grid_phot_mock,
            "axes": axes_mock,
        }

        # Build both ways
        lookup_closure = build_skirtor_photometry_lookup(precomp)
        grid_arrays_traced = (grid_phot_mock, axes_mock)
        lookup_traced = build_skirtor_photometry_lookup(
            precomp, grid_arrays_traced=grid_arrays_traced
        )

        # Call with same parameters
        params = {
            "agn_log_lbol": 44.0,
            "agn_tau_skirtor": 2.0,
            "agn_p_skirtor": 1.0,
            "agn_q_skirtor": 0.4,
            "agn_oa_skirtor": 50.0,
            "agn_cos_inc": 0.5,
            "agn_torus_frac": 0.5,
        }

        result_closure = lookup_closure(**params)
        result_traced = lookup_traced(**params)

        # Results should be identical
        assert jnp.allclose(result_closure, result_traced, rtol=1e-6)
