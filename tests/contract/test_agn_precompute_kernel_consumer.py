# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for AGN disc/torus precompute kernel consumer integration.

Tests for the six wired adapters:
  1. powerlaw_disc
  2. ss_disc
  3. cigale_disc
  4. qsogen
  5. silva04
  6. cat3d_wind

Each adapter is tested to verify:
  - Lookup is JIT-compatible and returns finite values.
  - Lookup is grid-dimension aware.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.contract

_DATA = Path(__file__).resolve().parents[4] / "data"


@pytest.fixture(scope="module")
def simple_agn_filters():
    """Simple 3-filter set (UV, optical, IR) for AGN testing."""
    centers = np.array([2000, 5000, 30000])  # Angstrom
    widths = np.array([500, 1500, 5000])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 32)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


class TestPowerlawDiscPrecomputeConsumer:
    """Test powerlaw_disc adapter in hybrid kernel."""

    def test_smoke_lookup_jit_compatible(self, simple_agn_filters):
        """Lookup is JIT-compatible, returns finite values."""
        from tengri.components.agn import disc_precompute as adapter

        waves, trans = simple_agn_filters
        result = adapter.precompute(
            waves, trans, redshift=0.1, parameters=None, model="powerlaw_disc"
        )
        lookup = adapter.build_lookup(result, model="powerlaw_disc")

        # Test JIT compilation
        # powerlaw_disc has 1 axis: agn_alpha (power-law index). The adapter
        # declared this axis as ``agn_alpha_pl``, a name no Parameters can hold,
        # so it could never collapse (#1738).
        phot = assert_jit_matches_eager(lookup, jnp.float64(10.5), jnp.float64(-1.0))

        assert phot.shape == (len(waves),), f"Expected shape ({len(waves)},), got {phot.shape}"
        chex.assert_tree_all_finite(np.asarray(phot))
        assert np.all(np.asarray(phot) >= 0.0), "Lookup produced negative photometry"


class TestSSDiscPrecomputeConsumer:
    """Test ss_disc (Shakura-Sunyaev) adapter in hybrid kernel."""

    def test_smoke_lookup_jit_compatible(self, simple_agn_filters):
        """Lookup is JIT-compatible."""
        from tengri.components.agn import disc_precompute as adapter

        waves, trans = simple_agn_filters
        result = adapter.precompute(waves, trans, redshift=0.1, parameters=None, model="ss_disc")
        lookup = adapter.build_lookup(result, model="ss_disc")

        # ss_disc has 2 axes: agn_log_mbh, agn_log_lbol (#902)
        phot = assert_jit_matches_eager(
            lookup, jnp.float64(10.5), jnp.float64(8.0), jnp.float64(11.0)
        )

        assert phot.shape == (len(waves),), f"Expected shape ({len(waves)},), got {phot.shape}"
        chex.assert_tree_all_finite(np.asarray(phot))


class TestCigaleDiscPrecomputeConsumer:
    """Test cigale_disc adapter in hybrid kernel."""

    def test_smoke_lookup_jit_compatible(self, simple_agn_filters):
        """Lookup is JIT-compatible."""
        from tengri.components.agn import disc_precompute as adapter

        waves, trans = simple_agn_filters
        result = adapter.precompute(
            waves, trans, redshift=0.1, parameters=None, model="cigale_disc"
        )
        lookup = adapter.build_lookup(result, model="cigale_disc")

        # cigale_disc has no axes (pure scaling)
        phot = assert_jit_matches_eager(lookup, jnp.float64(10.5))

        # Note: cigale_disc may return 1D or 2D depending on precompute path
        assert phot.shape[-1] == len(waves), f"Expected last dim {len(waves)}, got {phot.shape}"
        chex.assert_tree_all_finite(np.asarray(phot))


class TestQSOgenPrecomputeConsumer:
    """Test qsogen adapter in hybrid kernel."""

    def test_smoke_lookup_jit_compatible(self, simple_agn_filters):
        """Lookup is JIT-compatible."""
        from tengri.components.agn import qsogen_precompute as adapter

        waves, trans = simple_agn_filters
        result = adapter.precompute(waves, trans, redshift=0.1, parameters=None)
        lookup = adapter.build_lookup(result)

        # Signature: (agn_log_lbol, *free_axes) for qsogen with 2 free axes
        phot = assert_jit_matches_eager(
            lookup, jnp.float64(10.5), jnp.float64(-0.35), jnp.float64(0.1)
        )

        assert phot.shape == (len(waves),), f"Expected shape ({len(waves)},), got {phot.shape}"
        chex.assert_tree_all_finite(np.asarray(phot))


class TestSilva04PrecomputeConsumer:
    """Test silva04 torus adapter in hybrid kernel."""

    @pytest.mark.skipif(
        not (_DATA / "silva04_torus_grid.h5").exists(),
        reason="Silva+04 torus grid not available; build via scripts/build_silva04_grid.py.",
    )
    def test_smoke_lookup_jit_compatible(self, simple_agn_filters):
        """Lookup is JIT-compatible."""
        from tengri.components.agn import silva04_precompute as adapter

        waves, trans = simple_agn_filters
        grid_path = str(_DATA / "silva04_torus_grid.h5")
        result = adapter.precompute(
            waves, trans, redshift=0.1, parameters=None, grid_path=grid_path
        )
        lookup = adapter.build_lookup(result)

        # Signature: (agn_log_lbol, *free_axes, agn_torus_frac=...)
        phot = assert_jit_matches_eager(
            lookup, jnp.float64(10.5), jnp.float64(21.5), agn_torus_frac=jnp.float64(0.5)
        )

        assert phot.shape == (len(waves),), f"Expected shape ({len(waves)},), got {phot.shape}"
        chex.assert_tree_all_finite(np.asarray(phot))


class TestCat3dPrecomputeConsumer:
    """Test cat3d_wind torus adapter in hybrid kernel."""

    @pytest.mark.skipif(
        not (_DATA / "cat3d_wind_torus_grid.h5").exists(),
        reason="CAT3D-Wind torus grid not available; build via scripts/build_cat3d_wind_grid.py.",
    )
    def test_smoke_lookup_jit_compatible(self, simple_agn_filters):
        """Lookup is JIT-compatible."""
        from tengri.components.agn import cat3d_precompute as adapter

        waves, trans = simple_agn_filters
        grid_path = str(_DATA / "cat3d_wind_torus_grid.h5")
        result = adapter.precompute(
            waves, trans, redshift=0.1, parameters=None, grid_path=grid_path
        )
        lookup = adapter.build_lookup(result)

        # Signature: (agn_log_lbol, *free_axes, agn_torus_frac=...)
        # cat3d has 3 axes: cos_inc, a, fwd
        phot = assert_jit_matches_eager(
            lookup,
            jnp.float64(10.5),
            jnp.float64(0.5),  # agn_cos_inc
            jnp.float64(0.5),  # agn_a_cat3d
            jnp.float64(0.3),  # agn_fwd_cat3d
            agn_torus_frac=jnp.float64(0.5),
        )

        assert phot.shape == (len(waves),), f"Expected shape ({len(waves)},), got {phot.shape}"
        chex.assert_tree_all_finite(np.asarray(phot))
