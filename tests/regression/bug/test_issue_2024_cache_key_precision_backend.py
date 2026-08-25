# SPDX-License-Identifier: BSD-3-Clause
"""Cache keys include session precision and JAX backend (#2024).

The precompute z-table and IGM transmission caches store values computed through
JAX (luminosity_distance, igm_transmission) that inherit session precision. A
float32 session writing a cache entry leaves values ~1e-7 in relative error
compared to float64 — silently contaminating precision benchmarks or parity tests
that share a cache directory between arms.

The fix: include bool(jax.config.jax_enable_x64) and jax.default_backend() in
all three cache key computations. These tests verify that the keys actually
change when x64 and backend change, preventing mutations that remove the new
tuple entries from passing.
"""

import numpy as np
import pytest

from tengri.components.igm import _subband_cache as sc
from tengri.components.stellar.sps import precompute as pc

pytestmark = pytest.mark.regression_bug


# ── Regression: keys CHANGE when x64 flips ────────────────────────────────────


def test_ztable_cache_key_changes_with_x64():
    """_ztable_cache_key differs when jax_enable_x64 flips.

    Uses jax.config.update to flip x64, compute the key, restore original
    state, and verify the two keys differ. No arrays cross the boundary,
    so nothing persists post-test.
    """
    import jax

    # Create minimal test inputs
    class MockSSPData:
        ssp_wave = np.array([100.0, 200.0, 300.0])
        ssp_flux = np.ones((1, 1, 3))

    ssp_data = MockSSPData()
    filter_waves = [np.array([100.0, 200.0])]
    filter_trans = [np.array([0.5, 0.8])]
    z_grid = np.array([0.1, 0.5, 1.0])
    apply_igm = False
    taylor_correction = False
    convention = "bessell"
    n_subbands = 0

    # Save original x64 state and flip it
    original_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        key_with_x64_true = pc._ztable_cache_key(
            ssp_data,
            filter_waves,
            filter_trans,
            z_grid,
            apply_igm,
            taylor_correction,
            convention,
            n_subbands,
        )

        jax.config.update("jax_enable_x64", False)
        key_with_x64_false = pc._ztable_cache_key(
            ssp_data,
            filter_waves,
            filter_trans,
            z_grid,
            apply_igm,
            taylor_correction,
            convention,
            n_subbands,
        )
    finally:
        jax.config.update("jax_enable_x64", original_x64)

    assert key_with_x64_true != key_with_x64_false, (
        "Keys must differ when x64 changes; fix may be reverted"
    )


def test_subband_cache_key_changes_with_x64():
    """cache_key differs when jax_enable_x64 flips."""
    import jax

    rng = np.random.default_rng(0)
    base = {
        "waves_rest": rng.uniform(1000.0, 9000.0, size=(3, 2, 2, 4, 5)),
        "z_grid": np.array([0.5, 1.5, 2.5]),
        "igm_model": "inoue",
    }

    original_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        key_with_x64_true = sc.cache_key(**base)

        jax.config.update("jax_enable_x64", False)
        key_with_x64_false = sc.cache_key(**base)
    finally:
        jax.config.update("jax_enable_x64", original_x64)

    assert key_with_x64_true != key_with_x64_false, (
        "Keys must differ when x64 changes; fix may be reverted"
    )


def test_band_factor_key_changes_with_x64():
    """band_factor_key differs when jax_enable_x64 flips."""
    import jax

    rng = np.random.default_rng(1)
    base = {
        "wave_rest": np.linspace(1000.0, 9000.0, 64),
        "filter_waves": [
            np.linspace(3000.0, 4000.0, 16),
            np.linspace(5000.0, 6000.0, 16),
        ],
        "filter_trans": [rng.uniform(0, 1, 16), rng.uniform(0, 1, 16)],
        "z_grid": np.array([0.5, 1.5]),
        "igm_model": "inoue",
        "convention": "bessell",
    }

    original_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        key_with_x64_true = sc.band_factor_key(**base)

        jax.config.update("jax_enable_x64", False)
        key_with_x64_false = sc.band_factor_key(**base)
    finally:
        jax.config.update("jax_enable_x64", original_x64)

    assert key_with_x64_true != key_with_x64_false, (
        "Keys must differ when x64 changes; fix may be reverted"
    )


# ── Regression: keys CHANGE when backend is monkeypatched ──────────────────────


def test_ztable_cache_key_changes_with_backend(monkeypatch):
    """_ztable_cache_key differs when jax.default_backend() returns different values.

    Monkeypatches jax.default_backend to return a fake backend string and verifies
    the key changes. The builders call jax.default_backend() inside the function,
    so the monkeypatch is seen by them.
    """
    import jax

    class MockSSPData:
        ssp_wave = np.array([100.0, 200.0, 300.0])
        ssp_flux = np.ones((1, 1, 3))

    ssp_data = MockSSPData()
    filter_waves = [np.array([100.0, 200.0])]
    filter_trans = [np.array([0.5, 0.8])]
    z_grid = np.array([0.1, 0.5, 1.0])
    apply_igm = False
    taylor_correction = False
    convention = "bessell"
    n_subbands = 0

    # Key with real backend
    key_real = pc._ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid,
        apply_igm,
        taylor_correction,
        convention,
        n_subbands,
    )

    # Key with fake backend (monkeypatched)
    monkeypatch.setattr(jax, "default_backend", lambda: "faketpu")
    key_fake = pc._ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid,
        apply_igm,
        taylor_correction,
        convention,
        n_subbands,
    )

    assert key_real != key_fake, "Keys must differ when backend changes; fix may be reverted"


def test_subband_cache_key_changes_with_backend(monkeypatch):
    """cache_key differs when jax.default_backend() returns different values."""
    import jax

    rng = np.random.default_rng(0)
    base = {
        "waves_rest": rng.uniform(1000.0, 9000.0, size=(3, 2, 2, 4, 5)),
        "z_grid": np.array([0.5, 1.5, 2.5]),
        "igm_model": "inoue",
    }

    key_real = sc.cache_key(**base)

    monkeypatch.setattr(jax, "default_backend", lambda: "faketpu")
    key_fake = sc.cache_key(**base)

    assert key_real != key_fake, "Keys must differ when backend changes; fix may be reverted"


def test_band_factor_key_changes_with_backend(monkeypatch):
    """band_factor_key differs when jax.default_backend() returns different values."""
    import jax

    rng = np.random.default_rng(1)
    base = {
        "wave_rest": np.linspace(1000.0, 9000.0, 64),
        "filter_waves": [
            np.linspace(3000.0, 4000.0, 16),
            np.linspace(5000.0, 6000.0, 16),
        ],
        "filter_trans": [rng.uniform(0, 1, 16), rng.uniform(0, 1, 16)],
        "z_grid": np.array([0.5, 1.5]),
        "igm_model": "inoue",
        "convention": "bessell",
    }

    key_real = sc.band_factor_key(**base)

    monkeypatch.setattr(jax, "default_backend", lambda: "faketpu")
    key_fake = sc.band_factor_key(**base)

    assert key_real != key_fake, "Keys must differ when backend changes; fix may be reverted"


# ── Baseline: version bumps pin the fix ──────────────────────────────────────


def test_schema_version_bumps_prevent_collisions():
    """Version bumps (schema=2→3, v1→v2) prevent old cache entries from colliding.

    The schema change strings in the key ensure that a pre-fix entry (with the
    old version) cannot collide with a post-fix entry (with the new version).
    """
    assert pc._ZTABLE_CACHE_VERSION == 2, (
        "ztable version not bumped; old cache entries may collide"
    )
    assert sc._CACHE_VERSION == 2, "subband version not bumped; old cache entries may collide"


def test_cache_key_determinism():
    """Identical inputs produce identical keys (sanity check for mutations)."""
    rng = np.random.default_rng(42)
    base = {
        "waves_rest": rng.uniform(1000.0, 9000.0, size=(2, 2, 3, 4, 5)),
        "z_grid": np.array([0.5, 1.5]),
        "igm_model": "inoue",
    }

    key1 = sc.cache_key(**base)
    key2 = sc.cache_key(**base)
    assert key1 == key2, "Key is not deterministic"
