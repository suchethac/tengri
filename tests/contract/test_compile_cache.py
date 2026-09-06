# SPDX-License-Identifier: BSD-3-Clause
"""Tests for CompileCache — explicit owner of JIT compilation state.

ADR-deepen Step C: per-Fitter or per-CatalogFitter cache isolation.
"""

from __future__ import annotations

import threading

import pytest

from tengri.inference.jit_engine import CompileCache, lean, persistent

pytestmark = pytest.mark.contract


class TestCompileCacheBasics:
    """Test CompileCache fundamentals: creation, modes, clear."""

    def test_cache_creation_default(self):
        """Test CompileCache creates with sensible defaults."""
        cache = CompileCache()
        assert cache.max_entries == 2  # default from env or fallback
        assert cache.mode == "normal"
        assert len(cache._store) == 0

    def test_cache_creation_custom(self):
        """Test CompileCache accepts custom max_entries and mode."""
        cache = CompileCache(max_entries=5, mode="lean")
        assert cache.max_entries == 5
        assert cache.mode == "lean"

    def test_cache_get_or_compile_miss_then_hit(self):
        """Test get_or_compile caches value and returns it on hits."""
        cache = CompileCache(max_entries=2)

        # Track calls
        call_count = [0]

        def build_fn():
            call_count[0] += 1
            return {"compiled": "value"}

        # First call: miss, calls builder
        result1 = cache.get_or_compile("key1", build_fn)
        assert result1 == {"compiled": "value"}
        assert call_count[0] == 1

        # Second call: hit, no builder call
        result2 = cache.get_or_compile("key1", build_fn)
        assert result2 == {"compiled": "value"}
        assert call_count[0] == 1  # no new call

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max_entries exceeded."""
        cache = CompileCache(max_entries=2)

        cache.get_or_compile("key1", lambda: "value1")
        cache.get_or_compile("key2", lambda: "value2")
        assert len(cache._store) == 2

        # Third key should evict the oldest (key1)
        cache.get_or_compile("key3", lambda: "value3")
        assert len(cache._store) == 2
        assert "key1" not in cache._store
        assert "key2" in cache._store
        assert "key3" in cache._store

    def test_cache_lru_move_to_end_on_hit(self):
        """Test that hits move entry to end of LRU."""
        cache = CompileCache(max_entries=2)

        cache.get_or_compile("key1", lambda: "value1")
        cache.get_or_compile("key2", lambda: "value2")

        # Access key1 again: moves it to end
        cache.get_or_compile("key1", lambda: "value1")

        # Adding key3 should evict key2 (which is now the oldest)
        cache.get_or_compile("key3", lambda: "value3")
        assert "key1" in cache._store  # still here (was hit, moved to end)
        assert "key2" not in cache._store  # evicted
        assert "key3" in cache._store

    def test_cache_clear(self):
        """Test clear() empties the cache."""
        cache = CompileCache()
        cache.get_or_compile("key1", lambda: "value1")
        cache.get_or_compile("key2", lambda: "value2")
        assert len(cache._store) == 2

        cache.clear()
        assert len(cache._store) == 0

    def test_cache_set_mode(self):
        """Test set_mode updates the mode."""
        cache = CompileCache(mode="normal")
        assert cache.mode == "normal"

        cache.set_mode("lean")
        assert cache.mode == "lean"

        cache.set_mode("persistent")
        assert cache.mode == "persistent"

    def test_cache_set_mode_invalid(self):
        """Test set_mode raises on invalid mode."""
        cache = CompileCache()
        with pytest.raises(ValueError, match="mode must be one of"):
            cache.set_mode("invalid_mode")

    def test_cache_memory_estimate_gb(self):
        """Test memory_estimate_gb returns None (placeholder)."""
        cache = CompileCache()
        cache.get_or_compile("key1", lambda: {"fn": "value"})
        # Currently returns None on all platforms
        assert cache.memory_estimate_gb() is None


class TestCompileCacheIsolation:
    """Test per-Fitter and per-CatalogFitter cache isolation."""

    def test_separate_caches_dont_evict_each_other(self):
        """Test two caches with max_entries=1 don't interfere."""
        cache1 = CompileCache(max_entries=1)
        cache2 = CompileCache(max_entries=1)

        # Each cache builds and holds one entry
        cache1.get_or_compile("sig1", lambda: "engine1")
        cache2.get_or_compile("sig2", lambda: "engine2")

        # Neither evicted the other
        assert "sig1" in cache1._store
        assert "sig2" in cache2._store
        assert len(cache1._store) == 1
        assert len(cache2._store) == 1


class TestCompileCacheThreading:
    """Test CompileCache thread safety."""

    def test_concurrent_gets_thread_safe(self):
        """Test concurrent get_or_compile calls are thread-safe."""
        cache = CompileCache(max_entries=10)
        results = []
        errors = []

        def worker(key_id):
            try:
                for _ in range(5):
                    result = cache.get_or_compile(
                        f"key_{key_id}", lambda key_id=key_id: f"value_{key_id}"
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 15  # 3 workers × 5 iterations


class TestDeprecationWarnings:
    """Test that module-level lean() and persistent() emit DeprecationWarning."""

    def test_lean_context_manager_deprecation(self):
        """Test lean() emits DeprecationWarning."""
        with (
            pytest.warns(
                DeprecationWarning, match="tengri.lean\\(\\) context manager is deprecated"
            ),
            lean(),
        ):
            pass

    def test_persistent_context_manager_deprecation(self):
        """Test persistent() emits DeprecationWarning."""
        with (
            pytest.warns(
                DeprecationWarning, match="tengri.persistent\\(\\) context manager is deprecated"
            ),
            persistent(),
        ):
            pass


class TestCompileCacheWithFitter:
    """Integration tests: CompileCache with Fitter instances."""

    @pytest.fixture
    def minimal_model_and_data(self, synthetic_ssp, simple_observation):
        """A minimal SEDModel and data for testing.

        No try/except. This fixture used to wrap its setup in
        ``except Exception: pytest.skip("Could not create minimal model")`` --
        and the build inside raised, because ``observation=`` was a plain dict
        and ``ssp_data=`` a bare string. Every test below therefore skipped,
        under a message that reads as a missing fixture. Build failures fail.
        """
        import jax.numpy as jnp

        from tengri import SEDModel, recipes

        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=simple_observation,
            **recipes.mock_recovery_minimal(),
        )

        data = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.1, 0.1])
        return model, data, noise

    def test_fitter_accepts_cache_parameter(self, minimal_model_and_data):
        """Test Fitter accepts cache parameter."""
        from tengri.inference.fitter import Fitter

        model, data, noise = minimal_model_and_data
        cache = CompileCache(max_entries=2)

        fitter = Fitter(model, data, noise, cache=cache)
        assert fitter.cache is cache

    def test_fitter_defaults_to_singleton_cache(self, minimal_model_and_data):
        """Test Fitter defaults to module-level singleton when cache=None."""
        from tengri.inference.fitter import Fitter
        from tengri.inference.jit_engine import _get_singleton_cache

        model, data, noise = minimal_model_and_data

        fitter1 = Fitter(model, data, noise, cache=None)
        fitter2 = Fitter(model, data, noise, cache=None)

        # Both should use the same singleton
        assert fitter1.cache is fitter2.cache
        assert fitter1.cache is _get_singleton_cache()

    def test_separate_fitters_separate_caches(self, minimal_model_and_data):
        """Test two Fitters with separate caches don't interfere."""
        from tengri.inference.fitter import Fitter

        model, data, noise = minimal_model_and_data

        cache1 = CompileCache(max_entries=1)
        cache2 = CompileCache(max_entries=1)

        fitter1 = Fitter(model, data, noise, cache=cache1)
        fitter2 = Fitter(model, data, noise, cache=cache2)

        assert fitter1.cache is not fitter2.cache
        assert len(fitter1.cache._store) == 0
        assert len(fitter2.cache._store) == 0


class TestCatalogFitterCache:
    """Test that CatalogFitter threads a single cache through all galaxies."""

    @pytest.fixture
    def minimal_catalog_setup(self, synthetic_ssp, simple_observation):
        """A minimal model and catalog for testing.

        Same repair as :meth:`minimal_model_and_data` above -- the skip-on-any-
        exception handler hid a dict passed where an ``Observation`` belongs.
        """
        import jax.numpy as jnp

        from tengri import SEDModel, recipes

        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=simple_observation,
            **recipes.mock_recovery_minimal(),
        )

        galaxies = [
            {
                "flux_obs": jnp.array([1.0, 2.0, 3.0]),
                "noise": jnp.array([0.1, 0.1, 0.1]),
            },
            {
                "flux_obs": jnp.array([1.5, 2.5, 3.5]),
                "noise": jnp.array([0.1, 0.1, 0.1]),
            },
        ]
        return model, galaxies

    def test_catalog_fitter_creates_cache(self, minimal_catalog_setup):
        """Test CatalogFitter creates its own CompileCache."""
        from tengri.inference.catalog_fitter import CatalogFitter

        model, galaxies = minimal_catalog_setup

        cat = CatalogFitter(model, galaxies)
        assert isinstance(cat.cache, CompileCache)

    def test_catalog_fitter_threads_cache(self, minimal_catalog_setup):
        """Test CatalogFitter threads cache through dummy and sequential fitters."""
        from tengri.inference.catalog_fitter import CatalogFitter

        model, galaxies = minimal_catalog_setup

        cat = CatalogFitter(model, galaxies)
        dummy_fitter = cat._get_dummy_fitter()

        # Dummy fitter should use the same cache
        assert dummy_fitter.cache is cat.cache
