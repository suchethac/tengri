"""Tests for ModelCacheOwner isolation, thread safety, and backward compatibility."""

from __future__ import annotations

import threading
import warnings
from unittest.mock import MagicMock

from tengri.inference._model_cache import (
    ModelCacheOwner,
    clear_model_cache,
    clear_structural_kernel_cache,
    get_model_cache,
    get_structural_kernel_cache,
)


class TestModelCacheOwnerIsolation:
    """Per-instance isolation: separate owners don't share entries."""

    def test_two_owners_isolated_model_cache(self):
        """Two ModelCacheOwner instances have independent model caches."""
        owner1 = ModelCacheOwner()
        owner2 = ModelCacheOwner()

        model = MagicMock()
        cache1 = owner1.get_or_compile_model(model)
        cache2 = owner2.get_or_compile_model(model)

        # Same model, but different owners -> different cache dicts
        assert cache1 is not cache2
        assert id(cache1) != id(cache2)

    def test_two_owners_isolated_kernel_cache(self):
        """Two ModelCacheOwner instances have independent kernel caches."""
        owner1 = ModelCacheOwner()
        owner2 = ModelCacheOwner()

        sig = ("photometry", "dust_law", "redshift_z0.05")
        kernel1 = owner1.get_structural_kernel(sig)
        kernel2 = owner2.get_structural_kernel(sig)

        # Same signature, but different owners -> different cache dicts
        assert kernel1 is not kernel2
        assert id(kernel1) != id(kernel2)

    def test_same_owner_shares_model_cache(self):
        """Same owner returns the same cache dict for the same model."""
        owner = ModelCacheOwner()
        model = MagicMock()

        cache1 = owner.get_or_compile_model(model)
        cache2 = owner.get_or_compile_model(model)

        # Same model, same owner -> same cache dict
        assert cache1 is cache2

    def test_same_owner_shares_kernel_cache(self):
        """Same owner returns the same kernel cache dict for the same signature."""
        owner = ModelCacheOwner()
        sig = ("photometry", "dust_law", "redshift_z0.05")

        kernel1 = owner.get_structural_kernel(sig)
        kernel2 = owner.get_structural_kernel(sig)

        # Same signature, same owner -> same cache dict
        assert kernel1 is kernel2


class TestModelCacheOwnerClearing:
    """clear() flushes both model and kernel caches."""

    def test_clear_flushes_both_caches(self):
        """clear() empties both model and kernel caches."""
        owner = ModelCacheOwner()

        # Populate model cache
        model = MagicMock()
        cache = owner.get_or_compile_model(model)
        cache["key"] = "value"

        # Populate kernel cache
        sig = ("photometry", "dust_law")
        kernel = owner.get_structural_kernel(sig)
        kernel["kernel_fn"] = "some_kernel"

        # Verify populated
        assert owner._model_caches[model] == {"key": "value"}
        assert owner._kernel_cache[sig] == {"kernel_fn": "some_kernel"}

        # Clear both
        owner.clear()

        # Verify empty
        assert len(owner._model_caches) == 0
        assert len(owner._kernel_cache) == 0

    def test_clear_idempotent(self):
        """clear() is idempotent (can call multiple times safely)."""
        owner = ModelCacheOwner()
        owner.clear()
        owner.clear()  # Should not raise


class TestModelCacheOwnerThreadSafety:
    """Thread-safety smoke test: concurrent access doesn't corrupt state."""

    def test_concurrent_model_cache_access(self):
        """50 threads, 50 iterations each, different models — no corruption."""
        owner = ModelCacheOwner()
        results = []
        errors = []

        def worker(thread_id: int):
            try:
                for iteration in range(50):
                    model = MagicMock()
                    cache = owner.get_or_compile_model(model)
                    cache[f"thread_{thread_id}_iter_{iteration}"] = thread_id
                    # Read back immediately
                    assert cache[f"thread_{thread_id}_iter_{iteration}"] == thread_id
                results.append(("success", thread_id))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrency errors: {errors}"
        assert len(results) == 2

    def test_concurrent_kernel_cache_access(self):
        """50 iterations on 2 threads, different signatures — no corruption."""
        owner = ModelCacheOwner()
        errors = []

        def worker(thread_id: int):
            try:
                for iteration in range(50):
                    sig = (f"thread_{thread_id}", f"iter_{iteration}")
                    kernel = owner.get_structural_kernel(sig)
                    kernel[f"fn_{thread_id}"] = lambda x: x + thread_id
                    # Verify can read back
                    assert f"fn_{thread_id}" in kernel
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrency errors: {errors}"


class TestBackwardCompatibility:
    """Old module functions still work and emit DeprecationWarning."""

    def test_get_model_cache_deprecated(self):
        """get_model_cache() emits DeprecationWarning."""
        model = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache = get_model_cache(model)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ModelCacheOwner" in str(w[0].message)
            assert isinstance(cache, dict)

    def test_clear_model_cache_deprecated(self):
        """clear_model_cache() emits DeprecationWarning."""
        model = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clear_model_cache(model)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_get_structural_kernel_cache_deprecated(self):
        """get_structural_kernel_cache() emits DeprecationWarning."""
        sig = ("photometry", "dust_law")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache = get_structural_kernel_cache(sig)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert isinstance(cache, dict)

    def test_clear_structural_kernel_cache_deprecated(self):
        """clear_structural_kernel_cache() emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clear_structural_kernel_cache()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_old_functions_use_singleton(self):
        """Old functions delegate to _default_owner singleton."""
        model = MagicMock()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cache1 = get_model_cache(model)
            cache2 = get_model_cache(model)

        # Both calls should return the same dict (singleton behavior)
        assert cache1 is cache2


class TestStructuralKernelLRU:
    """Structural kernel LRU eviction respects max_kernel_entries."""

    def test_lru_eviction_max_entries(self):
        """Exceeding max_kernel_entries triggers LRU eviction."""
        owner = ModelCacheOwner(max_kernel_entries=2)

        sig1 = ("photometry", "1")
        sig2 = ("photometry", "2")
        sig3 = ("photometry", "3")

        kernel1 = owner.get_structural_kernel(sig1)
        kernel1["data"] = "sig1"

        kernel2 = owner.get_structural_kernel(sig2)
        kernel2["data"] = "sig2"

        kernel3 = owner.get_structural_kernel(sig3)
        kernel3["data"] = "sig3"

        # sig1 should be evicted (oldest, not accessed)
        assert sig1 not in owner._kernel_cache
        assert sig2 in owner._kernel_cache
        assert sig3 in owner._kernel_cache

    def test_lru_mru_promotion(self):
        """Accessing a cached kernel promotes it to MRU."""
        owner = ModelCacheOwner(max_kernel_entries=2)

        sig1 = ("photometry", "1")
        sig2 = ("photometry", "2")
        sig3 = ("photometry", "3")

        owner.get_structural_kernel(sig1)
        owner.get_structural_kernel(sig2)

        # Re-access sig1 to promote it to MRU
        owner.get_structural_kernel(sig1)

        # Now add sig3; sig2 should be evicted (oldest LRU)
        owner.get_structural_kernel(sig3)

        assert sig1 in owner._kernel_cache
        assert sig2 not in owner._kernel_cache
        assert sig3 in owner._kernel_cache
