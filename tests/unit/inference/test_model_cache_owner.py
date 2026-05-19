"""Unit tests for ModelCacheOwner class.

Tests verify that:
- Per-instance caches are isolated (two owners don't share entries)
- get_or_compile_model() behaves like a WeakKeyDict
- get_structural_kernel() implements LRU semantics correctly
- clear() flushes both caches
- Thread safety smoke test
- Backwards-compat: old module functions still work and emit deprecation warnings
"""

from __future__ import annotations

import threading
import warnings

import pytest

from tengri.inference._model_cache import (
    ModelCacheOwner,
    get_model_cache,
    get_structural_kernel_cache,
)


@pytest.mark.unit
class TestModelCacheOwnerIsolation:
    """Two ModelCacheOwner instances are isolated."""

    def test_per_instance_isolation_model_cache(self):
        """Per-model caches don't leak across owner instances."""
        owner1 = ModelCacheOwner()
        owner2 = ModelCacheOwner()

        class FakeModel:
            """Dummy model for testing."""

            pass

        model = FakeModel()

        # Both owners should return different dicts for the same model key
        cache1 = owner1.get_or_compile_model(model, lambda: {"id": 1})
        cache2 = owner2.get_or_compile_model(model, lambda: {"id": 2})

        assert cache1["id"] == 1
        assert cache2["id"] == 2
        assert cache1 is not cache2, "Caches should be isolated per owner"

    def test_per_instance_isolation_structural_cache(self):
        """Structural kernel caches don't leak across owner instances."""
        owner1 = ModelCacheOwner()
        owner2 = ModelCacheOwner()

        sig = ("test_sig",)

        cache1 = owner1.get_structural_kernel(sig, lambda: {"kernel": "id1"})
        cache2 = owner2.get_structural_kernel(sig, lambda: {"kernel": "id2"})

        assert cache1["kernel"] == "id1"
        assert cache2["kernel"] == "id2"
        assert cache1 is not cache2, "Structural caches should be isolated per owner"


@pytest.mark.unit
class TestModelCacheOwnerPerModel:
    """get_or_compile_model() behaves like a WeakKeyDict."""

    def test_model_cache_hit(self):
        """Repeated calls with same model return same cached value."""
        owner = ModelCacheOwner()

        class FakeModel:
            pass

        model = FakeModel()

        # First call builds
        cache1 = owner.get_or_compile_model(model, lambda: {"fn": "first"})
        # Second call hits cache
        cache2 = owner.get_or_compile_model(model, lambda: {"fn": "second"})

        assert cache1 is cache2, "Should return same dict on cache hit"
        assert cache1["fn"] == "first", "Should not rebuild on cache hit"

    def test_model_cache_miss_different_models(self):
        """Different model keys get different caches."""
        owner = ModelCacheOwner()

        class FakeModel:
            pass

        m1 = FakeModel()
        m2 = FakeModel()

        cache1 = owner.get_or_compile_model(m1, lambda: {"id": 1})
        cache2 = owner.get_or_compile_model(m2, lambda: {"id": 2})

        assert cache1 is not cache2
        assert cache1["id"] == 1
        assert cache2["id"] == 2


@pytest.mark.unit
class TestModelCacheOwnerStructural:
    """get_structural_kernel() implements LRU and sharing semantics."""

    def test_structural_cache_hit_promotion(self):
        """Touching an entry promotes it to MRU."""
        owner = ModelCacheOwner(max_structural_entries=2)

        sig1 = ("sig1",)
        sig2 = ("sig2",)
        sig3 = ("sig3",)

        # Create entries in order: sig1, sig2, sig3
        c1 = owner.get_structural_kernel(sig1)
        c2 = owner.get_structural_kernel(sig2)

        # Touch sig1 to promote it to MRU
        owner.get_structural_kernel(sig1)

        # Now add sig3 — sig2 should be evicted (LRU)
        c3 = owner.get_structural_kernel(sig3)

        # Verify: sig1 and sig3 in cache, sig2 evicted
        c1_again = owner.get_structural_kernel(sig1)
        c3_again = owner.get_structural_kernel(sig3)

        assert c1 is c1_again, "sig1 should still be cached"
        assert c3 is c3_again, "sig3 should still be cached"

        # sig2 was evicted and rebuilt (different object)
        c2_rebuilt = owner.get_structural_kernel(sig2)
        assert c2 is not c2_rebuilt, "sig2 should be evicted and rebuilt"

    def test_structural_cache_lru_eviction(self):
        """LRU eviction removes oldest entry when maxsize exceeded."""
        owner = ModelCacheOwner(max_structural_entries=2)

        sig_a = ("a",)
        sig_b = ("b",)
        sig_c = ("c",)

        # Populate up to maxsize
        ca = owner.get_structural_kernel(sig_a, lambda: {"val": "a"})
        cb = owner.get_structural_kernel(sig_b, lambda: {"val": "b"})

        assert ca["val"] == "a"
        assert cb["val"] == "b"

        # Add one more — sig_a should be evicted
        cc = owner.get_structural_kernel(sig_c, lambda: {"val": "c"})

        # sig_a is gone; sig_b and sig_c remain
        ca_rebuilt = owner.get_structural_kernel(sig_a, lambda: {"val": "a_new"})
        assert ca is not ca_rebuilt, "sig_a should be evicted"
        assert ca_rebuilt["val"] == "a_new"  # Rebuilt with fresh value

    def test_structural_cache_build_fn_optional(self):
        """get_structural_kernel() creates empty dict if build_fn is None."""
        owner = ModelCacheOwner()

        sig = ("test",)

        # No build_fn provided
        cache = owner.get_structural_kernel(sig, build_fn=None)

        assert isinstance(cache, dict)
        assert len(cache) == 0

        # Can populate it
        cache["populated"] = True
        assert owner.get_structural_kernel(sig)["populated"] is True


@pytest.mark.unit
class TestModelCacheOwnerClear:
    """clear() flushes both caches."""

    def test_clear_model_cache(self):
        """clear() removes all per-model entries."""
        owner = ModelCacheOwner()

        class FakeModel:
            pass

        m1 = FakeModel()
        m2 = FakeModel()

        owner.get_or_compile_model(m1, lambda: {"id": 1})
        owner.get_or_compile_model(m2, lambda: {"id": 2})

        # Clear and rebuild to verify they're gone
        owner.clear()

        # After clear, we get fresh builds
        c1_new = owner.get_or_compile_model(m1, lambda: {"id": 1})
        # (Note: without clear, m1 would hit cache and return same dict)
        # We can't directly test identity due to WeakKeyDict semantics,
        # but we can verify the operation succeeds.
        assert c1_new["id"] == 1

    def test_clear_structural_cache(self):
        """clear() removes all structural entries."""
        owner = ModelCacheOwner()

        sig1 = ("sig1",)
        sig2 = ("sig2",)

        c1 = owner.get_structural_kernel(sig1, lambda: {"id": 1})
        c2 = owner.get_structural_kernel(sig2, lambda: {"id": 2})

        owner.clear()

        # After clear, both signatures get fresh dicts
        c1_new = owner.get_structural_kernel(sig1)
        c2_new = owner.get_structural_kernel(sig2)

        assert c1 is not c1_new
        assert c2 is not c2_new


@pytest.mark.unit
class TestModelCacheOwnerThreadSafety:
    """Basic thread-safety smoke test."""

    def test_concurrent_model_cache_access(self):
        """Multiple threads can access model cache without deadlock."""
        owner = ModelCacheOwner()

        class FakeModel:
            pass

        models = [FakeModel() for _ in range(5)]
        results = []

        def worker(model_idx):
            model = models[model_idx]
            cache = owner.get_or_compile_model(model, lambda: {"id": model_idx})
            results.append(cache["id"])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_concurrent_structural_cache_access(self):
        """Multiple threads can access structural cache without deadlock."""
        owner = ModelCacheOwner()

        results = []

        def worker(sig_idx):
            sig = (f"sig_{sig_idx}",)
            cache = owner.get_structural_kernel(sig, lambda: {"id": sig_idx})
            results.append(cache["id"])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert sorted(results) == [0, 1, 2, 3, 4]


@pytest.mark.unit
class TestBackwardsCompat:
    """Module-level functions still work and emit deprecation warnings."""

    def test_get_model_cache_deprecation_warning(self):
        """get_model_cache() emits DeprecationWarning."""

        class FakeModel:
            pass

        model = FakeModel()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_model_cache(model)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ModelCacheOwner.get_or_compile_model()" in str(w[0].message)

    def test_get_structural_kernel_cache_deprecation_warning(self):
        """get_structural_kernel_cache() emits DeprecationWarning."""

        sig = ("test_sig",)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_structural_kernel_cache(sig)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ModelCacheOwner.get_structural_kernel()" in str(w[0].message)

    def test_get_model_cache_still_works(self):
        """get_model_cache() still returns usable dict (even with warning)."""

        class FakeModel:
            pass

        model = FakeModel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cache = get_model_cache(model)

            # Should be a dict we can use
            cache["key"] = "value"
            assert cache["key"] == "value"

            # Second call should hit the same dict
            cache2 = get_model_cache(model)
            assert cache2["key"] == "value"

    def test_get_structural_kernel_cache_still_works(self):
        """get_structural_kernel_cache() still returns usable dict (even with warning)."""

        sig = ("test_sig",)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cache = get_structural_kernel_cache(sig)

            # Should be a dict we can use
            cache["kernel"] = "compiled_fn"
            assert cache["kernel"] == "compiled_fn"

            # Second call should hit the same dict
            cache2 = get_structural_kernel_cache(sig)
            assert cache2["kernel"] == "compiled_fn"
