"""Tests for XLA persistent compilation cache configuration.

Validates that the compilation cache is properly configured in diffsed,
and that cache directory is writable.
"""

import os

import jax
import pytest


class TestXLACompilationCache:
    """Verify XLA compilation cache is enabled."""

    def test_cache_dir_configured(self):
        """diffsed sets jax_compilation_cache_dir on import."""
        import diffsed  # noqa: F401

        cache_dir = jax.config.jax_compilation_cache_dir
        assert cache_dir is not None and cache_dir != "", (
            "jax_compilation_cache_dir not set after importing diffsed"
        )

    def test_cache_dir_writable(self):
        """Cache directory is writable."""
        import diffsed  # noqa: F401

        cache_dir = jax.config.jax_compilation_cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            test_file = os.path.join(cache_dir, "_diffsed_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except OSError as e:
                pytest.fail(f"Cache dir {cache_dir} not writable: {e}")

    def test_min_entry_size_zero(self):
        """Minimum entry size is 0 so all compilations are cached."""
        import diffsed  # noqa: F401

        min_size = jax.config.jax_persistent_cache_min_entry_size_bytes
        assert min_size == 0, f"Expected min_entry_size=0, got {min_size}"
