"""Tests for XLA persistent compilation cache configuration."""

import os

import jax
import pytest


@pytest.mark.contract
def test_cache_dir_writable():
    """Cache directory configured by tengri must be writable.

    The cache dir is set on `import tengri`; if it is not writable, every
    notebook restart pays the full compile cost. Verifying the
    *configured* dir is set is a tautology (we set it ourselves); the
    only failure mode worth catching is an unwritable location.
    """
    import tengri  # noqa: F401

    cache_dir = jax.config.jax_compilation_cache_dir
    if not cache_dir:
        pytest.skip("compilation cache disabled in this environment")
    os.makedirs(cache_dir, exist_ok=True)
    test_file = os.path.join(cache_dir, "_tengri_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except OSError as e:
        pytest.fail(f"Cache dir {cache_dir} not writable: {e}")
