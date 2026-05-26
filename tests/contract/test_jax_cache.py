# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for tengri.utils.jax_cache."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from tengri.utils import jax_cache

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Reset module-level state between tests."""
    monkeypatch.setattr(jax_cache, "_ENABLED_DIR", None)
    monkeypatch.delenv("TENGRI_DISABLE_JAX_CACHE", raising=False)
    monkeypatch.delenv("TENGRI_JAX_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_enable_persistent_cache_explicit_dir(tmp_path):
    target = jax_cache.enable_persistent_cache(tmp_path)
    assert target == tmp_path
    assert target.exists()
    assert jax.config.jax_compilation_cache_dir == str(tmp_path)
    assert jax_cache.is_cache_enabled()


def test_enable_persistent_cache_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TENGRI_JAX_CACHE_DIR", str(tmp_path))
    target = jax_cache.enable_persistent_cache()
    assert target == tmp_path
    assert jax.config.jax_compilation_cache_dir == str(tmp_path)


def test_enable_persistent_cache_xdg_default(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = jax_cache.enable_persistent_cache()
    assert target == tmp_path / "tengri_jax_cache"
    assert target.exists()


def test_enable_persistent_cache_legacy_default(monkeypatch, tmp_path):
    """No env var, no XDG → falls back to ~/.cache/tengri_jax_cache."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = jax_cache._default_cache_dir()
    assert target == tmp_path / ".cache" / "tengri_jax_cache"


def test_enable_idempotent(tmp_path):
    p1 = jax_cache.enable_persistent_cache(tmp_path)
    p2 = jax_cache.enable_persistent_cache(tmp_path)
    assert p1 == p2
    assert jax_cache.is_cache_enabled()


def test_disable_env_var(monkeypatch, tmp_path):
    """TENGRI_DISABLE_JAX_CACHE=1 makes enable a no-op."""
    monkeypatch.setenv("TENGRI_DISABLE_JAX_CACHE", "1")
    target = jax_cache.enable_persistent_cache(tmp_path)
    # Returns resolved dir but does NOT touch JAX config or set state.
    assert not jax_cache.is_cache_enabled()
    # Path still resolved (caller may want to know what it would have been).
    assert isinstance(target, Path)


def test_min_compile_time_secs_applied(tmp_path):
    jax_cache.enable_persistent_cache(tmp_path, min_compile_time_secs=12.5)
    assert jax.config.jax_persistent_cache_min_compile_time_secs == pytest.approx(12.5)


def test_cache_size_bytes_empty(tmp_path):
    assert jax_cache.cache_size_bytes(tmp_path) == 0


def test_cache_size_bytes_with_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 2048)
    assert jax_cache.cache_size_bytes(tmp_path) == 3072


def test_cache_size_bytes_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert jax_cache.cache_size_bytes(missing) == 0


def test_clear_cache(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 16)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 16)
    jax_cache.clear_cache(tmp_path)
    assert tmp_path.exists()  # dir preserved
    assert list(tmp_path.iterdir()) == []  # contents wiped


def test_clear_cache_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    # Should not raise.
    jax_cache.clear_cache(missing)


def test_clear_cache_uses_enabled_dir(tmp_path):
    """clear_cache() with no arg uses the currently-enabled dir."""
    jax_cache.enable_persistent_cache(tmp_path)
    (tmp_path / "marker.bin").write_bytes(b"data")
    jax_cache.clear_cache()
    assert list(tmp_path.iterdir()) == []


def test_resolve_dir_explicit(tmp_path):
    assert jax_cache._resolve_dir(tmp_path) == tmp_path


def test_resolve_dir_falls_back_to_enabled(tmp_path):
    jax_cache.enable_persistent_cache(tmp_path)
    assert jax_cache._resolve_dir(None) == tmp_path


def test_resolve_dir_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TENGRI_JAX_CACHE_DIR", str(tmp_path))
    assert jax_cache._resolve_dir(None) == tmp_path
