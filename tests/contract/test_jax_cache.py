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


# ---------------------------------------------------------------- size cap (#1507)
#
# The cache reached 141 GB on a 48 GB machine because nothing ever passed
# max_size_bytes: enable_persistent_cache() supports the cap, __init__ called it
# without one, and JAX does not evict by default. These pin the cap on.


def test_max_size_bytes_sets_the_jax_cap(tmp_path):
    jax_cache.enable_persistent_cache(tmp_path, max_size_bytes=3 * 1024**3)
    assert jax.config.jax_compilation_cache_max_size == 3 * 1024**3


def test_default_cap_is_applied_when_not_specified(tmp_path):
    """The auto-enable path must bound the cache, not leave it unlimited."""
    jax_cache.enable_persistent_cache(tmp_path)
    cap = jax.config.jax_compilation_cache_max_size
    assert cap > 0, "persistent cache enabled with no size cap — this is #1507"
    assert cap == jax_cache.DEFAULT_MAX_CACHE_BYTES


def test_cap_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TENGRI_JAX_CACHE_MAX_GB", "2.5")
    jax_cache.enable_persistent_cache(tmp_path)
    assert jax.config.jax_compilation_cache_max_size == int(2.5 * 1024**3)


def test_cap_env_override_zero_means_unbounded(monkeypatch, tmp_path):
    """An explicit 0 opts out, for anyone who really wants an unbounded cache.

    Must map to JAX's -1 sentinel, not literal 0: 0 is a zero-BYTE ceiling, which
    would silently switch caching off rather than removing the cap.
    """
    monkeypatch.setenv("TENGRI_JAX_CACHE_MAX_GB", "0")
    jax_cache.enable_persistent_cache(tmp_path)
    assert jax.config.jax_compilation_cache_max_size == jax_cache.UNBOUNDED_CACHE
    assert jax_cache.UNBOUNDED_CACHE == -1


def test_cap_env_garbage_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TENGRI_JAX_CACHE_MAX_GB", "not-a-number")
    jax_cache.enable_persistent_cache(tmp_path)
    assert jax.config.jax_compilation_cache_max_size == jax_cache.DEFAULT_MAX_CACHE_BYTES


def test_enable_does_not_walk_the_cache_when_quiet(tmp_path, monkeypatch):
    """The size is only needed for an INFO log line.

    Walking the tree unconditionally costs ~3.5 us/file on every ``import
    tengri``, which is wasted work whenever nobody is listening.
    """
    calls = []
    monkeypatch.setattr(jax_cache, "cache_size_bytes", lambda *a, **k: calls.append(1) or 0)
    monkeypatch.setattr(jax_cache.logger, "isEnabledFor", lambda level: False)
    jax_cache.enable_persistent_cache(tmp_path)
    assert calls == [], "cache was walked even though the log line is suppressed"


def test_clear_cache_reports_what_it_freed(tmp_path, capsys):
    (tmp_path / "blob").write_bytes(b"x" * 4096)
    freed = jax_cache.clear_cache(tmp_path)
    assert freed >= 4096, "clear_cache should report the bytes it reclaimed"
