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
    """Reset module-level state between tests.

    ``_eviction_supported`` is forced True so the cap tests pin tengri's own
    resolution logic rather than whether filelock happens to be installed in the
    test environment; the two tests that care about that path override it.
    """
    monkeypatch.setattr(jax_cache, "_eviction_supported", lambda: True)
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


def test_cap_stands_down_without_filelock(monkeypatch, tmp_path):
    """No filelock means the cap BREAKS the cache, so we must not set one.

    JAX gates jax_compilation_cache_max_size behind filelock and raises
    "Please install the `filelock` package..." on every cache read when a cap is
    set without it. Unbounded is bad; broken is worse.
    """
    monkeypatch.setattr(jax_cache, "_eviction_supported", lambda: False)
    jax_cache.enable_persistent_cache(tmp_path)
    assert jax.config.jax_compilation_cache_max_size == jax_cache.UNBOUNDED_CACHE


def test_cap_applies_when_filelock_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(jax_cache, "_eviction_supported", lambda: True)
    jax_cache.enable_persistent_cache(tmp_path)
    assert jax.config.jax_compilation_cache_max_size == jax_cache.DEFAULT_MAX_CACHE_BYTES


# -------------------------------------------------- orphaned atime markers (#1661)


def test_entry_suffixes_match_jax():
    """Our mirrored copies must track ``jax._src.lru_cache``.

    :func:`repair_orphaned_atimes` finds entries by suffix. A rename upstream
    would not raise -- it would silently match nothing, leaving the cache broken
    in exactly the way the repair exists to prevent. Fail loudly here instead.
    """
    lru = pytest.importorskip("jax._src.lru_cache")
    assert jax_cache._CACHE_SUFFIX == lru._CACHE_SUFFIX
    assert jax_cache._ATIME_SUFFIX == lru._ATIME_SUFFIX


def test_atime_encoding_matches_jax(tmp_path):
    """The marker JAX reads back must be the integer we meant to write.

    JAX decodes with ``int.from_bytes(..., "little")``; encoding width or order
    drift would be read as a wildly wrong timestamp rather than an error, and
    eviction would silently pick the wrong victims.
    """
    (tmp_path / "k-cache").write_bytes(b"payload")
    jax_cache.repair_orphaned_atimes(tmp_path)

    raw = (tmp_path / "k-atime").read_bytes()
    assert len(raw) == 8
    assert int.from_bytes(raw, "little") == (tmp_path / "k-cache").stat().st_mtime_ns


def test_filelock_is_a_declared_dependency():
    """The default cap is inert unless filelock ships with tengri."""
    import re

    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    deps = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M).group(1)
    assert re.search(r'"filelock[><=]', deps), "filelock must be a runtime dependency"


# --------------------------------------------------------------------------
# cache_stats: is the cap binding, and on WHAT
# --------------------------------------------------------------------------
#
# ``cache_size_bytes`` answers "how big" and cannot answer "and would a higher
# min_compile_time_secs help", because a cache at 8.5 GB looks identical
# whether a hundred compiled samplers or ten thousand per-filter micro-kernels
# put it there -- and only the first is worth raising a cap for. Measured on a
# real cache at 99.0 % of its cap (bench/reports/2026-08-31_fast_nuts.md
# Finding 5): 1585 of 2284 entries were under 64 KB and held 0.11 % of the
# bytes, while the largest 100 held 79.2 %.


def test_cache_stats_counts_artifacts_not_atimes(tmp_path):
    """Only ``*-cache`` files are entries; the 8-byte ``*-atime`` siblings are not.

    Counting both would double every entry count and add nothing to the bytes,
    which is exactly the kind of quietly-wrong denominator this file exists to
    keep out.
    """
    (tmp_path / "k1-cache").write_bytes(b"x" * 4096)
    (tmp_path / "k1-atime").write_bytes(b"\x00" * 8)
    (tmp_path / "k2-cache").write_bytes(b"y" * 8192)
    (tmp_path / "k2-atime").write_bytes(b"\x00" * 8)

    stats = jax_cache.cache_stats(tmp_path)
    assert stats["n_entries"] == 2
    assert stats["total_bytes"] == 4096 + 8192


def test_cache_stats_separates_the_small_entries_from_the_bytes(tmp_path):
    """The whole point: many small entries, almost none of the size."""
    for i in range(20):
        (tmp_path / f"small{i}-cache").write_bytes(b"x" * 1024)
    (tmp_path / "big-cache").write_bytes(b"y" * (4 * 1024 * 1024))

    stats = jax_cache.cache_stats(tmp_path)
    assert stats["n_entries"] == 21
    assert stats["n_entries_under_64kb"] == 20
    assert stats["bytes_under_64kb"] == 20 * 1024
    assert stats["largest_bytes"] == 4 * 1024 * 1024
    # One entry of 21 holds 99.5 % of the bytes: raising min_compile_time_secs
    # would evict the other twenty and free half a percent.
    assert stats["top10_bytes"] / stats["total_bytes"] > 0.99


def test_cache_stats_reports_the_fraction_of_the_configured_cap(tmp_path):
    jax_cache.enable_persistent_cache(tmp_path, max_size_bytes=1024)
    (tmp_path / "k-cache").write_bytes(b"x" * 512)
    stats = jax_cache.cache_stats(tmp_path)
    assert stats["max_bytes"] == 1024
    assert stats["fraction_of_cap"] == pytest.approx(0.5)


def test_cache_stats_missing_dir(tmp_path):
    stats = jax_cache.cache_stats(tmp_path / "does_not_exist")
    assert stats["n_entries"] == 0
    assert stats["total_bytes"] == 0
    assert stats["largest_bytes"] == 0
