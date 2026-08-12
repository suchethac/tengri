# SPDX-License-Identifier: BSD-3-Clause
"""Regression: one orphaned ``-cache`` file kills every cache write (#1661).

JAX's LRU cache stores each entry as a pair: ``KEY-cache`` holds the compiled
artifact, ``KEY-atime`` holds an 8-byte little-endian nanosecond timestamp used
to order eviction. ``_evict_if_needed`` walks every ``*-cache`` file and reads
its companion **unconditionally** -- no ``exists()`` check, no ``try``::

    key = cache_path.name.removesuffix(_CACHE_SUFFIX)
    atime_path = self.path / f"{key}{_ATIME_SUFFIX}"
    file_atime = int.from_bytes(atime_path.read_bytes(), "little")

``put()`` calls that sweep *before* writing, so a single orphaned ``-cache``
file makes every subsequent write raise ``FileNotFoundError``. ``get()`` never
calls it, so reads keep working and the cache looks healthy while being unable
to accept anything new.

Observed: a 1.2 GB cache whose entries were all timestamped inside one
four-minute window five days earlier, with the ``-atime`` markers still
updating daily. Seven orphans, every one >= 14.15 MB against sixteen healthy
entries all <= 14.14 MB -- consistent with processes dying inside the window
``put()`` leaves open between writing the artifact and writing its marker,
which widens with the size of the value.

The repair reconstructs the marker from the cache file's own mtime instead of
deleting the entry, because the entries worth losing least are exactly the
large ones that orphan most often.

Note the interaction with #1507: eviction is the only consumer of these
markers, and the 8 GiB cap is what turns eviction on. An unbounded cache never
trips this.
"""

from __future__ import annotations

import pytest

from tengri.utils import jax_cache

pytestmark = pytest.mark.regression_bug


def _atime_of(path):
    """Read back the nanosecond timestamp JAX stores in an ``-atime`` file."""
    return int.from_bytes(path.read_bytes(), "little")


@pytest.fixture(autouse=True)
def _reset_enabled_dir(monkeypatch):
    """``enable_persistent_cache`` is idempotent; unlatch it for each test."""
    monkeypatch.setattr(jax_cache, "_ENABLED_DIR", None)
    monkeypatch.setattr(jax_cache, "_eviction_supported", lambda: True)
    monkeypatch.delenv("TENGRI_DISABLE_JAX_CACHE", raising=False)
    monkeypatch.delenv("TENGRI_JAX_CACHE_DIR", raising=False)
    monkeypatch.delenv("TENGRI_JAX_CACHE_MAX_GB", raising=False)


def _plant(cache_dir, key, payload=b"compiled", *, with_atime):
    """Write a cache entry, optionally without its ``-atime`` companion."""
    (cache_dir / f"{key}-cache").write_bytes(payload)
    if with_atime:
        (cache_dir / f"{key}-atime").write_bytes((123456789).to_bytes(8, "little"))


# --------------------------------------------------------------- the repair itself


def test_repair_synthesizes_the_missing_marker(tmp_path):
    _plant(tmp_path, "orphan", with_atime=False)

    repaired = jax_cache.repair_orphaned_atimes(tmp_path)

    assert repaired == 1
    marker = tmp_path / "orphan-atime"
    assert marker.exists(), "the missing -atime companion was not reconstructed"
    # Reconstructed from the artifact's own mtime -- a faithful "last known access".
    assert _atime_of(marker) == (tmp_path / "orphan-cache").stat().st_mtime_ns


def test_repair_preserves_the_artifact(tmp_path):
    """Repair, not eviction: the compiled artifact is what is expensive."""
    _plant(tmp_path, "orphan", payload=b"a 417 MB kernel, in spirit", with_atime=False)

    jax_cache.repair_orphaned_atimes(tmp_path)

    assert (tmp_path / "orphan-cache").read_bytes() == b"a 417 MB kernel, in spirit"


def test_repair_does_not_disturb_healthy_entries(tmp_path):
    """Rewriting a live marker would reset the LRU order on every import.

    Eviction picks victims by ascending atime. Stamping every entry with "now"
    would flatten that ordering and make eviction arbitrary, so the sweep must
    touch only the entries that are actually missing one.
    """
    _plant(tmp_path, "healthy", with_atime=True)
    _plant(tmp_path, "orphan", with_atime=False)
    before = _atime_of(tmp_path / "healthy-atime")

    repaired = jax_cache.repair_orphaned_atimes(tmp_path)

    assert repaired == 1, "only the orphan should have been repaired"
    assert _atime_of(tmp_path / "healthy-atime") == before


def test_repair_is_idempotent(tmp_path):
    _plant(tmp_path, "orphan", with_atime=False)

    assert jax_cache.repair_orphaned_atimes(tmp_path) == 1
    assert jax_cache.repair_orphaned_atimes(tmp_path) == 0


def test_repair_handles_every_orphan_not_just_the_first(tmp_path):
    """One survivor still breaks every write, so a partial sweep is no fix."""
    for i in range(5):
        _plant(tmp_path, f"orphan{i}", with_atime=False)

    assert jax_cache.repair_orphaned_atimes(tmp_path) == 5
    assert not list(_orphans(tmp_path))


def test_repair_ignores_a_stray_marker_with_no_artifact(tmp_path):
    """The mirror-image inconsistency is harmless: the sweep globs ``*-cache``."""
    (tmp_path / "ghost-atime").write_bytes((1).to_bytes(8, "little"))

    assert jax_cache.repair_orphaned_atimes(tmp_path) == 0


def test_repair_tolerates_a_missing_directory(tmp_path):
    assert jax_cache.repair_orphaned_atimes(tmp_path / "not_here") == 0


def test_repair_ignores_unrelated_files(tmp_path):
    """``ionspec_tables/`` and ``.lockfile`` legitimately share this directory."""
    (tmp_path / ".lockfile").write_bytes(b"")
    (tmp_path / "ionspec_tables").mkdir()
    (tmp_path / "ionspec_tables" / "table.npz").write_bytes(b"data")

    assert jax_cache.repair_orphaned_atimes(tmp_path) == 0
    assert (tmp_path / "ionspec_tables" / "table.npz").exists()


def _orphans(cache_dir):
    for entry in cache_dir.glob("*-cache"):
        if not (cache_dir / f"{entry.name.removesuffix('-cache')}-atime").exists():
            yield entry


# ------------------------------------------------------- wired into startup


def test_enable_persistent_cache_repairs_on_startup(tmp_path):
    """The whole point: ``import tengri`` must not hand JAX a broken directory."""
    _plant(tmp_path, "orphan", with_atime=False)

    jax_cache.enable_persistent_cache(tmp_path)

    assert not list(_orphans(tmp_path)), "startup left the cache unable to accept writes"


def test_no_repair_when_eviction_is_disabled(tmp_path, monkeypatch):
    """Eviction is the only consumer of these markers; unbounded never reads them.

    Skipping the sweep keeps the opt-out path free of work it cannot need.
    """
    monkeypatch.setenv("TENGRI_JAX_CACHE_MAX_GB", "0")  # -> UNBOUNDED_CACHE
    _plant(tmp_path, "orphan", with_atime=False)

    jax_cache.enable_persistent_cache(tmp_path)

    assert not (tmp_path / "orphan-atime").exists()


def test_startup_survives_an_unreadable_cache_directory(tmp_path, monkeypatch):
    """A repair that raises would break ``import tengri`` for everyone."""

    def _explode(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(jax_cache.os, "scandir", _explode)
    jax_cache.enable_persistent_cache(tmp_path)  # must not raise


# ----------------------------------------------------------- reporting the condition


def test_orphaned_entries_lists_every_orphan(tmp_path):
    _plant(tmp_path, "healthy", with_atime=True)
    _plant(tmp_path, "a", with_atime=False)
    _plant(tmp_path, "b", with_atime=False)

    assert sorted(jax_cache.orphaned_entries(tmp_path)) == ["a", "b"]


def test_orphaned_entries_empty_on_a_healthy_cache(tmp_path):
    _plant(tmp_path, "healthy", with_atime=True)

    assert jax_cache.orphaned_entries(tmp_path) == []


def test_orphaned_entries_tolerates_a_missing_directory(tmp_path):
    assert jax_cache.orphaned_entries(tmp_path / "not_here") == []


def test_doctor_reports_a_write_dead_cache(tmp_path):
    """``os.access`` says "writable" for the whole outage; doctor must not.

    Orphans can appear *after* startup -- a sibling process dying inside its
    own ``put`` is enough -- so the diagnostic has to detect the condition and
    not merely rely on the startup repair having run.
    """
    import tengri

    jax_cache.enable_persistent_cache(tmp_path)
    _plant(tmp_path, "orphan", with_atime=False)  # appears mid-session

    report = tengri.doctor()

    assert "ALL cache writes are failing" in report
    assert "repair_orphaned_atimes" in report, "the report must name the remedy"


def test_doctor_is_quiet_on_a_healthy_cache(tmp_path):
    import tengri

    jax_cache.enable_persistent_cache(tmp_path)
    _plant(tmp_path, "healthy", with_atime=True)

    assert "cache writes are failing" not in tengri.doctor()


# ------------------------------------------- against the real JAX consumer


def test_orphan_breaks_a_real_jax_put_and_the_repair_fixes_it(tmp_path):
    """End-to-end through ``LRUCache`` -- the component that actually fails.

    The first half pins upstream behavior. If it ever stops raising, JAX has
    fixed ``_evict_if_needed`` and this workaround can be retired.
    """
    lru = pytest.importorskip("jax._src.lru_cache")
    cache = lru.LRUCache(str(tmp_path), max_size=8 * 1024**3)

    cache.put("good", b"x" * 64)
    assert cache.get("good") == b"x" * 64, "harness is broken before we start"

    _plant(tmp_path, "orphan", payload=b"y" * 64, with_atime=False)

    with pytest.raises(FileNotFoundError, match="orphan-atime"):
        cache.put("blocked", b"z" * 64)
    assert cache.get("blocked") is None, "the entry was silently not cached"

    jax_cache.repair_orphaned_atimes(tmp_path)

    cache.put("blocked", b"z" * 64)
    assert cache.get("blocked") == b"z" * 64, "writes still broken after repair"
    assert cache.get("orphan") == b"y" * 64, "repair cost us the orphan's artifact"
