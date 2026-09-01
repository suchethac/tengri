# SPDX-License-Identifier: BSD-3-Clause
"""Persistent JAX compilation cache for tengri.

Wraps JAX's persistent compilation cache with sensible defaults, an env-var
override, idempotency, and helpers to inspect or clear the on-disk cache.

Quick start
-----------
The cache is enabled automatically on ``import tengri`` to a default
location under ``~/.cache``. Override the directory via env var::

    export TENGRI_JAX_CACHE_DIR=/scratch/$USER/jax_cache

Disable entirely via::

    export TENGRI_DISABLE_JAX_CACHE=1

Programmatic control::

    import tengri

    tengri.enable_persistent_cache("/scratch/jax_cache", min_compile_time_secs=5.0)
    tengri.clear_cache()  # wipe after JAX upgrade

Cache key
---------
JAX's persistent cache keys on:

- jaxpr (function structure)
- abstract input shapes / dtypes
- ``static_argnames`` values (e.g. ``n_samples`` in native VI)
- JAX / jaxlib version
- CPU class / GPU model

Cache misses across:

- New JAX/jaxlib version → wipe with :func:`clear_cache`
- New hardware → wipe
- Different ``n_samples`` (or any other static arg)

Cache hits across:

- Different RNG seeds (PRNGKey concrete values are not part of the key)
- Different processes / notebook restarts / slurm tasks (this is the win)

Eviction
--------
JAX does not auto-evict by default. The ``max_size_bytes`` argument enables
JAX's built-in size cap when supported. Otherwise call :func:`clear_cache`
periodically (e.g. after ``pip install -U jax``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from pathlib import Path

import jax

logger = logging.getLogger(__name__)

# Module-level state for idempotent enable/disable tracking.
_ENABLED_DIR: Path | None = None


_ENV_DIR = "TENGRI_JAX_CACHE_DIR"
_ENV_DISABLE = "TENGRI_DISABLE_JAX_CACHE"
_ENV_MAX_GB = "TENGRI_JAX_CACHE_MAX_GB"

#: Default ceiling for the on-disk compilation cache [bytes].
#:
#: JAX does not evict unless ``jax_compilation_cache_max_size`` is set, and
#: nothing used to set it: ``enable_persistent_cache`` accepted the argument but
#: the auto-enable in ``tengri/__init__.py`` never passed one. Combined with
#: ``min_compile_time_secs=0.05`` -- which deliberately persists every per-filter
#: micro-kernel -- the cache grew without bound and was measured at **141 GB** on
#: a 48 GB machine (#1507).
#:
#: 8 GiB is comfortably more than a working set of compiled kernels while staying
#: small next to a laptop disk. Override with ``TENGRI_JAX_CACHE_MAX_GB``; set it
#: to ``0`` for the old unbounded behavior.
DEFAULT_MAX_CACHE_BYTES = 8 * 1024**3

#: JAX's sentinel for "no limit" is -1, not 0. Measured on jax 0.9.1: an unbounded
#: cache reports ``jax_compilation_cache_max_size == -1``. Passing 0 would mean a
#: zero-byte ceiling, i.e. caching silently switched off -- the opposite of what a
#: user asking to opt out of the cap wants.
UNBOUNDED_CACHE = -1

#: Suffixes JAX's ``LRUCache`` gives the two files that make up one cache entry:
#: ``KEY-cache`` holds the compiled artifact, ``KEY-atime`` an 8-byte
#: little-endian nanosecond timestamp used to order eviction.
#:
#: Mirrored rather than imported: ``jax._src.lru_cache`` is private, and an
#: ImportError here would break ``import tengri`` outright. The copies are pinned
#: against the originals by ``tests/contract/test_jax_cache.py`` so a rename
#: upstream fails a test instead of silently turning :func:`repair_orphaned_atimes`
#: into a no-op.
_CACHE_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"
_ATIME_BYTES = 8
_ATIME_BYTEORDER = "little"


def _eviction_supported() -> bool:
    """True if JAX can actually enforce a cache size cap here.

    JAX guards ``jax_compilation_cache_max_size`` behind ``filelock``. Without it
    the config still accepts the value and then raises on every cache read, so a
    cap set in that environment silently disables the cache rather than bounding
    it. Declared as a dependency, but check at runtime: an older or partial
    install must degrade to "unbounded", never to "broken".
    """
    return importlib.util.find_spec("filelock") is not None


def _resolve_max_size(explicit: int | None) -> int:
    """Cache ceiling in bytes: explicit argument, else env, else the default.

    ``TENGRI_JAX_CACHE_MAX_GB=0`` maps to :data:`UNBOUNDED_CACHE`, restoring the
    pre-#1507 behavior for anyone who wants it.
    """
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get(_ENV_MAX_GB, "").strip()
    if not raw:
        return DEFAULT_MAX_CACHE_BYTES
    try:
        gb = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using the default %.1f GiB cap",
            _ENV_MAX_GB,
            raw,
            DEFAULT_MAX_CACHE_BYTES / 1024**3,
        )
        return DEFAULT_MAX_CACHE_BYTES
    if gb <= 0:
        return UNBOUNDED_CACHE
    return int(gb * 1024**3)


def _default_cache_dir() -> Path:
    """Resolve the default cache directory.

    Resolution order
    ----------------
    1. ``$TENGRI_JAX_CACHE_DIR`` env var (if non-empty).
    2. Legacy path ``~/.cache/tengri_jax_cache`` (preserves caches built
       by older tengri versions which hard-coded this path).

    XDG-style ``$XDG_CACHE_HOME/tengri_jax_cache`` is honored when
    ``XDG_CACHE_HOME`` is set; otherwise the legacy path under
    ``~/.cache`` is used.

    Returns
    -------
    Path
    """
    env_dir = os.environ.get(_ENV_DIR, "").strip()
    if env_dir:
        return Path(env_dir).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "tengri_jax_cache"

    return Path.home() / ".cache" / "tengri_jax_cache"


def enable_persistent_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    min_compile_time_secs: float = 0.05,
    max_size_bytes: int | None = None,
) -> Path:
    """Enable JAX's persistent on-disk compilation cache.

    Calling this is idempotent: re-calling with the same ``cache_dir``
    is a no-op. Re-calling with a different directory updates the JAX
    config and logs the change.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Cache directory. If ``None``, resolves via
        :func:`_default_cache_dir` (env var → XDG → ``~/.cache``).
    min_compile_time_secs : float
        Minimum compile time before a cached entry is persisted to disk.
        Default 0.05 captures the per-filter ``compute_flux_density``
        micro-kernels (~150–250 ms each) and other orchestrator-component
        ops that dispatch individually under the eager
        :meth:`SEDModel.predict_observables` path. Persisting these is
        what makes a fresh-process ``predict_observables`` go from
        ~12 s cold to ~1 s warm-disk.

        Threshold history: 5.0 (≤ 2026-05-04, missed orchestrator),
        0.5 (≤ 2026-05-22, missed per-filter micro-compiles),
        0.05 (current).
    max_size_bytes : int or None
        If provided and supported by the installed JAX, sets
        ``jax_compilation_cache_max_size`` so JAX evicts old entries
        once the cache exceeds this size. Older JAX versions silently
        ignore this argument.

    Returns
    -------
    Path
        Resolved cache directory (created if missing).

    Notes
    -----
    The cache is opt-out via the ``TENGRI_DISABLE_JAX_CACHE`` env var.
    When that env var is truthy, this function is a no-op and returns
    the resolved-but-not-applied directory.
    """
    global _ENABLED_DIR

    if os.environ.get(_ENV_DISABLE, "").strip():
        logger.debug("TENGRI_DISABLE_JAX_CACHE set; persistent cache not enabled")
        return _default_cache_dir()

    target = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    target.mkdir(parents=True, exist_ok=True)

    if _ENABLED_DIR is not None and target == _ENABLED_DIR:
        return target  # idempotent

    jax.config.update("jax_compilation_cache_dir", str(target))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time_secs))

    cap = _resolve_max_size(max_size_bytes)
    if cap != UNBOUNDED_CACHE and not _eviction_supported():
        # JAX needs filelock to enforce a size cap, and raises
        # "Please install the `filelock` package to set
        # jax_compilation_cache_max_size" on EVERY cache read when the cap is set
        # without it -- which does not merely skip eviction, it breaks the cache.
        # Leaving it unbounded is strictly better than breaking it, so say so and
        # stand down.
        logger.warning(
            "Cannot bound the JAX compilation cache: the `filelock` package is "
            "not installed, and JAX errors on every cache read if a cap is set "
            "without it. The cache at %s is UNBOUNDED -- install filelock, or "
            "run tengri.clear_cache() periodically (it reached 141 GB once, #1507).",
            target,
        )
        cap = UNBOUNDED_CACHE

    try:
        jax.config.update("jax_compilation_cache_max_size", cap)
    except (AttributeError, ValueError) as exc:
        # Older JAX without max_size support. WARNING rather than DEBUG: on such a
        # version the cache is unbounded and only clear_cache() reclaims it, which
        # is exactly how #1507 happened.
        logger.warning(
            "This JAX ignores jax_compilation_cache_max_size (%s), so the "
            "persistent cache at %s is UNBOUNDED. Run tengri.clear_cache() "
            "periodically or set %s=0 to acknowledge.",
            exc,
            target,
            _ENV_MAX_GB,
        )

    if cap != UNBOUNDED_CACHE:
        # Hand JAX a directory it can actually evict from. Eviction is the only
        # consumer of the -atime markers -- and the only thing a missing one
        # breaks -- so an unbounded cache needs no sweep (#1661). Costs one
        # directory listing; entries are stat'd only if they need repairing.
        repair_orphaned_atimes(target)

    if _ENABLED_DIR is None and logger.isEnabledFor(logging.INFO):
        # Only measured for this log line, and the walk is O(files) stat calls
        # (~3.5 us each) on every import. Skip it when nobody is listening.
        size_mb = cache_size_bytes(target) / (1024**2)
        logger.info(
            "tengri JAX persistent cache enabled at %s "
            "(min_compile_time=%.1fs, cap=%.1f GiB, current size=%.1f MB)",
            target,
            min_compile_time_secs,
            cap / 1024**3,
            size_mb,
        )
    else:
        logger.info(
            "tengri JAX persistent cache moved from %s to %s",
            _ENABLED_DIR,
            target,
        )

    _ENABLED_DIR = target
    return target


def orphaned_entries(cache_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """Cache keys whose ``-cache`` file has lost its ``-atime`` companion.

    A non-empty result means the cache cannot accept **any** new entry: JAX's
    eviction sweep reads every marker unconditionally and raises on the first
    one missing, and that sweep runs on every write (#1661). Reads are
    unaffected, so nothing else reveals the condition.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Directory to inspect. Defaults to the currently enabled cache, falling
        back to :func:`_default_cache_dir`.

    Returns
    -------
    list of str
        Orphaned cache keys, unordered. Empty if the directory is absent or
        unreadable.
    """
    target = _resolve_dir(cache_dir)
    try:
        names = {entry.name for entry in os.scandir(target)}
    except OSError:
        # Absent, unreadable, or racing with clear_cache(). Callers treat this
        # as "nothing to do"; a diagnostic must not raise, and a repair that
        # raised would break ``import tengri``.
        return []

    cut = len(_CACHE_SUFFIX)
    return [
        name[:-cut]
        for name in names
        if name.endswith(_CACHE_SUFFIX) and f"{name[:-cut]}{_ATIME_SUFFIX}" not in names
    ]


def repair_orphaned_atimes(cache_dir: str | os.PathLike[str] | None = None) -> int:
    """Give every cache entry the ``-atime`` companion JAX's eviction requires.

    A cache entry is two files, ``KEY-cache`` and ``KEY-atime``.
    ``LRUCache._evict_if_needed`` walks every ``*-cache`` file and reads its
    companion with no ``exists()`` check and no ``try``, so **one** entry missing
    its marker raises ``FileNotFoundError`` for the whole sweep. ``put()`` runs
    that sweep before writing, so a single orphan makes every subsequent cache
    write fail. ``get()`` never runs it, which is why the failure is silent:
    reads keep working and the cache simply stops growing (#1661).

    Missing markers are reconstructed from the artifact's own mtime -- a faithful
    "last known access" -- rather than deleted with the entry. The entries that
    orphan most often are the large ones, because ``put()`` writes the artifact
    and its marker as two operations and the window between them scales with
    value size; those are exactly the compiles worth keeping.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Directory to repair. Defaults to the currently enabled cache, falling
        back to :func:`_default_cache_dir`.

    Returns
    -------
    int
        Number of entries repaired. ``0`` if the directory is absent or
        unreadable.

    Notes
    -----
    Healthy markers are left untouched. Eviction chooses victims by ascending
    timestamp, so restamping live entries would flatten that ordering and make
    eviction arbitrary.

    Deliberately takes no file lock. JAX holds one across a whole ``put``,
    including the artifact write, so contending for it would block ``import
    tengri`` behind another process's multi-hundred-MB write. Every outcome of
    losing the race is benign -- a concurrent ``put`` overwrites our marker with
    its own, and a concurrent eviction leaves a stray marker that nothing reads
    -- so per-entry ``OSError`` is skipped rather than raised.
    """
    target = _resolve_dir(cache_dir)
    repaired = 0
    for key in orphaned_entries(target):
        try:
            mtime_ns = os.stat(os.path.join(target, f"{key}{_CACHE_SUFFIX}")).st_mtime_ns
            (target / f"{key}{_ATIME_SUFFIX}").write_bytes(
                mtime_ns.to_bytes(_ATIME_BYTES, _ATIME_BYTEORDER)
            )
        except OSError:
            # Raced with another process's put or eviction. Both outcomes are
            # benign, and one unrepairable entry must not abandon the rest.
            continue
        repaired += 1

    if repaired:
        logger.warning(
            "Repaired %d orphaned entr%s in the JAX compilation cache at %s. "
            "Cache writes were failing until now; reads were unaffected, so the "
            "cache served hits while quietly refusing to grow (#1661).",
            repaired,
            "y" if repaired == 1 else "ies",
            target,
        )
    return repaired


def is_cache_enabled() -> bool:
    """Return ``True`` if :func:`enable_persistent_cache` has been called."""
    return _ENABLED_DIR is not None


def cache_size_bytes(cache_dir: str | os.PathLike[str] | None = None) -> int:
    """Recursively compute the on-disk size of the persistent cache.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Directory to measure. Defaults to the currently enabled cache,
        falling back to :func:`_default_cache_dir`.

    Returns
    -------
    int
        Total size in bytes. Returns 0 if the directory does not exist.
    """
    target = _resolve_dir(cache_dir)
    if not target.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(target):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                # Race with eviction, broken symlink, etc.: skip.
                continue
    return total


def cache_stats(cache_dir: str | os.PathLike[str] | None = None) -> dict:
    """Report whether the cache cap binds, and on what.

    :func:`cache_size_bytes` answers "how big"; this answers "and is that a
    problem, and would a smaller ``min_compile_time_secs`` help". The two
    questions come apart, which is why this exists as its own call.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Directory to measure. Defaults to the currently enabled cache, falling
        back to :func:`_default_cache_dir`.

    Returns
    -------
    dict
        ``n_entries`` (compiled artifacts, i.e. ``*-cache`` files),
        ``total_bytes``, ``max_bytes`` (the configured cap, or
        :data:`UNBOUNDED_CACHE`), ``fraction_of_cap``, ``largest_bytes``,
        ``n_entries_under_64kb``, ``bytes_under_64kb`` and
        ``top10_bytes`` / ``top100_bytes`` -- the share held by the largest few.

    Notes
    -----
    **Not JIT-compatible** -- it walks the filesystem.

    Measured on a shared four-worktree box on 2026-08-31 (see
    ``bench/reports/2026-08-31_fast_nuts.md`` Finding 5): 2 200 entries filling
    7.91 GB of an 8 GiB cap, of which **1 506 entries were under 64 KB and held
    about 1.2 % of the bytes** while the **largest 100 held 79.6 %**. The cap is
    reached by a few dozen compiled samplers, not by the per-filter micro-kernels
    ``min_compile_time_secs = 0.05`` deliberately persists -- so raising that
    threshold frees essentially nothing, and ``TENGRI_JAX_CACHE_MAX_GB`` is the
    knob that does. Reading ``top100_bytes`` against ``max_bytes`` is how a
    caller tells those two apart for their own workload.

    Examples
    --------
    >>> from tengri.utils.jax_cache import cache_stats
    >>> s = cache_stats()  # doctest: +SKIP
    >>> s["top100_bytes"] / s["total_bytes"]  # doctest: +SKIP
    0.796
    """
    target = _resolve_dir(cache_dir)
    sizes: list[int] = []
    if target.exists():
        for root, _dirs, files in os.walk(target):
            for f in files:
                if not f.endswith(_CACHE_SUFFIX):
                    continue
                try:
                    sizes.append(os.path.getsize(os.path.join(root, f)))
                except OSError:
                    # Race with eviction, broken symlink, etc.: skip.
                    continue
    sizes.sort(reverse=True)
    total = sum(sizes)
    small = [s for s in sizes if s < 65536]
    try:
        cap = int(jax.config.jax_compilation_cache_max_size)
    except Exception:
        cap = UNBOUNDED_CACHE
    return {
        "cache_dir": str(target),
        "n_entries": len(sizes),
        "total_bytes": total,
        "max_bytes": cap,
        "fraction_of_cap": (total / cap) if cap > 0 else None,
        "largest_bytes": sizes[0] if sizes else 0,
        "n_entries_under_64kb": len(small),
        "bytes_under_64kb": sum(small),
        "top10_bytes": sum(sizes[:10]),
        "top100_bytes": sum(sizes[:100]),
    }


def clear_cache(cache_dir: str | os.PathLike[str] | None = None) -> int:
    """Remove all entries from the persistent cache directory.

    The directory itself is preserved (the cache stays enabled). Use
    after ``pip install -U jax`` to evict stale-key artifacts, or
    periodically for general housekeeping.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Directory to clear. Defaults to the currently enabled cache,
        falling back to :func:`_default_cache_dir`.

    Returns
    -------
    int
        Bytes reclaimed. Reported because the cache is otherwise invisible:
        it reached 141 GB before anyone looked (#1507).
    """
    target = _resolve_dir(cache_dir)
    if not target.exists():
        return 0
    freed = cache_size_bytes(target)
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                continue
    logger.info(
        "Cleared tengri JAX persistent cache at %s (%.1f MB reclaimed)",
        target,
        freed / 1024**2,
    )
    return freed


def _resolve_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve a cache directory argument to a concrete Path."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if _ENABLED_DIR is not None:
        return _ENABLED_DIR
    return _default_cache_dir()
