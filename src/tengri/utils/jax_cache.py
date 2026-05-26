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

    if max_size_bytes is not None:
        try:
            jax.config.update("jax_compilation_cache_max_size", int(max_size_bytes))
        except (AttributeError, ValueError) as exc:
            # Older JAX without max_size support — surface as debug only.
            logger.debug("jax_compilation_cache_max_size unavailable: %s", exc)

    if _ENABLED_DIR is None:
        size_mb = cache_size_bytes(target) / (1024**2)
        logger.info(
            "tengri JAX persistent cache enabled at %s "
            "(min_compile_time=%.1fs, current size=%.1f MB)",
            target,
            min_compile_time_secs,
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
                # Race with eviction, broken symlink, etc. — skip.
                continue
    return total


def clear_cache(cache_dir: str | os.PathLike[str] | None = None) -> None:
    """Remove all entries from the persistent cache directory.

    The directory itself is preserved (the cache stays enabled). Use
    after ``pip install -U jax`` to evict stale-key artifacts, or
    periodically for general housekeeping.

    Parameters
    ----------
    cache_dir : str or PathLike or None
        Directory to clear. Defaults to the currently enabled cache,
        falling back to :func:`_default_cache_dir`.
    """
    target = _resolve_dir(cache_dir)
    if not target.exists():
        return
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                continue
    logger.info("Cleared tengri JAX persistent cache at %s", target)


def _resolve_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve a cache directory argument to a concrete Path."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if _ENABLED_DIR is not None:
        return _ENABLED_DIR
    return _default_cache_dir()
