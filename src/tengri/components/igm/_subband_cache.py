# SPDX-License-Identifier: BSD-3-Clause
"""Content-hashed cache for the sub-band IGM transmission table.

``IGMSEDComponent.subband_node_transmission`` evaluates the IGM transmission at
every sub-band quadrature node on the photometry z-grid; a Python loop over
the z axis, one :func:`igm_absorption` call per redshift. It is a **build-time
constant**, so neither the JAX compilation cache nor the photometry z-table
cache covers it, and every ``SEDModel.build`` re-paid it in full: measured at
~9 s per build on a free-redshift model against ~0.3 s with ``igm='none'``,
identical across three consecutive builds in one process (#1453).

The z-table computed in the same call is content-hashed to
``~/.cache/tengri_precomp``; this is the piece that was left over.

Key completeness
----------------
A cross-process cache whose key omits an input returns wrong physics silently
and persistently; that is exactly how #1122 happened, and the z-table's own
key carries a comment about it. So the key here is every field of a frozen
request dataclass (``SubbandRequest``, ``SubbandBandRequest``): a new input is
a new field, and the perturbation test derives its population from
``dataclasses.fields``, so a field that does not move the digest is caught.

``subband_node_transmission`` reads exactly three things on its cacheable path:

* ``subband_waves_rest``: the node wavelengths, by content;
* ``z_grid``: the redshift grid, by content;
* ``config.igm_model``: which transmission law.

The other two config fields cannot reach it: ``igm_patchy`` and ``use_dla``
both read *free parameters* at apply time, so the method returns ``None`` for
them before any of this runs and the exact path is kept. They are folded into
the key regardless, so that if that gate is ever loosened a stale entry cannot
outlive the change.

:data:`_CACHE_VERSION` invalidates every entry when the transmission formula or
the table layout changes; bump it in the same commit as any such change.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from tengri._cache_keys import array_key, frozen_dataclass_key, stable_digest

__all__ = ["cache_dir", "cache_key", "clear_memo", "load", "memo_get", "memo_put", "store"]

#: Bump when the stored table's meaning changes (transmission formula, node
#: layout, dtype convention). Entries keyed with an older version are ignored.
_CACHE_VERSION = 3

#: In-process memo. The on-disk layer alone still costs an npz read per build,
#: and #1453 measured repeat builds *within* one process re-paying in full.
_MEMO: dict[str, np.ndarray] = {}


@dataclasses.dataclass(frozen=True)
class SubbandRequest:
    """Request dataclass for sub-band IGM transmission table, keyed by every field.

    All fields are hashable (tuples, arrays via array_key, scalars, strings).
    Each field represents a dimension of the transmission table computation.
    """

    #: Schema version: bump when the table structure changes
    version: int
    #: Array key for rest-frame sub-band node wavelengths
    waves_rest: tuple
    #: Array key for redshift grid
    z_grid: tuple
    #: IGM transmission law name ("inoue", "madau", etc.)
    igm_model: str
    #: Whether patchy reionization is modeled (currently unused on this path)
    igm_patchy: bool
    #: Whether DLAs are included (currently unused on this path)
    use_dla: bool
    #: Whether JAX X64 mode is enabled
    x64: bool
    #: JAX backend name
    backend: str


@dataclasses.dataclass(frozen=True)
class SubbandBandRequest(SubbandRequest):
    """Extended request for band-factor (filter-averaged) transmission table.

    Inherits all SubbandRequest fields and adds filter and convolution convention.

    The band path carries ``igm_patchy`` and ``use_dla`` at their constant
    ``False``: the precomputable path is disabled when either is set
    (``IGMSEDComponent.precompute``), so they are kept at their constants rather
    than dropped from the key.
    """

    #: Tuple of (wave_key, trans_key) per filter, in order
    filters: tuple
    #: Filter convolution convention (str)
    convention: str


def cache_dir() -> Path | None:
    """Resolve the on-disk cache directory, or ``None`` when disabled.

    Shares ``TENGRI_DISABLE_PRECOMP_CACHE`` / ``TENGRI_PRECOMP_CACHE_DIR`` with
    the photometry z-table, so one pair of knobs governs both halves of the
    build-time precompute.
    """
    if os.environ.get("TENGRI_DISABLE_PRECOMP_CACHE"):
        return None
    custom = os.environ.get("TENGRI_PRECOMP_CACHE_DIR")
    return Path(custom) if custom else Path.home() / ".cache" / "tengri_precomp"


def cache_key(waves_rest, z_grid, igm_model, *, igm_patchy=False, use_dla=False) -> str:
    """Content hash over everything the transmission table depends on.

    Includes session precision (jax_enable_x64) and JAX backend since the
    transmission values are computed through JAX and inherit session precision.
    Contamination without these: float32 session writes an entry, float64
    session reads it (~1e-7 relative error silently poisoning precision
    benchmarks/parity tests that share a cache dir between arms). Issue #2024.

    The key is built from a frozen SubbandRequest dataclass, ensuring every
    input dimension is represented: omitting a field cannot be a silent bug
    because the dataclass has every field explicitly listed and type-annotated.

    Parameters
    ----------
    waves_rest : array_like
        Sub-band node wavelengths [Angstrom], rest frame. Hashed by content
        *and* shape: two tables with the same values in a different layout
        index different nodes.
    z_grid : array_like
        Redshift grid the table is evaluated on.
    igm_model : str
        Transmission law name (``"inoue"``, ``"madau"``, ...).
    igm_patchy, use_dla : bool, optional
        Included for the reason given in the module docstring: the current
        gate makes them unreachable, and the key should not depend on that
        staying true.

    Returns
    -------
    str
        Hex digest.
    """
    import jax

    req = SubbandRequest(
        version=_CACHE_VERSION,
        waves_rest=array_key(waves_rest),
        z_grid=array_key(z_grid),
        igm_model=str(igm_model),
        igm_patchy=bool(igm_patchy),
        use_dla=bool(use_dla),
        x64=bool(jax.config.jax_enable_x64),
        backend=jax.default_backend(),
    )

    return stable_digest(repr(frozen_dataclass_key(req)).encode())


def band_factor_key(wave_rest, filter_waves, filter_trans, z_grid, igm_model, convention) -> str:
    """Content hash for the filter-averaged band-factor table.

    A second key rather than a parameter on :func:`cache_key`, because the two
    tables consume different things and sharing one key function would invite
    hashing a field that only one of them reads.

    ``precompute_band_factors`` integrates the transmission against each filter
    response, so its result depends on the filter curves and the convolution
    convention as well: inputs the sub-band node table never sees.

    Includes session precision (jax_enable_x64) and JAX backend since the
    band factor values are computed through JAX and inherit session precision.
    Contamination without these: float32 session writes an entry, float64
    session reads it (~1e-7 relative error silently poisoning precision
    benchmarks/parity tests that share a cache dir between arms). Issue #2024.

    The key is built from a frozen SubbandBandRequest dataclass, ensuring every
    input dimension is represented: omitting a field cannot be a silent bug.

    Parameters
    ----------
    wave_rest : array_like
        Rest-frame wavelength grid [Angstrom].
    filter_waves, filter_trans : sequence of array_like
        Per-filter wavelength and transmission curves. Hashed in order: the
        table is indexed by filter position, so a reordering is a different
        table.
    z_grid : array_like
        Redshift nodes.
    igm_model : str
        Transmission law name.
    convention : Any
        Filter convolution convention (ADR-0017); changes the integral.

    Returns
    -------
    str
        Hex digest.
    """
    import jax

    # Convert filter arrays to array_key tuples
    filters_keyed = tuple(
        (array_key(fw), array_key(ft)) for fw, ft in zip(filter_waves, filter_trans)
    )

    req = SubbandBandRequest(
        version=_CACHE_VERSION,
        waves_rest=array_key(wave_rest),
        z_grid=array_key(z_grid),
        igm_model=str(igm_model),
        igm_patchy=False,
        use_dla=False,
        filters=filters_keyed,
        convention=str(convention),
        x64=bool(jax.config.jax_enable_x64),
        backend=jax.default_backend(),
    )

    return stable_digest(repr(frozen_dataclass_key(req)).encode())


def _enabled() -> bool:
    """Whether caching is on at all.

    Both layers honor ``TENGRI_DISABLE_PRECOMP_CACHE``, not just the disk one.
    A kill-switch that leaves an in-process memo serving is a partial switch
    that reads as total: ``tests/conftest.py`` disables the precomp cache for
    hermeticity, and a live memo would quietly share a table between two tests
    that each believe they built from scratch. Read per call rather than at
    import, so a test that sets the variable with ``monkeypatch`` takes effect.
    """
    return not os.environ.get("TENGRI_DISABLE_PRECOMP_CACHE")


def memo_get(key: str):
    """In-process lookup, or ``None`` (always ``None`` when caching is off)."""
    if not _enabled():
        return None
    return _MEMO.get(key)


def memo_put(key: str, table) -> None:
    """Record in the in-process memo, unless caching is off."""
    if not _enabled():
        return
    _MEMO[key] = np.asarray(table)


def clear_memo() -> None:
    """Drop the in-process memo. Used by tests; harmless at runtime."""
    _MEMO.clear()


def load(key: str):
    """Load a cached table from disk, or ``None`` on any miss.

    Fails soft on a corrupt or unreadable entry: a cache is an optimization,
    and a bad file must degrade to recomputation rather than take the build
    down.
    """
    directory = cache_dir()
    if directory is None:
        return None
    path = directory / f"igm_subband_{key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return data["table"]
    except Exception:
        return None


def store(key: str, table) -> None:
    """Persist a table, silently declining if the cache is unwritable.

    Writes via a temporary file and renames, so a process interrupted
    mid-write cannot leave a truncated entry that a later run would read as
    valid.
    """
    directory = cache_dir()
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"igm_subband_{key}.npz"
        # The temp name must itself end in ``.npz``: ``savez_compressed``
        # appends the extension when it is absent, so a name like
        # ``foo.npz.tmp123`` is written as ``foo.npz.tmp123.npz`` and the
        # rename below silently finds nothing. That failure is invisible;
        # every build recomputes and the cache simply never hits.
        tmp = directory / f"igm_subband_{key}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, table=np.asarray(table))
        os.replace(tmp, path)
    except Exception:
        return
