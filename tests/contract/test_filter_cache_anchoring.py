# SPDX-License-Identifier: BSD-3-Clause
"""The SVO filter cache must not be addressed relative to the working directory.

``_DEFAULT_CACHE_DIR`` was the string ``"data/filters"``, and
:func:`~tengri.observation.filters.download_filter` does
``Path(cache_dir).mkdir(parents=True, exist_ok=True)`` on it. Run anything from
a directory other than the repository root -- a notebook, a scheduled job,
sphinx-gallery, a user's own script -- and two things happen silently: a stray
``data/filters/`` appears wherever the process started, and the filters that
*are* already cached are missed, so every curve is re-fetched from the SVO
service over the network.

That is the #1486 failure class living in the library rather than in a notebook,
so it reached every caller. The fix routes the default through the same
read/write split the rest of tengri's data story uses:
:func:`~tengri._data_setup.data_dirs` to find an existing cache,
:func:`~tengri._data_setup.download_dir` to decide where a new one is written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def test_default_is_a_resolved_path_not_a_relative_string():
    """The default must be absolute, or it means different directories per cwd."""
    from tengri.observation.filters import default_filter_cache_dir

    assert default_filter_cache_dir().is_absolute()


def test_an_existing_cache_is_found_from_an_unrelated_working_directory(tmp_path, monkeypatch):
    """A configured cache is used wherever the process happens to start."""
    import tengri._data_setup as ds
    from tengri.observation.filters import default_filter_cache_dir

    configured = tmp_path / "data"
    (configured / "filters").mkdir(parents=True)
    monkeypatch.setenv(ds.TENGRI_DATA_ENV, str(configured))

    elsewhere = tmp_path / "some" / "unrelated" / "cwd"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)

    assert default_filter_cache_dir() == configured / "filters"


def test_a_fresh_cache_is_written_where_the_loaders_will_look(tmp_path, monkeypatch):
    """With no cache yet, the write location must be one ``data_dirs()`` searches.

    A write path the read path does not cover is how a cache ends up populated
    and permanently unused.

    ``data_dirs`` is replaced rather than merely redirected by environment: in a
    development checkout it also yields the repository's own ``data/``, which
    already holds a ``filters/`` -- so the found-an-existing-cache branch would
    win and this fallback would never run.
    """
    import tengri._data_setup as ds
    from tengri.observation.filters import default_filter_cache_dir

    scratch = tmp_path / "scratch"
    monkeypatch.setenv(ds.TENGRI_DATA_ENV, str(scratch))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ds, "data_dirs", lambda: [scratch])

    target = default_filter_cache_dir()
    assert target == scratch / "filters"
    assert target.parent in ds.data_dirs(), "filters would be cached where nothing reads"


def test_no_public_entry_point_defaults_to_a_relative_directory():
    """Pin the rule, so the literal cannot come back by way of a refactor.

    The default has to be resolved per call, not frozen at import: a module
    constant computed once would capture whatever the working directory was
    when ``tengri`` was first imported, which is the same bug wearing a hat.
    So the contract is that every entry point defaults to ``None`` and asks
    :func:`default_filter_cache_dir` at call time.
    """
    import inspect

    import tengri.observation.filters as f
    from tengri.observation.photometry_config import Photometry

    entry_points = [
        f.download_filter,
        f.load_filter,
        f.load_filter_set,
        Photometry.from_names,
    ]
    offenders = []
    for fn in entry_points:
        default = inspect.signature(fn).parameters["cache_dir"].default
        if isinstance(default, str | Path) and not Path(default).is_absolute():
            offenders.append(f"{fn.__qualname__}(cache_dir={default!r})")
    assert not offenders, "cwd-relative filter-cache defaults: " + ", ".join(offenders)


# ── A partial cache must not hide a complete one ──────────────────────────
#
# #1486 anchored the cache so the *directory* no longer depended on the working
# directory. That left the harder half: resolution still picked the first
# ancestor that merely owned a ``filters/`` folder, whether or not it held the
# curve being asked for. Two committed partial caches -- ten curves under
# ``examples/advanced/data/filters/``, five under ``examples/inference/`` --
# therefore shadowed the 249 in ``data/filters/`` for every gallery example,
# because the runner chdirs into each script's directory. Every band outside
# those partials was re-fetched from SVO on every CI run, and nothing said so:
# a miss is indistinguishable from a cold cache, so it failed open.


def _seed(directory, *names):
    """Create ``directory`` and write a minimal two-column curve per name."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("1000.0 0.0\n2000.0 1.0\n3000.0 0.0\n")
    return directory


def test_a_partial_cache_does_not_hide_a_complete_one(tmp_path, monkeypatch):
    """A curve missing from the nearest cache is still found further up.

    ``data_dirs`` is replaced rather than redirected by environment for the
    reason the fallback test above gives: in a development checkout it also
    yields the repository's own ``data/filters/``, which holds every curve, so
    a partial shadow could never be observed to lose.
    """
    import tengri._data_setup as ds
    from tengri.observation.filters import find_cached_filter

    complete = _seed(tmp_path / "root" / "filters", "NEAR.dat", "FAR.dat")
    partial = _seed(tmp_path / "nested" / "filters", "NEAR.dat")
    monkeypatch.setattr(ds, "data_dirs", lambda: [partial.parent, complete.parent])

    # The nearer cache still wins for what it actually holds.
    assert find_cached_filter("NEAR.dat") == partial / "NEAR.dat"
    # And no longer vetoes what it does not.
    assert find_cached_filter("FAR.dat") == complete / "FAR.dat"
    # A curve no cache holds reports honestly rather than guessing a path.
    assert find_cached_filter("ABSENT.dat") is None


def test_a_shadowed_curve_loads_without_reaching_for_the_network(tmp_path, monkeypatch):
    """The end-to-end claim: a partial cache must not trigger a download.

    Asserting on :func:`find_cached_filter` alone would not catch a
    ``download_filter`` that ignores it, which is precisely how the original
    defect survived -- the pieces were each defensible and the composition was
    not. ``_fetch_from_svo`` is replaced with a detonator so any fetch fails
    loudly instead of silently succeeding over the network.
    """
    import tengri._data_setup as ds
    import tengri.observation.filters as f

    complete = _seed(tmp_path / "root" / "filters", "GALEX_GALEX_FUV.dat")
    partial = _seed(tmp_path / "nested" / "filters", "SLOAN_SDSS_u.dat")
    monkeypatch.setattr(ds, "data_dirs", lambda: [partial.parent, complete.parent])

    def _detonate(svo_id, short_name=None):
        raise AssertionError(f"reached SVO for {svo_id!r}, which is cached on disk")

    monkeypatch.setattr(f, "_fetch_from_svo", _detonate)

    wave, trans = f.download_filter("GALEX/GALEX.FUV")
    assert wave.shape == (3,)
    assert trans.max() == 1.0


def test_a_failed_download_leaves_no_empty_cache_behind(tmp_path, monkeypatch):
    """A fetch that raises must not create the directory it would have written.

    ``download_filter`` ran ``cache_path.mkdir(parents=True)`` before deciding
    whether it needed to download at all, so a fetch that then failed -- SVO
    unreachable, or ``astroquery`` simply not installed -- left an *empty*
    ``filters/`` directory behind. That is the worst residue of the lot: an
    empty cache holds nothing, yet under directory-level resolution it was
    still enough to win and hide a populated cache further up. The mkdir now
    happens only on the path that actually writes a curve.
    """
    import tengri.observation.filters as f

    def _unavailable(svo_id, short_name=None):
        raise ImportError("astroquery is not installed")

    monkeypatch.setattr(f, "_fetch_from_svo", _unavailable)
    target = tmp_path / "cwd" / "data" / "filters"

    with pytest.raises(ImportError):
        f.download_filter("GALEX/GALEX.FUV", cache_dir=target)

    assert not target.exists(), "a failed download left an empty cache that can shadow"


def test_no_filter_curves_are_shipped_outside_the_canonical_cache():
    """Duplicate curves under ``examples/`` are what created the shadows.

    All twenty were byte-identical copies of files already in ``data/filters/``,
    so they bought nothing and cost the resolution bug above. The library fix
    makes a stray cache harmless; this keeps the duplicates from coming back.
    """
    repo = Path(__file__).resolve().parents[2]
    examples = repo / "examples"
    if not examples.is_dir():
        pytest.skip("not a source checkout")

    strays = sorted(str(p.relative_to(repo)) for p in examples.rglob("data/filters/*.dat"))
    assert not strays, (
        "filter curves belong in data/filters/ only; these shadow it for any "
        "example run from their directory: " + ", ".join(strays)
    )
