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
