# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #1209 — the data bootstrap must be one story.

The two documented first steps did not connect. ``tengri.download_ssp()`` wrote
one file; ``tengri.load_ssp()`` looked for another. Around them, five separate
answers to "where is my data?" had drifted apart:

* ``$TENGRI_DATA_DIR`` governed *writes*, ``$TENGRI_DATA`` governed *finds*, and
  they never met — setting either alone left the other half pointed elsewhere.
* Two ``download_ssp`` functions with incompatible signatures, both hitting the
  same URL, only one honoring the environment.
* ``doctor()`` and ``bench._find_ssp`` globbed ``ssp_*.h5``, which cannot match
  the ``fsps_*.h5`` files ``download_ssp()`` actually writes, so a correctly
  populated install reported "no SSP data found".

* ``download_ssp()`` and ``load_ssp()`` each hardcoded their own default
  filename, so nothing forced agreement — the fetched grid and the loaded grid
  were different files, and the loaded one was not obtainable at all.

These pin the invariants that make those states unrepresentable.

Design: docs/internal/specs/2026-07-17-consolidate-approx-and-bootstrap-design.md
"""

import warnings

import pytest

from tengri._data_setup import (
    KNOWN_SSP_FILENAMES,
    TENGRI_DATA_ENV,
    data_dirs,
    download_dir,
    find_ssp_files,
)

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def clean_env(monkeypatch):
    """Neither data environment variable set."""
    monkeypatch.delenv("TENGRI_DATA_DIR", raising=False)
    monkeypatch.delenv("TENGRI_DATA", raising=False)


# ── one environment variable, both halves ────────────────────────────────


def test_downloads_land_where_loaders_look(clean_env, monkeypatch, tmp_path):
    """The core invariant: writes go to a directory reads search first.

    The bug: ``$TENGRI_DATA_DIR`` moved downloads while ``doctor()`` kept
    looking at ``$TENGRI_DATA``, so a user who set one saw the other half of the
    library disagree about where the data was.
    """
    monkeypatch.setenv(TENGRI_DATA_ENV, str(tmp_path))
    assert download_dir() == tmp_path
    assert data_dirs()[0] == tmp_path, (
        "downloads are written to a directory the loaders do not search first"
    )


def test_default_download_dir_is_searched(clean_env, tmp_path, monkeypatch):
    """With no environment set, downloads still land on the search path."""
    monkeypatch.chdir(tmp_path)
    assert download_dir() in data_dirs(), (
        "the default download directory is not one the loaders search"
    )


def test_legacy_env_var_is_honored_with_a_warning(clean_env, monkeypatch, tmp_path):
    """$TENGRI_DATA keeps working, warns, and now governs writes too.

    It previously governed only ``doctor()``'s search, so honoring it for writes
    as well is what makes the single variable sufficient.
    """
    monkeypatch.setenv("TENGRI_DATA", str(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = download_dir()
    assert resolved == tmp_path
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "the deprecated $TENGRI_DATA spelling was honored silently"
    assert TENGRI_DATA_ENV in str(deprecations[0].message), (
        "the deprecation warning does not name its replacement"
    )


def test_canonical_env_var_wins_over_legacy(clean_env, monkeypatch, tmp_path):
    """With both set, the canonical spelling wins and no warning fires."""
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    monkeypatch.setenv(TENGRI_DATA_ENV, str(canonical))
    monkeypatch.setenv("TENGRI_DATA", str(legacy))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = download_dir()
    assert resolved == canonical
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)], (
        "warned about the deprecated variable while using the canonical one"
    )


# ── one SSP finder, and it can see what download_ssp writes ──────────────


def test_find_ssp_files_sees_catalog_filenames(clean_env, monkeypatch, tmp_path):
    """The glob bug, pinned: a downloaded grid must be discoverable.

    ``doctor()`` globbed ``ssp_*.h5``, which matches no catalog filename, so a
    correctly populated install reported "no SSP data found".
    """
    monkeypatch.setenv(TENGRI_DATA_ENV, str(tmp_path))
    catalog_name = next(iter(sorted(KNOWN_SSP_FILENAMES)))
    (tmp_path / catalog_name).write_bytes(b"")

    found = [p.name for p in find_ssp_files()]
    assert catalog_name in found, (
        f"{catalog_name} — a filename download_ssp() writes — is invisible to the SSP finder"
    )


def test_find_ssp_files_sees_locally_generated_grids(clean_env, monkeypatch, tmp_path):
    """Locally generated ``ssp_*`` grids (incl. wNE) stay discoverable."""
    monkeypatch.setenv(TENGRI_DATA_ENV, str(tmp_path))
    (tmp_path / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").write_bytes(b"")
    assert any(p.name.startswith("ssp_") for p in find_ssp_files())


def test_find_ssp_files_ignores_component_templates(clean_env, monkeypatch, tmp_path):
    """Template libraries are not SSP grids.

    Guards the obvious over-correction: globbing ``*.h5`` would report
    ``dl07_templates.h5`` as an SSP and make ``doctor()`` claim a usable install
    where there is none.

    Scoped to ``tmp_path`` rather than asserting an empty result: ``data_dirs``
    also walks ancestors by design, so a developer running from a populated
    checkout legitimately sees real grids alongside.
    """
    monkeypatch.setenv(TENGRI_DATA_ENV, str(tmp_path))
    for name in ("dl07_templates.h5", "skirtor_templates_v3.h5", "cue_weights.npz"):
        (tmp_path / name).write_bytes(b"")

    from_tmp = [p.name for p in find_ssp_files() if p.parent == tmp_path]
    assert from_tmp == [], f"component templates reported as SSP grids: {from_tmp}"


def test_doctor_and_bench_share_one_answer(clean_env, monkeypatch, tmp_path):
    """The diagnostic and the benchmark must not disagree about the install."""
    monkeypatch.setenv(TENGRI_DATA_ENV, str(tmp_path))
    catalog_name = next(iter(sorted(KNOWN_SSP_FILENAMES)))
    (tmp_path / catalog_name).write_bytes(b"")

    from tengri.bench import _find_ssp

    assert _find_ssp() == find_ssp_files()[0]


# ── one download_ssp ─────────────────────────────────────────────────────


def test_data_download_ssp_is_deprecated_and_delegates(clean_env, monkeypatch, tmp_path):
    """The duplicate twin warns and routes to the canonical implementation.

    The two were separate implementations of one job, reachable under the same
    name with incompatible signatures, and only one honored the environment.
    """
    import tengri._data_setup as ds
    import tengri.data as td

    seen = {}

    def _fake(name, dest=None, force=False, progress=True):
        seen.update(name=name, dest=dest, force=force, progress=progress)
        return tmp_path / "x.h5"

    monkeypatch.setattr(ds, "download_ssp", _fake)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        td.download_ssp("fsps_prsc_miles_chabrier.h5", dest_dir=tmp_path, overwrite=True)

    assert [w for w in caught if issubclass(w.category, DeprecationWarning)], (
        "tengri.data.download_ssp did not warn"
    )
    assert seen["dest"] == tmp_path, "dest_dir was not forwarded as dest"
    assert seen["force"] is True, "overwrite was not forwarded as force"


def test_download_ssp_progress_is_not_silently_dropped():
    """``progress`` reaches the downloader rather than vanishing in the shim.

    A kwarg accepted and ignored is the shape of several shipped no-ops in this
    codebase; the delegation must forward it, not swallow it.
    """
    import inspect

    from tengri._data_setup import _download_file, download_ssp

    assert "progress" in inspect.signature(download_ssp).parameters
    assert "progress" in inspect.signature(_download_file).parameters


@pytest.mark.parametrize(
    "name,expected",
    [
        ("fsps_prsc_miles_chabrier", "fsps_prsc_miles_chabrier.h5"),  # short identifier
        ("fsps_prsc_miles_chabrier.h5", "fsps_prsc_miles_chabrier.h5"),  # catalog filename
        ("some_new_grid.h5", "some_new_grid.h5"),  # live-catalog filename
    ],
)
def test_download_ssp_accepts_both_spellings(name, expected):
    """One function serves short identifiers and catalog filenames alike.

    The twin took filenames, the canonical one took short names; users met
    whichever their docs mentioned. Accepting both is what lets the twin retire
    without dropping capability.
    """
    from tengri._data_setup import _resolve_ssp_filename

    assert _resolve_ssp_filename(name) == expected


def test_download_ssp_rejects_typos_and_paths():
    """A bare unknown word is a typo; a path is never a catalog name."""
    from tengri._data_setup import _resolve_ssp_filename

    with pytest.raises(KeyError, match="Unknown SSP name"):
        _resolve_ssp_filename("fsps_prsc_miles_chabier")  # typo, no .h5
    with pytest.raises(ValueError, match="not a path"):
        _resolve_ssp_filename("../../etc/passwd.h5")


# ── one default SSP: download and load cannot drift apart ────────────────


def test_download_and_load_share_one_default():
    """THE bug: ``download_ssp()`` wrote one grid and ``load_ssp()`` read another.

    Each hardcoded its own filename, so nothing forced agreement and they drifted:
    a fresh user fetched ``fsps_prsc_miles_chabrier.h5`` and the loader then looked
    for ``ssp_prsc_miles_chabrier_wNE_...h5`` — a file absent from both
    ``list_known_ssps()`` and the hosted catalog. One constant makes that
    unrepresentable.
    """
    import inspect

    from tengri._data_setup import _KNOWN_SSPS, DEFAULT_SSP, download_ssp
    from tengri.components.stellar.sps import dsps_wrapper

    # download_ssp must default to DEFAULT_SSP
    assert inspect.signature(download_ssp).parameters["name"].default == DEFAULT_SSP, (
        "download_ssp no longer defaults to DEFAULT_SSP — the two can drift again"
    )
    # Behavioral test: both load_ssp() (no args) and load_ssp(DEFAULT_SSP) must return
    # the same grid (same metallicity axis shape proves they're the same file)
    ssp_no_args = dsps_wrapper.load_ssp()
    ssp_explicit = dsps_wrapper.load_ssp(DEFAULT_SSP)
    assert ssp_no_args is not None, "load_ssp() with no arguments returned None"
    assert ssp_explicit is not None, f"load_ssp({DEFAULT_SSP!r}) returned None"
    # Both must have matching grid dimensions (ssp_lgmet shape is an SSP identifier)
    assert ssp_no_args.ssp_lgmet.shape == ssp_explicit.ssp_lgmet.shape, (
        f"load_ssp() and load_ssp({DEFAULT_SSP!r}) returned different grids: "
        f"ssp_lgmet shape {ssp_no_args.ssp_lgmet.shape} vs {ssp_explicit.ssp_lgmet.shape}. "
        f"The no-argument default is not DEFAULT_SSP."
    )
    # The loaded SSP must match the known catalog entry
    assert DEFAULT_SSP in _KNOWN_SSPS, "the default must be fetchable by download_ssp()"


def test_default_ssp_is_bare_stellar():
    """The default must be bare-stellar: the catalog ships nothing else.

    A wNE (nebular-baked) default is unobtainable via ``download_ssp()`` *and*
    rejected by the Cue backend, which is what half the recipe catalog uses.
    """
    from tengri._data_setup import DEFAULT_SSP

    assert "wNE" not in DEFAULT_SSP, "the default SSP must not be a nebular-baked grid"


@pytest.mark.parametrize(
    "recipe_name",
    ["star_forming_photometry", "quiescent_z0", "stochastic_sfh_jwst", "agn_panchromatic"],
)
def test_default_ssp_builds_the_cue_recipes(recipe_name):
    """The documented ``load_ssp()`` + ``recipes.*`` pattern must actually build.

    These four recipes use the Cue nebular backend, which *rejects* wNE grids
    (``CueWNESSPError``). Under the old wNE default this exact pattern — the one
    ``docs/internal/recipes/jwst_nirspec_spectroscopy.md`` prints — raised on the first
    line a new user would type. Five of ten recipes were unbuildable that way.
    """
    tengri = pytest.importorskip("tengri")
    from tengri.observation.observation import Observation
    from tengri.observation.photometry_config import Photometry

    try:
        ssp = tengri.load_ssp()
    except FileNotFoundError:
        pytest.skip("default SSP not available")

    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
    cfg = getattr(tengri.recipes, recipe_name)()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tengri.SEDModel.build(ssp_data=ssp, observation=obs, **cfg)
