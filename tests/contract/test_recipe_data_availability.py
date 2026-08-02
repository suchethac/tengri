# SPDX-License-Identifier: BSD-3-Clause
"""``list_recipes()`` must say which recipes need data that does not ship (#1462 §3).

The menu presented all ten recipes as equals while ``unified_agn`` cannot
produce a number without a Synthesizer AGN grid that tengri does not package.
A recipe is by definition the thing a new user is told to start from, so
"this one needs a download" belongs in the table rather than in a traceback.

Worse, the failure is **deferred**: ``unified_agn`` builds successfully and
raises at the first ``predict``, so a user gets a model object back and
believes it worked.

The status is computed by calling the loader's own resolver, not by
re-deriving the search path, so the column cannot claim ``ready`` while the
loader disagrees. These tests pin that coupling — a column that is merely
*present* is worthless if it is wrong.
"""

import pytest

import tengri

pytestmark = pytest.mark.contract


def test_every_recipe_reports_a_data_status():
    rows = list(tengri.list_recipes())
    assert rows, "no recipes listed"
    missing = [r["name"] for r in rows if not r.get("data")]
    assert not missing, f"recipes with no data status: {missing}"


def test_recipes_needing_nothing_extra_say_ready():
    """Guard against a column that flags everything and so says nothing."""
    rows = {r["name"]: r["data"] for r in tengri.list_recipes()}
    for name in ("photoz", "high_z", "dust_demo", "star_forming_photometry"):
        assert rows[name] == "ready", f"{name} should need no extra data, got {rows[name]!r}"


def test_the_status_follows_the_loader_not_a_hardcoded_list(monkeypatch, tmp_path):
    """The column must track real availability in both directions.

    A hardcoded "unified_agn is unavailable" would pass a one-directional
    test. Point the loader's env var at a directory containing the file it
    looks for and the row must flip to ``ready``; unset it and it must flip
    back.
    """
    name = "unified_agn"

    monkeypatch.delenv("TENGRI_SYNTHESIZER_AGN_GRID_DIR", raising=False)
    before = {r["name"]: r["data"] for r in tengri.list_recipes()}[name]

    (tmp_path / "test_grid_agn-nlr.hdf5").write_bytes(b"")
    monkeypatch.setenv("TENGRI_SYNTHESIZER_AGN_GRID_DIR", str(tmp_path))
    after = {r["name"]: r["data"] for r in tengri.list_recipes()}[name]

    assert after == "ready", f"grid present but status is {after!r}"
    assert before != after, (
        "the status did not change when the grid appeared — it is not reading "
        "the loader's resolver"
    )
    assert "synthesizer-download" in before, (
        f"the unavailable status must name the fetch command, got {before!r}"
    )


def test_the_status_never_raises_even_if_a_recipe_is_broken(monkeypatch):
    """The menu must survive a recipe that cannot be called at all.

    Discovery is what a user reaches for when something is already wrong;
    it must not be the thing that breaks.
    """
    import tengri.registry as reg

    def _explode(_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(reg, "_recipe_data_status", _explode)
    rows = list(tengri.list_recipes())
    assert rows, "list_recipes() died when the status helper raised"
    assert all(r["data"] == "unknown" for r in rows)


def test_listing_recipes_stays_cheap():
    """The status is computed per call, so it must not build anything."""
    import time

    tengri.list_recipes()  # warm imports
    start = time.perf_counter()
    for _ in range(5):
        tengri.list_recipes()
    per_call = (time.perf_counter() - start) / 5
    assert per_call < 0.5, f"list_recipes() takes {per_call:.2f}s — it is building something"
