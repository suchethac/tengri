# SPDX-License-Identifier: BSD-3-Clause
"""fig06's per-cell cache must be keyed on the cell, and must actually hit.

The first version stored the wrong key. ``load_fit_results`` computed
``key = _cache_key(...)`` and two later loops ran ``for key in npz.files``, so
the write stored the name of the last array in the cell -- ``"energy"`` on the
real grid -- and every read missed. Nothing failed: each render just rebuilt
all 87 configuration models, and a two-minute plot took an hour, with the cache
directory full of files that looked correct.

These tests stub the expensive recompute and the adoption verdict, so they run
without an SSP grid, and they include an ``energy`` array so the shadowing bug
stores exactly that string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import fig06_code_overlay as fig06

pytestmark = pytest.mark.contract


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """One adopted cell, a private cache dir, and a counting stand-in compute."""
    results = tmp_path / "fits"
    results.mkdir()
    np.savez(
        results / "79_III.npz",
        **{"sfh_x": np.arange(4.0), "energy": np.zeros(4)},
    )
    (results / "79_III.json").write_text(json.dumps({"z": 1.0}))

    monkeypatch.setattr(fig06, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        fig06, "is_adopted", lambda meta, config: SimpleNamespace(adopted=True, reason="")
    )
    calls = []

    def compute(gal_id, config, params, z):
        calls.append(gal_id)
        return np.full(4, 10.5), np.full(4, 10.3), np.full(4, 1.2)

    monkeypatch.setattr(fig06, "_compute_all_derived_quantities", compute)
    return results, calls


def test_the_stored_key_is_the_content_hash_not_a_loop_variable(cell):
    results, _ = cell
    fig06.load_fit_results(79, "III", results, 4)

    stored = str(np.load(fig06.CACHE_DIR / "79_III.npz")["key"])
    expected = fig06._cache_key(results / "79_III.npz", results / "79_III.json", 4)
    assert stored == expected, f"cache stored key {stored!r}"


def test_a_second_load_is_served_from_the_cache(cell):
    results, calls = cell
    first = fig06.load_fit_results(79, "III", results, 4)
    second = fig06.load_fit_results(79, "III", results, 4)

    assert calls == [79], f"the recompute ran {len(calls)} times; the cache never hit"
    np.testing.assert_array_equal(second.mass_survived, first.mass_survived)
    np.testing.assert_array_equal(second.sfr_100myr, first.sfr_100myr)


def test_a_refit_cell_misses_rather_than_serving_stale_draws(cell):
    results, calls = cell
    fig06.load_fit_results(79, "III", results, 4)
    np.savez(
        results / "79_III.npz",
        **{"sfh_x": np.arange(4.0) + 1.0, "energy": np.zeros(4)},
    )
    fig06.load_fit_results(79, "III", results, 4)

    assert calls == [79, 79], "a changed cell was served from the cache"
