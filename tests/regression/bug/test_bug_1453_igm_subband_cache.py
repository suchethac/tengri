# SPDX-License-Identifier: BSD-3-Clause
"""The IGM build-time constant is cached, and its key is complete (#1453).

``subband_node_transmission`` and ``precompute_band_factors`` each loop
``igm_absorption`` over the redshift grid. Both are build-time constants, so
neither the JAX compilation cache nor the photometry z-table cache covered
them, and every ``SEDModel.build`` re-paid them: ~9 s on a free-redshift model
against ~0.3 s with ``igm='none'``, identical across three consecutive builds
in one process.

The bulk of this file is about the **cache key**, not the speedup. A
cross-process cache whose key omits an input returns wrong physics silently and
persistently — that is how #1122 happened — so every input that changes the
stored table must change the hash, and the tests below vary them one at a time.
A speed test would pass on a key that ignores everything.
"""

import numpy as np
import pytest

from tengri.components.igm import _subband_cache as sc

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def base():
    """A representative (waves, z-grid) pair — 5-D node table, as shipped."""
    rng = np.random.default_rng(0)
    return {
        "waves_rest": rng.uniform(1000.0, 9000.0, size=(3, 2, 2, 4, 5)),
        "z_grid": np.array([0.5, 1.5, 2.5]),
        "igm_model": "inoue",
    }


# ── key completeness: vary one input, demand a different hash ───────────────


def test_wavelengths_change_the_key(base):
    other = dict(base, waves_rest=base["waves_rest"] + 1.0)
    assert sc.cache_key(**base) != sc.cache_key(**other)


def test_redshift_grid_changes_the_key(base):
    other = dict(base, z_grid=base["z_grid"] + 0.01)
    assert sc.cache_key(**base) != sc.cache_key(**other)


def test_igm_model_changes_the_key(base):
    """The one most likely to be forgotten — it is a string, not an array."""
    for model in ("madau", "meiksin06", "asada25"):
        assert sc.cache_key(**dict(base, igm_model=model)) != sc.cache_key(**base), model


def test_every_registered_igm_model_hashes_distinctly(base):
    """No two shipped models may share a key.

    Derived from the registry rather than a hand-written list, so a model added
    later is covered without editing this file.
    """
    import tengri

    names = [row["name"] for row in tengri.list_igm_models()]
    assert len(names) >= 3, f"only {len(names)} IGM models discovered — near-vacuous"
    keys = {n: sc.cache_key(**dict(base, igm_model=n)) for n in names}
    assert len(set(keys.values())) == len(names), f"colliding keys: {keys}"


def test_shape_is_part_of_the_key(base):
    """Identical bytes in a different layout index different nodes."""
    reshaped = base["waves_rest"].reshape(3, 2, 4, 2, 5)
    assert sc.cache_key(**dict(base, waves_rest=reshaped)) != sc.cache_key(**base)


def test_the_patchy_and_dla_gates_are_in_the_key(base):
    """Unreachable today; keyed so a loosened gate cannot read a stale entry."""
    assert sc.cache_key(**base, igm_patchy=True) != sc.cache_key(**base)
    assert sc.cache_key(**base, use_dla=True) != sc.cache_key(**base)


def test_identical_inputs_give_the_same_key(base):
    """The point of a cache — over-keying would make it never hit."""
    assert sc.cache_key(**base) == sc.cache_key(**base)


# ── band-factor key: strictly more inputs ──────────────────────────────────


@pytest.fixture
def bf():
    rng = np.random.default_rng(1)
    return {
        "wave_rest": np.linspace(1000.0, 9000.0, 64),
        "filter_waves": [np.linspace(3000.0, 4000.0, 16), np.linspace(5000.0, 6000.0, 16)],
        "filter_trans": [rng.uniform(0, 1, 16), rng.uniform(0, 1, 16)],
        "z_grid": np.array([0.5, 1.5]),
        "igm_model": "inoue",
        "convention": "bessell",
    }


@pytest.mark.parametrize("field", ["wave_rest", "z_grid", "igm_model", "convention"])
def test_band_factor_scalar_and_array_inputs_are_keyed(bf, field):
    mutated = dict(bf)
    if field == "igm_model":
        mutated[field] = "madau"
    elif field == "convention":
        mutated[field] = "photon"
    else:
        mutated[field] = np.asarray(bf[field]) + 1.0
    assert sc.band_factor_key(**mutated) != sc.band_factor_key(**bf), field


def test_band_factor_key_covers_the_filter_curves(bf):
    """The input the node-table key does not have, and the easiest to omit."""
    moved = dict(bf, filter_waves=[bf["filter_waves"][0] + 5.0, bf["filter_waves"][1]])
    assert sc.band_factor_key(**moved) != sc.band_factor_key(**bf)

    retrans = dict(bf, filter_trans=[bf["filter_trans"][0] * 0.5, bf["filter_trans"][1]])
    assert sc.band_factor_key(**retrans) != sc.band_factor_key(**bf)


def test_band_factor_key_is_order_sensitive(bf):
    """The table is indexed by filter position, so a reordering is a new table."""
    swapped = dict(
        bf,
        filter_waves=list(reversed(bf["filter_waves"])),
        filter_trans=list(reversed(bf["filter_trans"])),
    )
    assert sc.band_factor_key(**swapped) != sc.band_factor_key(**bf)


def test_the_two_key_functions_do_not_collide(base, bf):
    """A node table must never be served for a band-factor lookup."""
    assert sc.cache_key(**base) != sc.band_factor_key(**bf)


# ── round-trip and fail-soft ───────────────────────────────────────────────


def test_disk_round_trip_preserves_values(tmp_path, monkeypatch, base):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    key = sc.cache_key(**base)
    table = np.linspace(0.0, 1.0, 60).reshape(3, 4, 5)
    sc.store(key, table)
    loaded = sc.load(key)
    assert loaded is not None, "stored table did not come back — the write silently failed"
    np.testing.assert_array_equal(loaded, table)


def test_the_store_actually_writes_a_readable_file(tmp_path, monkeypatch, base):
    """Guards the bug found while building this: the temp name must end in
    ``.npz``, or ``savez_compressed`` appends the extension, the rename finds
    nothing, and the cache silently never hits."""
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    sc.store(sc.cache_key(**base), np.zeros((3, 4, 5)))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files, "nothing written"
    assert all(f.endswith(".npz") for f in files), f"stray temp files left behind: {files}"
    assert not any(".tmp" in f for f in files), f"temp file was never renamed: {files}"


def test_disabling_the_cache_is_honored(tmp_path, monkeypatch, base):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TENGRI_DISABLE_PRECOMP_CACHE", "1")
    assert sc.cache_dir() is None
    sc.store(sc.cache_key(**base), np.zeros((3, 4, 5)))
    assert not list(tmp_path.iterdir()), "wrote despite the cache being disabled"
    assert sc.load(sc.cache_key(**base)) is None


def test_a_corrupt_entry_degrades_to_a_miss(tmp_path, monkeypatch, base):
    """A cache is an optimization; a bad file must not take the build down."""
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    key = sc.cache_key(**base)
    (tmp_path / f"igm_subband_{key}.npz").write_bytes(b"not an npz file")
    assert sc.load(key) is None


def test_disabling_the_cache_also_silences_the_memo(monkeypatch, base):
    """The kill-switch must be total, not just the disk half.

    ``tests/conftest.py`` disables the precomp cache for hermeticity. A memo
    that kept serving would quietly share a table between two tests that each
    believe they built from scratch — a partial switch that reads as total.
    """
    sc.clear_memo()
    key = sc.cache_key(**base)
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    sc.memo_put(key, np.ones((2, 2)))
    assert sc.memo_get(key) is not None, "memo should work when caching is enabled"

    monkeypatch.setenv("TENGRI_DISABLE_PRECOMP_CACHE", "1")
    assert sc.memo_get(key) is None, "memo served a hit while caching was disabled"
    sc.memo_put(sc.cache_key(**dict(base, igm_model="madau")), np.zeros((2, 2)))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    assert sc.memo_get(sc.cache_key(**dict(base, igm_model="madau"))) is None, (
        "memo stored an entry while caching was disabled"
    )
    sc.clear_memo()


def test_the_memo_is_clearable(monkeypatch, base):
    # Caching must be enabled explicitly: ``tests/conftest.py`` sets
    # TENGRI_DISABLE_PRECOMP_CACHE suite-wide for hermeticity, and the memo
    # honors it, so without this the puts below are correctly no-ops.
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    key = sc.cache_key(**base)
    sc.memo_put(key, np.ones((2, 2)))
    assert sc.memo_get(key) is not None
    sc.clear_memo()
    assert sc.memo_get(key) is None
