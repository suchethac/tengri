# SPDX-License-Identifier: BSD-3-Clause
"""The IGM sub-band cache fails loud on a bad entry, and misses only on a miss (#3285).

``_subband_cache.load`` used to return ``None`` on any exception. A truncated or
corrupt entry was therefore recomputed without a word, and a readable entry of
the wrong shape was recomputed by each caller's own silent ``shape`` check.
Both now raise: ``None`` means the entry is absent (or the cache is disabled),
nothing else. ``load`` takes ``expected_shape``, the shape the caller derived
from the redshift grid and node layout, and a mismatch raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.igm import _subband_cache as sc

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    return tmp_path


@pytest.fixture
def key():
    return sc.cache_key(np.linspace(1000.0, 2000.0, 5), np.array([0.5, 1.5, 2.5]), "inoue")


def test_absent_entry_is_a_miss(cache_on, key):
    assert sc.load(key) is None


def test_corrupt_entry_raises_instead_of_recomputing_silently(cache_on, key):
    path = cache_on / f"igm_subband_{key}.npz"
    path.write_bytes(b"not an npz file")
    with pytest.raises(RuntimeError, match=str(path.name)):
        sc.load(key)


def test_truncated_entry_raises(cache_on, key):
    sc.store(key, np.zeros((3, 4, 5)))
    path = cache_on / f"igm_subband_{key}.npz"
    path.write_bytes(path.read_bytes()[:40])
    with pytest.raises(RuntimeError):
        sc.load(key)


def test_wrong_shape_entry_raises_when_expected_shape_given(cache_on, key):
    sc.store(key, np.zeros((3, 4, 6)))
    with pytest.raises(ValueError, match=r"\(3, 4, 6\).*\(3, 4, 5\)"):
        sc.load(key, expected_shape=(3, 4, 5))


def test_matching_shape_round_trips(cache_on, key):
    table = np.linspace(0.0, 1.0, 60).reshape(3, 4, 5)
    sc.store(key, table)
    np.testing.assert_array_equal(sc.load(key, expected_shape=(3, 4, 5)), table)


def test_disabled_cache_is_a_miss_even_with_a_corrupt_file(cache_on, key, monkeypatch):
    (cache_on / f"igm_subband_{key}.npz").write_bytes(b"junk")
    monkeypatch.setenv("TENGRI_DISABLE_PRECOMP_CACHE", "1")
    assert sc.load(key) is None
