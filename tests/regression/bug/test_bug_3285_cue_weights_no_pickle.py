# SPDX-License-Identifier: BSD-3-Clause
"""The Cue weights load without pickle (#3285, item 3).

``load_cue_weights`` called ``np.load(..., allow_pickle=True)`` on a weights archive
that holds only numeric and fixed-width string arrays. Pickle is an arbitrary-code
path, and the flag would also let a malformed archive load silently. The loader now
refuses object arrays, and the shipped weights still load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tengri.components.nebular.cue import load_cue_weights

pytestmark = pytest.mark.regression_bug

_SHIPPED = Path(__file__).resolve().parents[3] / "data" / "cue_weights.npz"


@pytest.mark.skipif(not _SHIPPED.is_file(), reason="data/cue_weights.npz not present")
def test_object_array_archive_is_refused(tmp_path):
    """The shipped weights plus one pickled object array raise instead of loading.

    The archive is otherwise complete, so the only defect it carries is the object array.
    """
    with np.load(_SHIPPED, allow_pickle=False) as shipped:
        arrays = {name: shipped[name] for name in shipped.files}
    arrays["pickled_payload"] = np.array([{"not": "weights"}], dtype=object)
    path = tmp_path / "pickled.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_cue_weights(str(path))


@pytest.mark.skipif(not _SHIPPED.is_file(), reason="data/cue_weights.npz not present")
def test_shipped_weights_load_without_pickle():
    """The shipped Cue weights load through the loader and hold no object arrays."""
    weights = load_cue_weights(str(_SHIPPED))
    assert weights is not None
    with np.load(_SHIPPED, allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)
