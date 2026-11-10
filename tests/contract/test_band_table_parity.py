# SPDX-License-Identifier: BSD-3-Clause
"""Band-table refactors leave every JIT input and every output of six builds unchanged.

For each of six builds (``tests/contract/_band_table_cases.py``) the golden records the
structure of ``_template_data_for_jit()`` (every leaf's path, dtype, shape and the SHA-256 of its
raw bytes, which is an exact equality), the ``compile_signature()`` digest, and the
``predict_photometry`` output at a sampled point (compared with ``np.array_equal``). A step of the
band-table move may not change any of them: the keys, dtypes and values are JIT-structural.

The goldens were captured before the seam existed. To recapture, run
``tests/contract/_band_table_capture.py:write_golden`` for each case on a tree known to be correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.contract import _band_table_cases as cases
from tests.contract._band_table_capture import capture, load_golden

pytestmark = pytest.mark.contract


def _structure(entry: dict) -> dict:
    """A tree entry without its digest, so the structure is compared on its own."""
    return {key: value for key, value in entry.items() if key != "sha256"}


@pytest.fixture(scope="module", params=list(cases.CASES))
def parity(request):
    """The golden pair and the freshly captured pair for one case."""
    name = request.param
    golden, golden_arrays = load_golden(name)
    record, arrays = capture(cases.CASES[name]())
    return golden, golden_arrays, record, arrays


def test_template_tree_structure_is_unchanged(parity):
    """Every leaf of ``_template_data_for_jit()`` keeps its path, kind, dtype and shape."""
    golden, _, record, _ = parity
    assert [_structure(x) for x in record["tree"]] == [_structure(x) for x in golden["tree"]]


def test_template_leaves_are_bit_identical(parity):
    """The raw bytes of every template array leaf match the golden digest."""
    golden, _, record, _ = parity
    for got, want in zip(record["tree"], golden["tree"], strict=True):
        if want["kind"] == "array":
            assert got["sha256"] == want["sha256"], want["path"]


def test_compile_signature_is_unchanged(parity):
    """``compile_signature()`` hashes to the golden digest."""
    golden, _, record, _ = parity
    assert record["signature_sha256"] == golden["signature_sha256"]


def test_photometry_is_unchanged(parity):
    """``predict_photometry`` at the sampled point equals the golden array exactly."""
    _, golden_arrays, _, arrays = parity
    assert np.array_equal(arrays["photometry"], golden_arrays["photometry"])


def test_sampled_point_is_unchanged(parity):
    """The sampled parameters that feed the photometry match the golden."""
    golden, _, record, _ = parity
    assert record["params"] == golden["params"]


def test_engagement_summary_is_unchanged(parity):
    """The engagement summary captured with the golden is reproduced."""
    golden, _, record, _ = parity
    assert record["summary"] == golden["summary"]
