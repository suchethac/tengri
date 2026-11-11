# SPDX-License-Identifier: BSD-3-Clause
"""Band-table refactors leave every JIT input and every output of six builds unchanged.

For each of six builds (``tests/contract/_band_table_cases.py``) the golden records the
structure of ``_template_data_for_jit()`` (every leaf's path, dtype, shape and the SHA-256 of its
raw bytes), the ``compile_signature()`` digest, and the ``predict_photometry`` output at a sampled
point. The structure is exact everywhere. On the platform that captured the golden (recorded as
``platform``, ``darwin-arm64``) the leaf digests, the signature digest and the photometry are
exact. On any other platform the stored leaf arrays and the photometry are compared at a
tolerance derived from their own magnitude, because XLA on x86-64 compiles the same program with
last-bit differences, and the signature is checked for self-consistency: two independent builds
on this platform give the same digest.

The goldens were captured before the seam existed. To recapture, run
``tests/contract/_band_table_capture.py:write_golden`` for each case on a tree known to be correct.
"""

from __future__ import annotations

import chex
import numpy as np
import pytest

from tests._jit_parity import _noise_floor
from tests.contract import _band_table_capture as capture_module, _band_table_cases as cases
from tests.contract._band_table_capture import capture, leaf_key, load_golden

pytestmark = pytest.mark.contract

#: Relative tolerance of the foreign-platform branch; the floor is derived from the values.
_RTOL = 1e-12


def _structure(entry: dict) -> dict:
    """A tree entry without its digest, so the structure is compared on its own."""
    return {key: value for key, value in entry.items() if key != "sha256"}


def _on_golden_platform(golden: dict) -> bool:
    """True when this process runs on the platform that captured ``golden``."""
    return golden["platform"] == capture_module.current_platform()


def _check_leaves(golden, golden_arrays, record, arrays) -> None:
    """Exact digests on the capturing platform; stored values within tolerance elsewhere."""
    if _on_golden_platform(golden):
        for got, want in zip(record["tree"], golden["tree"], strict=True):
            if want["kind"] == "array":
                assert got["sha256"] == want["sha256"], want["path"]
        return
    for index, want in enumerate(golden["tree"]):
        if want["kind"] != "array":
            continue
        stored = golden_arrays[leaf_key(index)]
        fresh = arrays[leaf_key(index)]
        if np.issubdtype(stored.dtype, np.floating):
            chex.assert_trees_all_close(fresh, stored, rtol=_RTOL, atol=_noise_floor(stored))
        else:
            assert np.array_equal(fresh, stored), want["path"]


def _check_photometry(golden, golden_arrays, arrays) -> None:
    """Exact on the capturing platform; within the derived tolerance elsewhere."""
    stored = golden_arrays["photometry"]
    if _on_golden_platform(golden):
        assert np.array_equal(arrays["photometry"], stored)
        return
    chex.assert_trees_all_close(
        arrays["photometry"], stored, rtol=_RTOL, atol=_noise_floor(stored)
    )


@pytest.fixture(scope="module", params=list(cases.CASES))
def parity(request):
    """The golden pair and the freshly captured pair for one case, with the case name."""
    name = request.param
    golden, golden_arrays = load_golden(name)
    assert "platform" in golden, "re-capture with the platform recorded"
    record, arrays = capture(cases.CASES[name]())
    return golden, golden_arrays, record, arrays, name


def test_template_tree_structure_is_unchanged(parity):
    """Every leaf of ``_template_data_for_jit()`` keeps its path, kind, dtype and shape."""
    golden, _, record, _, _ = parity
    assert [_structure(x) for x in record["tree"]] == [_structure(x) for x in golden["tree"]]


def test_template_leaves_are_bit_identical(parity):
    """Array leaves match the golden: digests on its platform, stored values within tolerance."""
    golden, golden_arrays, record, arrays, _ = parity
    _check_leaves(golden, golden_arrays, record, arrays)


def test_compile_signature_is_unchanged(parity):
    """``compile_signature()`` hashes to the golden digest, or is self-consistent off-platform."""
    golden, _, record, _, name = parity
    if _on_golden_platform(golden):
        assert record["signature_sha256"] == golden["signature_sha256"]
        return
    rebuilt = capture_module.signature_sha256(cases.CASES[name]())
    assert record["signature_sha256"] == rebuilt


def test_photometry_is_unchanged(parity):
    """``predict_photometry`` at the sampled point equals the golden array."""
    golden, golden_arrays, _, arrays, _ = parity
    _check_photometry(golden, golden_arrays, arrays)


def test_sampled_point_is_unchanged(parity):
    """The sampled parameters that feed the photometry match the golden."""
    golden, _, record, _, _ = parity
    assert record["params"] == golden["params"]


def test_engagement_summary_is_unchanged(parity):
    """The engagement summary captured with the golden is reproduced."""
    golden, _, record, _, _ = parity
    assert record["summary"] == golden["summary"]


def test_foreign_platform_branch_accepts_last_bit_drift_and_rejects_a_perturbed_leaf(
    monkeypatch,
):
    """Forces the foreign-platform branch on this machine and checks both directions.

    The stored arm64 values must pass the tolerance check when the platform is taken to be
    x86-64, and a leaf perturbed by one part in 1e6 in a copy must fail it. The platform is
    overridden through ``monkeypatch`` on the capture module, and restored afterwards.
    """
    monkeypatch.setattr(capture_module, "current_platform", lambda: "linux-x86_64")
    name = "dust_fixed_z"
    golden, golden_arrays = load_golden(name)
    record, arrays = capture(cases.CASES[name]())
    assert not _on_golden_platform(golden)
    _check_photometry(golden, golden_arrays, arrays)
    _check_leaves(golden, golden_arrays, record, arrays)

    index = next(
        i for i, x in enumerate(golden["tree"]) if x["kind"] == "array" and "float" in x["dtype"]
    )
    perturbed = dict(arrays)
    leaf = np.array(perturbed[leaf_key(index)], dtype=float)
    peak = np.argmax(np.abs(leaf))
    leaf.flat[peak] *= 1.0 + 1e-6
    perturbed[leaf_key(index)] = leaf
    with pytest.raises(AssertionError):
        _check_leaves(golden, golden_arrays, record, perturbed)


def test_golden_stores_every_array_leaf():
    """Each golden carries its platform and every array leaf for off-platform checks."""
    for name in cases.CASES:
        golden, golden_arrays = load_golden(name)
        assert isinstance(golden["platform"], str), name
        for index, entry in enumerate(golden["tree"]):
            if entry["kind"] == "array":
                assert leaf_key(index) in golden_arrays, (name, entry["path"])
