# SPDX-License-Identifier: BSD-3-Clause
"""The parity sweep's reference can go stale, and the sweep must say so (#2300).

The float64 reference file is correct for the tree it was written on. Every physics
merge after that moves some seam, and a float32 comparison against the old file then
reports a FAIL that has nothing to do with float32 or the device: #2260 moved the
panchromatic seam by 3.3e-3 and its gradient by 67%, and the "MPS residual" hunt
that followed cost two hours of GPU bisection for a ten-second float64 check.

Two helpers, both pure so they are testable without git or JAX:

* ``reference_staleness`` turns (reference tree SHA, number of src commits since it)
  into a warning banner, or ``None`` when there is nothing to warn about.
* ``self_check_drift`` compares float64 rows recomputed now against the file's rows,
  per seam, and is what ``--self-check`` gates on.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[2] / "bench" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_float32_mps_parity import (
    reference_staleness,
    self_check_drift,
)


def test_a_reference_at_the_current_tree_raises_no_banner():
    assert reference_staleness("abc1234", 0) is None


def test_src_commits_since_the_reference_produce_a_banner_naming_the_count_and_the_remedy():
    banner = reference_staleness("abc1234", 7)
    assert banner is not None
    assert "7" in banner and "abc1234" in banner
    assert "--self-check" in banner, "the banner must say how to measure the staleness"


def test_a_reference_without_a_tree_sha_is_flagged_not_trusted():
    banner = reference_staleness(None, None)
    assert banner is not None and "--self-check" in banner


def test_an_undeterminable_history_is_flagged_not_trusted():
    """Shallow clones cannot count commits; that is not the same as zero."""
    banner = reference_staleness("abc1234", None)
    assert banner is not None and "--self-check" in banner


def _rows(scale_pan=1.0, scale_grad=1.0):
    return {
        "stellar_dust": {
            "seam": "stellar_dust",
            "built": True,
            "photometry": [1.0, 2.0, 3.0],
            "grad": [0.5, -0.25],
        },
        "panchromatic": {
            "seam": "panchromatic",
            "built": True,
            "photometry": [1.0, 2.0, 3.0 * scale_pan],
            "grad": [0.5 * scale_grad, -0.25],
        },
    }


def test_identical_rows_have_zero_drift_on_every_seam():
    drift = self_check_drift(_rows(), _rows())
    assert drift == {"stellar_dust": 0.0, "panchromatic": 0.0}


def test_drift_is_the_max_relative_deviation_over_photometry_and_gradient_per_seam():
    drift = self_check_drift(_rows(), _rows(scale_pan=1.0 + 3.3e-3, scale_grad=1.67))
    assert drift["stellar_dust"] == 0.0
    assert drift["panchromatic"] == pytest.approx(0.67, rel=1e-12)


def test_a_seam_the_file_has_but_this_tree_did_not_rebuild_is_reported_not_skipped():
    here = {"stellar_dust": _rows()["stellar_dust"]}
    drift = self_check_drift(_rows(), here)
    assert np.isnan(drift["panchromatic"]), "an unmeasured seam must not read as clean"
    assert drift["stellar_dust"] == 0.0
