# SPDX-License-Identifier: BSD-3-Clause
"""The not-for-the-paper stamp must follow the data, not the command line.

``fill_demonstration_numbers.py`` once decided whether to brand its output
"NOT FOR THE PAPER" from ``args.from_dir is None`` -- that is, from whether a
flag had been passed rather than from what the directory held. Naming the
authoritative fits explicitly was enough to brand them, and a superseded
directory promoted to the default would have gone unbranded. A guard that cries
wolf on good data is one people learn to ignore, which is worse than no guard.

These tests pin the three states to the path itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.paper1.fill_demonstration_numbers import CANONICAL, classify_directory


def test_canonical_directory_is_canonical_by_default():
    assert classify_directory(CANONICAL) == "canonical"


def test_canonical_directory_named_explicitly_is_still_canonical():
    """The regression: naming the paper's own fits must not brand them."""
    assert classify_directory(Path(str(CANONICAL))) == "canonical"
    assert classify_directory(CANONICAL.resolve()) == "canonical"


@pytest.mark.parametrize(
    "name",
    ["fits_superseded_oldsuite_20260920", "fits_superseded_anything_else"],
)
def test_quarantined_directories_are_superseded(name):
    assert classify_directory(CANONICAL.parent / name) == "superseded"


def test_any_other_directory_is_other_not_superseded():
    """Readable and named, but never branded unusable on a guess."""
    assert classify_directory(CANONICAL.parent / "fits_profiled_2026-09-14") == "other"
    assert classify_directory(CANONICAL.parent / "backend_sweep") == "other"
