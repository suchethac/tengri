# SPDX-License-Identifier: BSD-3-Clause
"""#2328: Unknown key validation must precede grid-file existence check.

Issue #2328: When building with `neb={'type': 'cloudy'}` and no 'grid' key on a
machine without a cloudy grid file on disk, `parse_groups` raised
"The CLOUDY nebular backend needs a grid file..." before validating that the
supplied keys were correct. A typo in the neb group (e.g., 'totally_bogus_key')
was silently misreported as "missing grid", hiding the actual typo.

The fix orders validation so unknown-key errors are reported first, giving the
user the chance to correct them before encountering environment-dependent errors.

Test coverage:

1. With no grid file visible and neb={'type': 'cloudy', 'totally_bogus_key': ...},
   the bogus key is reported (did-you-mean message), not the grid error.
2. With no grid file and a valid neb group, the grid error is still raised.
3. With a grid file available and the bogus key, the key error is reported
   (control arm: the fix must not change behavior when the grid is present).
"""

from __future__ import annotations

import pytest

import tengri
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.regression_bug


def test_bogus_key_reported_without_grid_on_disk(monkeypatch, tmp_path):
    """A typo in neb raises unknown-key error, not 'missing grid' error."""
    # Monkeypatch the data locator so only the empty tmp_path is visible
    monkeypatch.setattr(
        "tengri._data_setup.data_dirs",
        lambda: [tmp_path],
    )

    with pytest.raises(
        ValueError,
        match=(r"Unknown key 'totally_bogus_key' in group 'neb'"),
    ):
        parse_groups(
            neb={
                "type": "cloudy",
                "totally_bogus_key": tengri.Fixed(0.5),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            redshift=tengri.Fixed(0.1),
        )


def test_valid_cloudy_group_still_raises_grid_message(monkeypatch, tmp_path):
    """Valid cloudy group with no grid still raises the grid error."""
    # Monkeypatch the data locator so only the empty tmp_path is visible
    monkeypatch.setattr(
        "tengri._data_setup.data_dirs",
        lambda: [tmp_path],
    )

    with pytest.raises(
        ValueError,
        match=(r"The CLOUDY nebular backend needs a grid file"),
    ):
        parse_groups(
            neb={
                "type": "cloudy",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            redshift=tengri.Fixed(0.1),
        )


def test_grid_present_unchanged(monkeypatch):
    """With grid present, bogus key is reported (control arm)."""
    # Monkeypatch _default_cloudy_grid to return a valid path
    monkeypatch.setattr(
        Parameters,
        "_default_cloudy_grid",
        staticmethod(lambda: "data/cloudy_grid_mist.h5"),
    )

    with pytest.raises(
        ValueError,
        match=(r"Unknown key 'totally_bogus_key' in group 'neb'"),
    ):
        parse_groups(
            neb={
                "type": "cloudy",
                "totally_bogus_key": tengri.Fixed(0.5),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            redshift=tengri.Fixed(0.1),
        )
