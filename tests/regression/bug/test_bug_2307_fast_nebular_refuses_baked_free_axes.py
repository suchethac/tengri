# SPDX-License-Identifier: BSD-3-Clause
"""Refuse freed CB19 optional axes when enabling fast-nebular (#2307)."""
from __future__ import annotations

import pytest

from tengri import FREE, Fixed, SEDModel
from tests._cb19_grid import write_synthetic_cb19_grid

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("axis_name", ["neb_hbfrac", "neb_log_nH", "neb_co", "neb_dno"])
def test_enable_fast_nebular_refuses_freed_optional_axes(tmp_path, synthetic_ssp_wide, axis_name):
    """enable_fast_nebular must refuse a freed optional CB19 axis baked into the grid."""
    grid_path = write_synthetic_cb19_grid(tmp_path / "varied_cb19.h5")

    neb_config = {
        "type": "cb19",
        "grid": str(grid_path),
        axis_name: FREE,
    }

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=None,
        neb=neb_config,
        redshift=Fixed(0.0),
    )

    with pytest.raises(
        ValueError,
        match=f".*{axis_name}.*reference value.*",
    ):
        model.enable_fast_nebular([4861.0])


def test_enable_fast_nebular_allows_pinned_optional_axes(tmp_path, synthetic_ssp_wide):
    """enable_fast_nebular must allow optional axes when pinned (not freed)."""
    grid_path = write_synthetic_cb19_grid(tmp_path / "varied_cb19.h5")

    neb_config = {
        "type": "cb19",
        "grid": str(grid_path),
        "neb_hbfrac": Fixed(0.5),
        "neb_log_nH": Fixed(0.0),
    }

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=None,
        neb=neb_config,
        redshift=Fixed(0.0),
    )

    try:
        model.enable_fast_nebular([4861.0])
    except ValueError as e:
        if "reference value" in str(e) and "neb_" in str(e):
            pytest.fail(f"Raised refusal for pinned axis: {e}")
        raise


def test_enable_fast_nebular_allows_freed_table_axes(tmp_path, synthetic_ssp_wide):
    """enable_fast_nebular allows freed axes that ARE in the table grid."""
    grid_path = write_synthetic_cb19_grid(tmp_path / "varied_cb19.h5")

    neb_config = {
        "type": "cb19",
        "grid": str(grid_path),
        "neb_logU": FREE,
    }

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=None,
        neb=neb_config,
        redshift=Fixed(0.0),
    )

    try:
        model.enable_fast_nebular([4861.0])
    except ValueError as e:
        if "reference value" in str(e):
            pytest.fail(f"Incorrectly refused neb_logU: {e}")
        raise
