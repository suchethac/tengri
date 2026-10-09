"""A registered grid's ages must sit on its companion table's nodes.

A registered grid is built on its isochrone set's age nodes, so its ages are checked
against the table's, not interpolated across them: an age more than ``AGE_NODE_ATOL``
dex from a table node raises. Run with::

    .venv/bin/pytest tests/physics/test_mass_remaining_age_nodes.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.stellar.sps import mass_remaining_tables as mrt


def _mist_grid_ages():
    comp = mrt.load_companion_table("mass_remaining_mist_chabrier.h5", "mist", "chabrier")
    return comp, comp.log_age_yr - 9.0


@pytest.mark.contract
def test_on_node_ages_return_the_table_row_for_row():
    comp, lg_age_gyr = _mist_grid_ages()
    res = mrt.resolve_mass_remaining(
        stem="fsps_mist_miles_chabrier",
        detected_imf="chabrier",
        lg_age_gyr=lg_age_gyr,
        lgmet=comp.log_z_abs,
        embedded=None,
        has_alpha_axis=False,
        mode="table",
    )
    np.testing.assert_array_equal(res.table, comp.table)


@pytest.mark.contract
def test_age_shifted_by_more_than_the_tolerance_raises():
    comp, lg_age_gyr = _mist_grid_ages()
    shifted = lg_age_gyr + 0.02
    with pytest.raises(ValueError, match="dex from the table"):
        mrt.resolve_mass_remaining(
            stem="fsps_mist_miles_chabrier",
            detected_imf="chabrier",
            lg_age_gyr=shifted,
            lgmet=comp.log_z_abs,
            embedded=None,
            has_alpha_axis=False,
            mode="table",
        )


@pytest.mark.contract
def test_age_within_the_tolerance_is_accepted():
    comp, lg_age_gyr = _mist_grid_ages()
    nudged = lg_age_gyr + 0.5 * mrt.AGE_NODE_ATOL
    res = mrt.resolve_mass_remaining(
        stem="fsps_mist_miles_chabrier",
        detected_imf="chabrier",
        lg_age_gyr=nudged,
        lgmet=comp.log_z_abs,
        embedded=None,
        has_alpha_axis=False,
        mode="table",
    )
    np.testing.assert_array_equal(res.table, comp.table)
