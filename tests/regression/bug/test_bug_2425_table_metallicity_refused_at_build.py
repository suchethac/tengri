# SPDX-License-Identifier: BSD-3-Clause
"""#2425: ``metallicity_model='table'`` with no Z(t) source must refuse at build.

A table metallicity reads its history from the build-time table on the config
or, at runtime, from ``params['met_history']``. That runtime channel exists
only with ``sfh_model='table'`` (#996). A config with neither can never predict,
so construction is where it must fail, not the first ``predict_photometry``.
"""

from __future__ import annotations

import pytest

from tengri.components.stellar.component import StellarSEDComponentConfig

pytestmark = pytest.mark.regression_bug

_TABLE_AGE = [6.0, 10.14]
_TABLE_Z = [-2.0, -2.0]


def test_table_without_any_source_refused_at_build():
    """sfh=dpl, met=table, no table pinned and no history channel: refused at build."""
    with pytest.raises(ValueError, match="met_table_log_age_yr"):
        StellarSEDComponentConfig(sfh_model="dpl", metallicity_model="table")


def test_message_names_the_sfh_requirement_for_met_history():
    with pytest.raises(ValueError, match="sfh_model='table'"):
        StellarSEDComponentConfig(sfh_model="continuity", metallicity_model="table")


def test_table_pinned_on_config_builds():
    cfg = StellarSEDComponentConfig(
        sfh_model="dpl",
        metallicity_model="table",
        met_table_log_age_yr=_TABLE_AGE,
        met_table_log_z_abs=_TABLE_Z,
    )
    assert cfg.metallicity_model == "table"


def test_runtime_history_channel_with_table_sfh_builds():
    """sfh=table is the promise that params['met_history'] arrives at runtime."""
    cfg = StellarSEDComponentConfig(sfh_model="table", metallicity_model="table")
    assert cfg.sfh_model == "table"


def test_delta_metallicity_needs_no_table():
    cfg = StellarSEDComponentConfig(sfh_model="dpl", metallicity_model="delta")
    assert cfg.metallicity_model == "delta"
