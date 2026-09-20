#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The configuration-provenance audit must catch a redefined configuration.

Five of the six configurations in `results/fits_superseded_oldsuite_20260920`
sample a different SFH family from the one `configs.CONFIGS` declares, and
nothing reported it: the cells keep their original file names, so every reader
that trusts `13097_IV.npz` to be Configuration IV was mislabeling it. These
tests pin the two things that make the audit able to see that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _cell_provenance import (
    SFH_PREFIX_BY_TYPE,
    Mismatch,
    audit,
    banner,
    sampled_sfh_prefix,
)

pytestmark = pytest.mark.contract


def _write_cell(directory: Path, gal_id: int, config: str, param_names: list[str]) -> None:
    payload = {name: np.zeros(4) for name in param_names}
    payload["sfh_lookback_time_yr"] = np.linspace(0.0, 1e10, 8)
    payload["sfh_sfr_median"] = np.ones(8)
    np.savez(directory / f"{gal_id}_{config}.npz", **payload)


def test_prefix_is_the_common_prefix_not_the_second_token():
    """`sfh_declining_exp_*` is the family `declining_exp`, not `declining`.

    Splitting on underscores and taking token 1 gets every single-token family
    right and this one wrong, so it passes a careless test suite.
    """
    keys = [
        "sfh_declining_exp_log_total_mass",
        "sfh_declining_exp_tau_gyr",
        "sfh_declining_exp_age_gyr",
        "sfh_lookback_time_yr",
        "sfh_sfr_median",
    ]
    assert sampled_sfh_prefix(keys) == "sfh_declining_exp_"


def test_prefix_ignores_reconstructed_history_arrays():
    """The history arrays are named `sfh_*` but are not sampled parameters."""
    only_history = ["sfh_lookback_time_yr", "sfh_sfr_median", "sfh_sfr_p16", "sfh_sfr_p84"]
    assert sampled_sfh_prefix(only_history) is None

    with_params = [*only_history, "sfh_dir_log_total_mass", "sfh_dir_z_0"]
    assert sampled_sfh_prefix(with_params) == "sfh_dir_"


def test_prefix_of_a_single_parameter_family():
    assert sampled_sfh_prefix(["sfh_const_log_total_mass"]) == "sfh_const_"


def test_audit_flags_a_redefined_configuration(tmp_path):
    """A cell sampling `sfh_const_` under a configuration declaring dirichlet."""
    configs = {"IV": {"sfh_type": "dirichlet"}}
    _write_cell(tmp_path, 13097, "IV", ["sfh_const_log_total_mass", "sfh_const_start_gyr"])

    mismatches, notes = audit(tmp_path, configs)

    assert notes == []
    assert len(mismatches) == 1
    found = mismatches[0]
    assert found == Mismatch(
        config="IV",
        declared_type="dirichlet",
        declared_prefix="sfh_dir_",
        found_prefixes=("sfh_const_",),
        n_cells=1,
    )
    assert "sfh_const_" in found.describe()
    assert banner(tmp_path, mismatches, notes) is not None


def test_audit_passes_a_matching_directory(tmp_path):
    configs = {"II": {"sfh_type": "dpl"}}
    _write_cell(tmp_path, 267, "II", ["sfh_dpl_alpha", "sfh_dpl_beta"])

    mismatches, notes = audit(tmp_path, configs)

    assert mismatches == []
    assert notes == []
    assert banner(tmp_path, mismatches, notes) is None


def test_audit_reports_a_mixed_configuration(tmp_path):
    """Half a grid re-run after a configuration edit is the dangerous case.

    Both families are named, because reporting only the unexpected one would
    hide that the directory holds two models under one label.
    """
    configs = {"I": {"sfh_type": "continuity"}}
    _write_cell(tmp_path, 1, "I", ["sfh_cont_log_total_mass", "sfh_cont_ratio_0"])
    _write_cell(tmp_path, 2, "I", ["sfh_delayed_log_total_mass", "sfh_delayed_tau_gyr"])

    mismatches, _ = audit(tmp_path, configs)

    assert len(mismatches) == 1
    assert mismatches[0].found_prefixes == ("sfh_cont_", "sfh_delayed_")
    assert mismatches[0].n_cells == 2


def test_audit_notes_an_undeclared_configuration(tmp_path):
    configs = {"VII": {}}
    mismatches, notes = audit(tmp_path, configs)
    assert mismatches == []
    assert any("no sfh_type declared" in note for note in notes)


def test_audit_notes_a_cell_with_no_sampled_sfh(tmp_path):
    configs = {"II": {"sfh_type": "dpl"}}
    np.savez(tmp_path / "267_II.npz", stellar_mass=np.ones(4))
    mismatches, notes = audit(tmp_path, configs)
    assert mismatches == []
    assert any("no sampled SFH parameter" in note for note in notes)


def test_prefix_table_matches_the_live_sfh_registry():
    """The static table exists so NPZ-only consumers need not import tengri.

    That is only safe while it agrees with the registry, which this checks by
    reading each model's first declared parameter name.
    """
    tengri = pytest.importorskip("tengri")
    rows = {row["name"]: row for row in tengri.list_sfh_models()}
    assert rows, "the SFH registry is empty"

    checked = 0
    for sfh_type, prefix in SFH_PREFIX_BY_TYPE.items():
        assert sfh_type in rows, f"{sfh_type} is not a registered SFH model"
        details = rows[sfh_type]["param_details"]
        assert details, f"{sfh_type} declares no parameters"
        for detail in details:
            assert detail["name"].startswith(prefix), (
                f"{sfh_type}: declared prefix {prefix!r} does not match "
                f"registered parameter {detail['name']!r}"
            )
        checked += 1
    assert checked == len(SFH_PREFIX_BY_TYPE)
