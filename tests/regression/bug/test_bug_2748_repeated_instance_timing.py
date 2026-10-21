# SPDX-License-Identifier: BSD-3-Clause
"""A repeated SFH instance settles its Gyr alias against its own fraction (#2748).

``sfh={'type': ['norm', 'norm']}`` numbers the second instance
(``sfh_norm_2_*``). Its ``*_gyr`` timing is the Fixed shim of its own
``sfh_norm_2_*_frac`` twin, exactly as the first instance's is of
``sfh_norm_*_frac``, so a Fixed Gyr value must reach the kernel as Gyr and the
fraction must be dropped. Before the fix only the first instance was paired:
the second kept both names and the kernel received the fraction as a keyword.
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.config.exceptions import ConfigError
from tengri.parameters.timing import timing_frac_params, timing_names_to_exclude

pytestmark = pytest.mark.regression_bug

_SFH = {
    "type": ["norm", "norm"],
    "all_params": Fixed(DEFAULT),
    "sfh_norm_log_total_mass": Fixed(10.0),
    "sfh_norm_peak_lbt_gyr": Fixed(1.0),
    "sfh_norm_width_gyr": Fixed(0.2),
    "sfh_norm_2_log_total_mass": Fixed(9.7),
    "sfh_norm_2_peak_lbt_gyr": Fixed(5.0),
    "sfh_norm_2_width_gyr": Fixed(0.8),
}


def test_second_instance_gyr_alias_pairs_with_its_own_fraction(ssp_data_fsps):
    """The Fixed Gyr alias of instance 2 builds and predicts; its fraction is dropped."""
    model = SEDModel.build(ssp_data=ssp_data_fsps, sfh=_SFH, redshift=Fixed(0.05))
    assert model.predict_state({}) is not None
    names = set(model.spec._valid_param_names)
    assert "sfh_norm_2_peak_lbt_gyr" in names
    assert "sfh_norm_2_peak_lbt_frac" not in names
    assert "sfh_norm_peak_lbt_frac" not in names


def test_numbered_pair_is_found_from_the_names_present():
    """The pairing table gains the numbered twin of a base pair, and only on request."""
    assert "sfh_norm_2_peak_lbt_frac" not in timing_frac_params()
    names = {"sfh_norm_2_peak_lbt_frac", "sfh_norm_2_peak_lbt_gyr"}
    assert timing_frac_params(names)["sfh_norm_2_peak_lbt_frac"] == "sfh_norm_2_peak_lbt_gyr"


def test_naming_both_names_of_a_repeated_pair_is_refused():
    """Both twins of the second instance named is the same ConfigError as for the first."""
    names = {"sfh_norm_2_peak_lbt_frac", "sfh_norm_2_peak_lbt_gyr"}
    with pytest.raises(ConfigError, match="not both"):
        timing_names_to_exclude(names, names)
