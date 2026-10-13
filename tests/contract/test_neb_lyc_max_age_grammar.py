# SPDX-License-Identifier: BSD-3-Clause
"""The ``neb`` key ``lyc_max_age_yr``: the oldest stellar age whose ionizing photons Cue counts.

The value is ``'all'`` (default) or a positive finite age [yr]; ``1e8`` is the former hard
100 Myr cut. Cue-only, like ``nitrogen``. Round-trip through ``to_groups()`` is covered with
the other structural keys in ``test_structural_settings_roundtrip.py``.
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed
from tengri.config.exceptions import ConfigError
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


def _spec(**neb):
    return parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), **neb},
        redshift=Fixed(0.1),
    )


def test_default_is_all_ages():
    assert _spec().cue_lyc_max_age_yr == "all"


@pytest.mark.parametrize("value", ["all", 1.0e8, 3.0e8, 1e10, 100000000])
def test_legal_values_are_stored(value):
    assert _spec(lyc_max_age_yr=value).cue_lyc_max_age_yr == value


def test_default_value_is_not_emitted_by_to_groups():
    assert "lyc_max_age_yr" not in _spec().to_groups()["neb"]


@pytest.mark.parametrize(
    "value", [0, -1.0e8, float("nan"), float("inf"), True, "young", "ALL", None, [1e8], "1e8"]
)
def test_illegal_values_are_refused(value):
    with pytest.raises(ConfigError, match="lyc_max_age_yr"):
        _spec(lyc_max_age_yr=value)


@pytest.mark.parametrize("neb_type", ["cloudy", "cb19", "ssp", "none"])
def test_only_cue_accepts_the_key(neb_type):
    with pytest.raises(ConfigError, match="lyc_max_age_yr"):
        parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": neb_type, "lyc_max_age_yr": 1.0e8},
            redshift=Fixed(0.1),
        )


def test_the_value_is_part_of_the_parameters_cache_key():
    from tengri.parameters.parameters import _PARAMETERS_CACHE_KEY_POLICY

    assert "cue_lyc_max_age_yr" in _PARAMETERS_CACHE_KEY_POLICY
