# SPDX-License-Identifier: BSD-3-Clause
"""Activation matrix: omitted optional groups equal {'type': 'none'}.

For each optional physics block, verify that omitting it and passing
{'type': 'none'} (where supported) produce the same free params.
"""

from __future__ import annotations

import pytest

from tengri.parameters import FIXED, Fixed, parse_groups

pytestmark = [pytest.mark.contract]


def test_dust_attenuation_omitted_equals_none():
    """Omitting dust_attenuation produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={"type": "none"},
    )
    # Both should have identical free params (no dust params).
    assert params_omitted.free_params == params_none.free_params
    # Both should have dust_model='off'
    assert params_omitted.dust_model == "off"
    assert params_none.dust_model == "off"


def test_dust_emission_omitted_equals_none():
    """Omitting dust_emission produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        dust_emission={"type": "none"},
    )
    # Both should have identical free params (no dust_emission params).
    assert params_omitted.free_params == params_none.free_params


def test_neb_omitted_equals_none():
    """Omitting neb produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        neb={"type": "none"},
    )
    # Both should have identical free params (no neb params).
    assert params_omitted.free_params == params_none.free_params


def test_shock_omitted_equals_none():
    """Omitting shock produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        shock={"type": "none"},
    )
    # Both should have identical free params (no shock params).
    assert params_omitted.free_params == params_none.free_params


def test_agn_omitted_equals_none():
    """Omitting agn produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        agn={"type": "none"},
    )
    # Both should have identical free params (no agn params).
    assert params_omitted.free_params == params_none.free_params


def test_igm_omitted_equals_none():
    """Omitting igm produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        igm={"type": "none"},
    )
    # Both should have identical free params (no igm params).
    assert params_omitted.free_params == params_none.free_params


def test_xray_omitted_equals_none():
    """Omitting xray produces identical free_params as {'type': 'none'}."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        xray={"type": "none"},
    )
    # Both should have identical free params (no xray params).
    assert params_omitted.free_params == params_none.free_params


def test_radio_omitted_equals_none():
    """Omitting radio produces identical free_params as {'type': 'none'} (composable form)."""
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
        radio={"sf": {"type": "none"}, "agn": {"type": "none"}},
    )
    # Both should have identical free params (no radio params).
    assert params_omitted.free_params == params_none.free_params


def test_dust_model_explicit_off_when_dust_attenuation_omitted():
    """When dust_attenuation is omitted, dust_model is explicitly 'off'."""
    params = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    # dust_model should be 'off', not the historical default 'two_component'
    assert params.dust_model == "off"
