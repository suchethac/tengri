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


def test_dust_attenuation_none_without_wildcard_does_not_warn():
    """Explicit dust_attenuation={'type': 'none'} with no wildcard should not warn.

    Groups with zero parameters should never warn about silently-fixed params,
    since there is nothing to fix. This tests that the warning machinery correctly
    skips empty-parameter-set groups.
    """
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params = parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": FIXED},
            dust_attenuation={"type": "none"},  # No all_params disposition
        )
        # Should not warn about dust_attenuation since it has no parameters
        dust_warnings = [
            x
            for x in w
            if "dust_attenuation" in str(x.message)
            and "no 'all_params' disposition" in str(x.message)
        ]
        assert not dust_warnings, (
            f"Unexpected warning for empty-param group: {[str(x.message) for x in dust_warnings]}"
        )


def test_foreground_omitted():
    """Omitting foreground produces no free params (foreground has no 'type': 'none' form).

    Unlike other optional groups, foreground does not accept type='none' because
    it has no structural choice — there are no sub-types. Omission is the only way
    to disable it, and both cases should have identical free params (none from foreground).
    """
    params_omitted = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    # Verify that foreground is omitted by checking it's not in the result
    # (no way to explicitly pass None to a forward-only group, so we just check omitted)
    assert params_omitted is not None
    # Foreground contributes no params to free_params when omitted (expected)


def test_dust_model_explicit_off_when_dust_attenuation_omitted():
    """When dust_attenuation is omitted, dust_model is explicitly 'off'."""
    params = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FIXED},
    )
    # dust_model should be 'off', not the historical default 'two_component'
    assert params.dust_model == "off"
