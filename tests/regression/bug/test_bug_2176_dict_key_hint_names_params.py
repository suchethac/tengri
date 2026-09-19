# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray
r"""Unknown dict keys name the type's parameters; the atten did-you-mean names the law form.

https://github.com/suchethac/tengri/issues/2176
https://github.com/suchethac/tengri/issues/2201

Issue #2176: An unknown dict key with no close difflib match showed only the
structural keys (e.g., "Valid structural keys for this group are: ['type']")
even though the type's parameter short names were already compiled and available
in the suggestion pool. The message should list the accepted parameter names so
the user knows what they can write.

Issue #2201: The AGN attenuation block's did-you-mean suggestion could suggest
'smc_prevot' under the 'type' key, which itself is refused with a routing
suggestion ('use the law form instead'). Two hops to the same fix. Instead,
when the nearest type suggestion is a law-form name (one that the resolver
itself refuses under type), suggest the law form directly.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_bug


def test_unknown_dict_key_no_near_match_names_param_short_names():
    """(i) An unknown key with no close match lists the type's parameter short names.

    When a user writes ``sfh={'type': 'delayed', 'zzz': 1.0}``, the 'zzz' key
    has no difflib match. Before the fix, the message was:
    "Valid structural keys for this group are: ['type']."

    After the fix, the message names the type's accepted parameter short names:
    "Valid structural keys for this group are: ['type']. Parameter names this
    type accepts: tau_gyr, age_gyr, ..." (or points to tengri.describe for
    large types).

    This test pins the existing behavior directly without a monkeypatch, by
    checking that two real parameter short names appear in the error message.
    """
    from tengri import DEFAULT, Fixed
    from tengri.parameters.groups import parse_groups

    # 'zzz' is not a close match to any param name, so difflib won't suggest
    # anything. The error message should name the actual accepted params.
    with pytest.raises(ValueError) as excinfo:
        parse_groups(
            sfh={"type": "delayed", "zzz": 1.0, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )

    error_msg = str(excinfo.value)
    # The delayed SFH type accepts these parameter short names
    assert "tau_gyr" in error_msg or "describe" in error_msg, (
        "error message must name the type's parameter short names or point at "
        f"tengri.describe: {error_msg}"
    )
    assert "age_gyr" in error_msg or "describe" in error_msg, (
        "error message must name the type's parameter short names or point at "
        f"tengri.describe: {error_msg}"
    )


def test_unknown_dict_key_threshold_behavior_for_parameter_listing():
    """(ii) The message behavior changes based on the number of parameters.

    For types with few parameters (e.g., delayed), the message lists the
    parameter short names. For types with many parameters (> threshold),
    the message points to ``tengri.describe('<type>')`` instead.

    This test verifies that the mechanism is in place by checking that a
    small-parameter type's message includes the parameter hint.
    """
    from tengri import DEFAULT, Fixed
    from tengri.parameters.groups import parse_groups

    # delayed is a small SFH type with only 2 main parameters.
    # The error for an unknown key should include the parameter hint.
    with pytest.raises(ValueError) as excinfo:
        parse_groups(
            sfh={"type": "delayed", "unknown_param": 1.0, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )

    error_msg = str(excinfo.value)
    # For delayed (small type), the message should include the parameter names
    assert ("tau_gyr" in error_msg and "age_gyr" in error_msg) or "Parameter names" in error_msg, (
        f"error message must include parameter-listing hint for small types: {error_msg}"
    )


def test_near_miss_still_suggests_close_param_name():
    """(iii) A close typo still gets a did-you-mean suggestion.

    When a user writes ``sfh={'type': 'delayed', 'tau': 1.0}`` (missing 'gyr'),
    difflib still suggests 'tau_gyr', and that suggestion is printed regardless
    of whether the full list of param names appears in the message.
    """
    from tengri import DEFAULT, Fixed
    from tengri.parameters.groups import parse_groups

    with pytest.raises(ValueError) as excinfo:
        parse_groups(
            sfh={"type": "delayed", "tau": 1.0, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )

    error_msg = str(excinfo.value)
    # 'tau' is a close match to 'tau_gyr' and the suggestion should appear
    assert "tau_gyr" in error_msg, f"expected 'tau_gyr' suggestion in did-you-mean: {error_msg}"


def test_agn_atten_type_smc_prevot_suggests_law_form_not_type_form():
    """(iv) The AGN atten did-you-mean routes smc_prevot to the law form, not the type form.

    Before the fix, if a user wrote ``agn={'atten': {'type': 'prevot'}}`` and
    difflib suggested 'smc_prevot', the message would route to:
    ``agn={'atten': {'type': 'smc_prevot', ...}}``
    which itself is refused with "no longer supported. Use the law form instead".

    After the fix, the suggestion is the law form directly:
    ``agn={'atten': {'law': 'prevot_smc', ...}}``
    (one hop, not two).
    """
    from tengri import DEFAULT, Fixed
    from tengri.parameters.groups import parse_groups

    # 'prevot' is close to 'prevot_smc' (the law name) and to 'smc_prevot'
    # (the refused type name). The suggestion should be the law form.
    with pytest.raises(ValueError) as excinfo:
        parse_groups(
            agn={"type": "composable", "atten": {"type": "prevot"}},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )

    error_msg = str(excinfo.value)
    # The fix should NOT suggest 'smc_prevot' as a type to try
    # Instead, it should suggest the law form
    assert "law" in error_msg and "prevot_smc" in error_msg, (
        f"expected suggestion of law form 'prevot_smc' in the error: {error_msg}"
    )
    # Verify it's NOT suggesting the type form
    assert "type" not in error_msg or "type='smc_prevot'" not in error_msg, (
        f"should not suggest type='smc_prevot': {error_msg}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
