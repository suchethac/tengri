# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for dust_attenuation grammar validation.

Ensure that passing a non-dict value for dust_attenuation raises ValueError
with helpful guidance about the correct grammar. This guards against common
mistakes like passing an attenuation law name as dust_attenuation directly.
"""

from __future__ import annotations

import pytest

from tengri.parameters import Fixed, parse_groups

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _default_fixed():
    """``Fixed(DEFAULT)`` built off ``pytest.importorskip("tengri")``.

    Preserves the original ``importorskip("tengri").FIXED`` guard idiom (skip
    the module cleanly if tengri is not installed) without duplicating the
    ``importorskip`` call itself -- what these tests actually exercise is
    dust_attenuation grammar validation, so the sfh wildcard here only needs
    to be *some* valid disposition.
    """
    tengri = pytest.importorskip("tengri")
    return tengri.Fixed(tengri.DEFAULT)


def test_dust_attenuation_string_law_raises():
    """Passing an attenuation law name as dust_attenuation string raises with guidance."""
    with pytest.raises(ValueError, match="dust_attenuation must be a group dict"):
        parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": _default_fixed()},
            dust_attenuation="calzetti",  # Wrong: this is a law name, not a group dict
        )


def test_dust_attenuation_bool_true_raises():
    """Passing a bool True for dust_attenuation raises with guidance."""
    with pytest.raises(ValueError, match="dust_attenuation must be a group dict"):
        parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": _default_fixed()},
            dust_attenuation=True,  # Wrong: not a dict
        )


def test_dust_attenuation_int_raises():
    """Passing an int for dust_attenuation raises with guidance."""
    with pytest.raises(ValueError, match="dust_attenuation must be a group dict"):
        parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": _default_fixed()},
            dust_attenuation=42,  # Wrong: not a dict
        )


def test_dust_attenuation_none_is_allowed():
    """Passing dust_attenuation=None (or omitting it) is allowed and means OFF."""
    # Explicit None should be treated as omitted
    params_none = parse_groups(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": _default_fixed()},
        dust_attenuation=None,
    )
    assert params_none.dust_model == "off"
    assert params_none.free_params == []
