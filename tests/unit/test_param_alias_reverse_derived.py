# SPDX-License-Identifier: BSD-3-Clause
"""The reverse alias table is derived from the forward alias table (no drift)."""

from __future__ import annotations

import pytest

from tengri.parameters import _aliases

pytestmark = pytest.mark.unit

# The hand-written table this derivation replaced, frozen verbatim.
_LEGACY_REVERSE_ALIASES = {
    "sfh_dpl_alpha": "sfh_alpha",
    "sfh_dpl_beta": "sfh_beta",
    "sfh_dpl_tau_gyr": "sfh_tau_peak_gyr",
    "sfh_field_psd_sigma": "psd_sigma",
    "sfh_field_psd_tau_myr": "psd_tau_myr",
}


def test_reverse_aliases_equal_legacy_literal():
    assert _aliases._REVERSE_ALIASES == _LEGACY_REVERSE_ALIASES


def test_reverse_aliases_are_the_sfh_entries_of_the_forward_table():
    expected = {
        canonical: legacy
        for legacy, canonical in _aliases._LEGACY_PARAM_ALIASES.items()
        if canonical.startswith("sfh_")
    }
    assert expected == _aliases._REVERSE_ALIASES


@pytest.mark.parametrize("canonical", ["agn_lum_ratio", "dust_T", "agn_torus_frac"])
def test_non_sfh_forward_aliases_are_not_reversed(canonical):
    assert canonical not in _aliases._REVERSE_ALIASES
