# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE spellings refused in tengri's AGN grammar name the tengri key to use.

A CIGALE user who writes ``delta`` on the disc, ``disk_type`` or
``lambda_fracAGN`` is refused, and the refusal carries the one-line hint from
:data:`CIGALE_AGN_SPELLINGS`. A key the selected type reads is still accepted.

Tier: contract.
"""

import pytest

from tengri import DEFAULT, Fixed
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


def _composable(**sub_blocks):
    return {"type": "composable", **sub_blocks}


def test_delta_on_fritz_torus_is_refused():
    """The Fritz torus reads no delta; writing one is refused, not dropped."""
    agn = _composable(torus={"type": "fritz", "delta": Fixed(-0.3), "all_params": Fixed(DEFAULT)})
    with pytest.raises(ParameterError, match=r"does not read the key 'delta'"):
        parse_groups(agn=agn)


def test_key_read_by_another_torus_type_is_named_in_the_refusal():
    """A torus key a different torus type reads: the refusal names that type."""
    agn = _composable(
        torus={"type": "fritz", "tau_skirtor": Fixed(0.5), "all_params": Fixed(DEFAULT)}
    )
    with pytest.raises(ParameterError, match=r"'tau_skirtor' is read by 'skirtor'"):
        parse_groups(agn=agn)


def test_delta_on_disc_hints_both_disc_delta_keys():
    agn = _composable(
        disc={"type": "schartmann2005", "delta": Fixed(0.3), "all_params": Fixed(DEFAULT)}
    )
    with pytest.raises(ValueError) as info:
        parse_groups(agn=agn)
    msg = str(info.value)
    assert "CIGALE's delta" in msg
    assert "agn_cigale_disk_delta" in msg
    assert "agn_cigale_adaf_delta" in msg


def test_disk_type_on_disc_hints_the_type_names():
    agn = _composable(
        disc={"type": "schartmann2005", "disk_type": Fixed(1), "all_params": Fixed(DEFAULT)}
    )
    with pytest.raises(ValueError) as info:
        parse_groups(agn=agn)
    msg = str(info.value)
    assert "CIGALE's disk_type" in msg
    for name in ("'skirtor'", "'schartmann2005'", "'adaf_lopez2024'"):
        assert name in msg


def test_lambda_frac_agn_hints_agn_ir_frac():
    agn = _composable(
        torus={"type": "skirtor", "lambda_fracAGN": "5/20", "all_params": Fixed(DEFAULT)}
    )
    with pytest.raises(ValueError) as info:
        parse_groups(agn=agn)
    msg = str(info.value)
    assert "CIGALE's lambda_fracAGN" in msg
    assert "agn_ir_frac" in msg
    assert "window" in msg
