# SPDX-License-Identifier: BSD-3-Clause
"""``{'type': 'off'}`` is a synonym for ``{'type': 'none'}`` in every group that
carries an off switch.

Fresh-user audit (2026-09): 'off' already disabled ``dust_attenuation``,
``dust_emission`` and ``agn``, but ``neb``, ``shock``, ``radio``'s ``sf``/``agn``
sub-blocks, ``xray`` and ``igm`` rejected it with ``Unknown type 'off'`` -- the
same disable request, spelled one of two equally reasonable ways, honored by
three groups and refused by five. These tests pin ``'off'`` as an accepted
synonym for ``'none'`` across all eight groups, by comparing the fully
resolved config each spelling produces (the free-parameter set plus the
group's own resolved kwarg), not merely that neither call raises.
"""

from __future__ import annotations

import pytest

from tengri.parameters import DEFAULT, Fixed, parse_groups

pytestmark = [pytest.mark.contract]

_BASE = {
    "redshift": Fixed(0.1),
    "sfh": {"type": "dpl", "all_params": Fixed(DEFAULT)},
}


def test_dust_attenuation_off_matches_none():
    off = parse_groups(**_BASE, dust_attenuation={"type": "off"})
    none = parse_groups(**_BASE, dust_attenuation={"type": "none"})
    assert off.free_params == none.free_params
    assert off.dust_model == none.dust_model == "off"


def test_dust_emission_off_matches_none():
    off = parse_groups(**_BASE, dust_emission={"type": "off"})
    none = parse_groups(**_BASE, dust_emission={"type": "none"})
    assert off.free_params == none.free_params


def test_agn_off_matches_none():
    off = parse_groups(**_BASE, agn={"type": "off"})
    none = parse_groups(**_BASE, agn={"type": "none"})
    assert off.free_params == none.free_params
    assert off.agn_model == none.agn_model is None


def test_neb_off_matches_none():
    off = parse_groups(**_BASE, neb={"type": "off"})
    none = parse_groups(**_BASE, neb={"type": "none"})
    assert off.free_params == none.free_params
    assert off.nebular_mode == none.nebular_mode == "off"
    assert off.nebular == none.nebular is False


def test_shock_off_matches_none():
    off = parse_groups(**_BASE, shock={"type": "off"})
    none = parse_groups(**_BASE, shock={"type": "none"})
    assert off.free_params == none.free_params
    assert off.shock == none.shock is False


def test_radio_off_matches_none():
    """Both radio sub-blocks (``sf`` and ``agn``) accept the synonym."""
    off = parse_groups(**_BASE, radio={"sf": {"type": "off"}, "agn": {"type": "off"}})
    none = parse_groups(**_BASE, radio={"sf": {"type": "none"}, "agn": {"type": "none"}})
    assert off.free_params == none.free_params
    assert off.radio == none.radio is False


def test_xray_off_matches_none():
    off = parse_groups(**_BASE, xray={"type": "off"})
    none = parse_groups(**_BASE, xray={"type": "none"})
    assert off.free_params == none.free_params
    assert off.xray == none.xray is False


def test_igm_off_matches_none():
    off = parse_groups(**_BASE, igm={"type": "off"})
    none = parse_groups(**_BASE, igm={"type": "none"})
    assert off.free_params == none.free_params
    assert off.apply_igm == none.apply_igm is False
