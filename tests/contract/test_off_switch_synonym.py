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

import warnings

import pytest

from tengri import Parameters
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


# ── #2260's per-source dust-screen selectors join the family ───────────────
#
# ``nebular_screen`` / ``shock_screen`` / ``agn_screen`` (and their flat
# ``dust_*_screen`` spellings) accept ``'off'`` for ``'none'`` too. They do so
# by passing each raw value through the SAME ``_normalize_off_switch`` every
# group's ``type`` goes through -- at the point the value is read, before
# ``_dust_keys.resolve_screen_choices`` validates it -- rather than by a
# second synonym table of their own. ``agn_screen`` is the one selector whose
# only accepted value today is 'none', so 'off' is the only other spelling it
# takes at all.

_TWO_COMPONENT = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}


def _flat_two_component(**screen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Parameters(
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_law_diff="calzetti",
            redshift=Fixed(0.1),
            **screen,
        )


@pytest.mark.parametrize("source", ["nebular", "shock", "agn"])
def test_screen_selector_off_matches_none(source):
    """``{source}_screen='off'`` resolves exactly as ``'none'``, on both surfaces."""
    key = f"{source}_screen"
    off = parse_groups(**_BASE, dust_attenuation={**_TWO_COMPONENT, key: "off"})
    none = parse_groups(**_BASE, dust_attenuation={**_TWO_COMPONENT, key: "none"})
    assert off.free_params == none.free_params
    assert getattr(off, f"dust_{key}") == getattr(none, f"dust_{key}") == "none"

    flat_off = _flat_two_component(**{f"dust_{key}": "off"})
    flat_none = _flat_two_component(**{f"dust_{key}": "none"})
    assert getattr(flat_off, f"dust_{key}") == getattr(flat_none, f"dust_{key}") == "none"


def test_screen_selectors_read_the_one_off_switch_vocabulary(monkeypatch):
    """The selectors have no synonym table of their own.

    A spelling admitted to ``_OFF_SWITCH_SPELLINGS`` -- the one home of the
    vocabulary -- reaches all three selectors on both surfaces without any
    other edit. A private ``{'off': 'none'}`` table inside the validator
    would leave this spelling refused.
    """
    from tengri.parameters import groups as groups_mod

    monkeypatch.setattr(
        groups_mod, "_OFF_SWITCH_SPELLINGS", frozenset({*groups_mod._OFF_SWITCH_SPELLINGS, "nil"})
    )
    for source in ("nebular", "shock", "agn"):
        key = f"{source}_screen"
        grammar = parse_groups(**_BASE, dust_attenuation={**_TWO_COMPONENT, key: "nil"})
        assert getattr(grammar, f"dust_{key}") == "none"
        flat = _flat_two_component(**{f"dust_{key}": "nil"})
        assert getattr(flat, f"dust_{key}") == "none"
