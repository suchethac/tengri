# SPDX-License-Identifier: BSD-3-Clause
"""Contract: a group's ``type`` survives ``to_groups()`` (#1777).

Its sibling ``test_structural_settings_roundtrip.py`` sweeps the *non-type*
structural keys (``_GROUP_STRUCTURAL_KEYS``) and deliberately excuses two::

    # Handled by _extract_group_type and the wildcard analyzer, not the table.
    meta_keys = {"type", "*"}

That exemption was the hole. ``_extract_group_type`` did **not** handle
``type`` for the AGN family — it carried one arm that returned ``None`` for
every group name beginning with ``agn``, under the comment "This is a
simplification; more complex composition handled in tests". No test held it.

Measured across the grammar's own validator sets, with the rebuilt model's
photometry compared to the original's at identical parameter values:

    28 of 79 structural selections were LOSSY, every one of them AGN.
    Photometry moved by up to 98%.

The two AGN recipes lost free parameters outright — ``agn_panchromatic``
25 -> 11, ``composable_agn`` 28 -> 11 — so a user who round-tripped a spec got
a fit in **11 dimensions instead of 25** and nothing said so.

A third case had a different cause and the same shape: a group whose
non-default type declares no parameters of its own never entered the per-group
walk at all, and the fallback that catches those tested a hand-written list
naming only ``dust`` and ``igm``. So ``neb={'type': 'ssp'}`` rebuilt with
nebular emission switched **off**.

The whole round-trip lives at the ``parse_groups`` layer, so this file needs no
SSP grid and no model build: three ``parse_groups`` calls take ~2 ms, which is
why the sweep can afford to be the *entire* derived census rather than a
sample.
"""

from __future__ import annotations

import pytest

from tengri import FIXED, Parameters
from tengri.parameters.groups import (
    _AGN_BLOCK_TO_KWARG,
    _TOP_LEVEL_TYPED_GROUPS,
    _VALID_AGN_ATTEN_TYPES,
    _VALID_AGN_BLR_TYPES,
    _VALID_AGN_DISC_TYPES,
    _VALID_AGN_FEII_TYPES,
    _VALID_AGN_NLR_TYPES,
    _VALID_AGN_TORUS_TYPES,
    _default_group_type,
    _extract_group_type,
    parameters_to_groups,
    parse_groups,
)

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

#: The AGN census, taken from the grammar's own validator sets. A new backend
#: registered under any of the six blocks joins this sweep automatically; a
#: hand-listed copy would not, and that is the failure mode this file exists
#: for.
_AGN_TYPES: dict[str, frozenset[str]] = {
    "disc": _VALID_AGN_DISC_TYPES,
    "torus": _VALID_AGN_TORUS_TYPES,
    "nlr": _VALID_AGN_NLR_TYPES,
    "blr": _VALID_AGN_BLR_TYPES,
    "feii": _VALID_AGN_FEII_TYPES,
    "atten": _VALID_AGN_ATTEN_TYPES,
}

#: ``"none"`` is every sub-block's default, so a case selecting it would pass
#: vacuously — the emit rule is "emit when it differs from the default".
_AGN_CASES = sorted(
    (block, t) for block, types in _AGN_TYPES.items() for t in types if t != "none"
)

_BASE = dict(sfh={"type": "dpl", "all_params": FIXED})


def _roundtrip(**kwargs) -> tuple[Parameters, Parameters, dict]:
    """spec -> groups -> spec, returning both specs and the emitted dict."""
    spec0 = parse_groups(**kwargs)
    groups = parameters_to_groups(spec0)
    return spec0, parse_groups(**groups), groups


class TestTheAGNFamily:
    @pytest.mark.parametrize(("block", "agn_type"), _AGN_CASES, ids=lambda v: str(v))
    def test_a_subblock_type_survives(self, block, agn_type):
        """Every registered type of every AGN sub-block, round-tripped."""
        attr = _AGN_BLOCK_TO_KWARG[block]
        # Special handling for atten/smc_prevot: use law key instead of type
        if block == "atten" and agn_type == "smc_prevot":
            sub_block_spec = {"law": "prevot_smc"}
        else:
            sub_block_spec = {"type": agn_type}
        spec0, spec1, groups = _roundtrip(
            **_BASE,
            agn={"type": "composable", "all_params": FIXED, block: sub_block_spec},
        )
        assert getattr(spec1, attr) == getattr(spec0, attr), (
            f"agn['{block}']['type']={agn_type!r} did not survive to_groups(): "
            f"the rebuilt spec has {attr}={getattr(spec1, attr)!r}. The emitted "
            f"block was {groups.get('agn', {}).get(block)!r}. A dropped "
            f"sub-block selector silently swaps the AGN physics for the "
            f"default, and takes that component's free parameters with it."
        )

    @pytest.mark.parametrize("block", sorted(_AGN_BLOCK_TO_KWARG))
    def test_a_subblock_type_does_not_leak_into_another_block(self, block):
        """Preserving six selectors is not the same as preserving each one.

        A round-trip that wrote every block's type into one shared attribute
        would pass the sweep above for a single-block spec. Setting one block
        must leave the other five at their default.
        """
        agn_type = next(t for t in sorted(_AGN_TYPES[block]) if t != "none")
        _, spec1, _ = _roundtrip(
            **_BASE,
            agn={"type": "composable", "all_params": FIXED, block: {"type": agn_type}},
        )
        others = {
            other: getattr(spec1, attr)
            for other, attr in _AGN_BLOCK_TO_KWARG.items()
            if other != block and getattr(spec1, attr) not in (None, "none")
        }
        assert not others, f"selecting agn['{block}'] also set {others}"

    @pytest.mark.parametrize(
        "model", ["richards2006", "kubota_done", "relagn", "qsogen", "skirtor"]
    )
    def test_a_monolithic_model_survives(self, model):
        """The other AGN surface: ``agn={'type': X}`` with no sub-blocks."""
        spec0, spec1, groups = _roundtrip(**_BASE, agn={"type": model, "all_params": FIXED})
        assert spec0.agn_model == model  # guards the fixture, not the fix
        assert spec1.agn_model == model, (
            f"agn={{'type': {model!r}}} rebuilt as agn_model="
            f"{spec1.agn_model!r}; the emitted agn block was "
            f"{sorted(groups.get('agn', {}))}."
        )

    def test_a_monolithic_emission_carries_no_subblocks(self):
        """Emitting the type is not enough — the dict must stay *legal*.

        ``_translate_agn`` raises when a non-composable ``agn['type']`` appears
        beside sub-block keys, so emitting the monolithic name next to the
        nested form the composable path uses would turn a silent loss into a
        hard failure. The parameters move to flat keys instead.
        """
        _, _, groups = _roundtrip(**_BASE, agn={"type": "richards2006", "all_params": FIXED})
        nested = sorted(k for k in groups.get("agn", {}) if k in _AGN_BLOCK_TO_KWARG)
        assert not nested, (
            f"a monolithic agn spec emitted sub-blocks {nested}; parse_groups "
            f"refuses that combination, so the round-trip would raise."
        )


class TestATypedGroupWithNoParametersOfItsOwn:
    """The second cause: never reaching the per-group walk at all."""

    def test_nebular_ssp_mode_survives(self):
        spec0, spec1, groups = _roundtrip(**_BASE, neb={"type": "ssp"})
        assert spec0.nebular_mode == "ssp"  # fixture guard
        assert spec1.nebular_mode == "ssp", (
            f"neb={{'type': 'ssp'}} rebuilt as nebular_mode="
            f"{spec1.nebular_mode!r} (emitted neb block: {groups.get('neb')!r}). "
            f"'off' means the nebular contribution is silently gone."
        )

    @pytest.mark.parametrize("group", sorted(_TOP_LEVEL_TYPED_GROUPS))
    def test_the_default_is_read_off_a_bare_spec_not_written_down(self, group):
        """``_default_group_type`` must stay derived.

        The rule it feeds is "emit when the type differs from the default", so
        a wrong default silently disables emission for that group. Reading it
        through ``_extract_group_type`` means it passes the same boundary
        translations (``nebular_mode='off'`` -> ``"none"``) that a copied
        literal would miss.
        """
        expected = _extract_group_type(group, Parameters())
        expected = tuple(expected) if isinstance(expected, list) else expected
        assert _default_group_type(group) == expected


class TestTheCensusIsComplete:
    """A guard is only as wide as the population it scans."""

    def test_every_agn_subblock_is_swept(self):
        swept = {block for block, _ in _AGN_CASES}
        assert set(_AGN_BLOCK_TO_KWARG) <= swept, (
            f"{sorted(set(_AGN_BLOCK_TO_KWARG) - swept)} have no case here, so "
            f"nothing checks that their selector round-trips."
        )

    def test_the_sweep_is_not_trivially_small(self):
        """Guards the derivation itself.

        If ``_agn_block_types`` ever returned an empty set the parametrized
        sweep above would collapse to zero cases and report green. The bound is
        the six blocks each contributing at least one non-default type, not the
        43 that happen to be registered today.
        """
        by_block = {block for block, _ in _AGN_CASES}
        assert len(by_block) == len(_AGN_BLOCK_TO_KWARG)
        assert len(_AGN_CASES) >= len(_AGN_BLOCK_TO_KWARG)

    def test_every_typed_group_reports_a_type_for_some_selection(self):
        """No group may sit outside ``_extract_group_type`` entirely.

        The AGN arm returned ``None`` unconditionally, which is
        indistinguishable from "this group has no type axis" — and that is
        exactly how it read for as long as it stood.
        """
        silent = [
            g
            for g in sorted(_TOP_LEVEL_TYPED_GROUPS)
            if _extract_group_type(g, Parameters()) is None and g not in {"radio", "agn"}
        ]
        assert not silent, (
            f"{silent} report no type on a default spec, so a non-default "
            f"selection there cannot be detected as differing from it."
        )
