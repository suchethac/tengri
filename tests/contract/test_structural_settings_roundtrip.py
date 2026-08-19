# SPDX-License-Identifier: BSD-3-Clause
"""Contract: every structural group setting survives ``to_groups()``.

``parse_groups`` accepts a set of non-parameter *structural* keys per group
(``_GROUP_STRUCTURAL_KEYS``) — the SFH→SSP kernel, the AGN normalization
policy, the WG00 screen selectors, the MW foreground screen. ``to_groups()``
is advertised as their inverse.

It was not. The emit side was hand-written per group and covered only three
of the eight, so the rest were accepted, stored on the spec, and then silently
dropped on the next round-trip:

* ``sfh['age_kernel']='dsps'`` reverted to the ``'cic'`` default — a 43% shift
  in the ``sfh_*_age_gyr`` gradient (#964);
* ``agn['norm']='independent'`` reverted to ``'cigale_joint'`` — energy-
  conserving AGN bookkeeping silently swapped back in (#556);
* ``igm['patchy']=True`` was worse than lost: the patchy *parameters* were
  emitted while the toggle that legalizes them was not, so re-parsing raised
  ``Unknown key 'bubble_mpc' in group 'igm'``;
* the WG00 ``dust_curve`` / ``geometry`` / ``structure`` selectors and the
  whole ``foreground`` group had no emit path at all.

The docstring's "roundtrip guarantee" is scoped to *free/fixed partitions and
distributions*, which is exactly why this class survived: the guarantee was
true as written, and structural settings were simply outside it.

The last test is the load-bearing one. The per-key sweep below proves the keys
that exist today round-trip; ``test_every_structural_key_has_a_roundtrip_rule``
proves the *next* key cannot be added without one.
"""

import numpy as np
import pytest

from tengri import FIXED
from tengri.parameters.groups import (
    _GROUP_STRUCTURAL_KEYS,
    _STRUCTURAL_ROUNDTRIP,
    parse_groups,
)

pytestmark = pytest.mark.contract


#: (id, parse_groups kwargs, spec attribute, expected value).
#:
#: Each case sets one structural key to a legal NON-default value. A case whose
#: value equals the default would pass vacuously — the emit rule is
#: "emit when it differs from the default", so a default-valued case exercises
#: nothing.
#:
#: ``foreground['law']`` is absent for that reason and not by oversight:
#: ``_VALID_FOREGROUND_LAWS`` holds exactly one law today, so no non-default
#: value exists to move it to. Its rule is still covered declaratively by
#: :func:`test_every_structural_key_has_a_roundtrip_rule`, and a second law
#: can be swept here the day one is registered.
CASES = [
    (
        "sfh.age_kernel",
        dict(sfh={"type": "dpl", "all_params": FIXED, "age_kernel": "dsps"}),
        "age_kernel",
        "dsps",
    ),
    (
        "sfh.field_centering",
        dict(sfh={"type": ["dpl", "field"], "all_params": FIXED, "field_centering": 0.5}),
        "field_centering",
        0.5,
    ),
    (
        "sfh.bin_edges_gyr",
        dict(
            sfh={
                "type": "continuity",
                "all_params": FIXED,
                "bin_edges_gyr": [0.0, 0.1, 0.5, 1.0, 13.0],
            }
        ),
        "bin_edges_gyr",
        [0.0, 0.1, 0.5, 1.0, 13.0],
    ),
    (
        "stellar.met_mode",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            met={"type": "chem_evol", "all_params": FIXED},
        ),
        "met_mode",
        "chem_evol",
    ),
    (
        "dust.dust_curve",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "wg00", "all_params": FIXED, "dust_curve": "smc"},
        ),
        "dust_wg00_curve",
        "smc",
    ),
    (
        "dust.geometry",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "wg00", "all_params": FIXED, "geometry": "dusty"},
        ),
        "dust_wg00_geometry",
        "dusty",
    ),
    (
        "dust.structure",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "wg00", "all_params": FIXED, "structure": "clumpy"},
        ),
        "dust_wg00_structure",
        "clumpy",
    ),
    (
        "neb.full_catalog",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            neb={"type": "cue", "all_params": FIXED, "full_catalog": True},
        ),
        "cue_full_catalog",
        True,
    ),
    (
        "shock.norm",
        dict(sfh={"type": "dpl", "all_params": FIXED}, shock={"norm": "lhalpha"}),
        "shock_norm",
        "lhalpha",
    ),
    (
        "shock.abundance",
        dict(sfh={"type": "dpl", "all_params": FIXED}, shock={"abundance": "twice_solar"}),
        "shock_abundance",
        "twice_solar",
    ),
    (
        "igm.patchy",
        dict(sfh={"type": "dpl", "all_params": FIXED}, igm={"type": "inoue", "patchy": True}),
        "igm_patchy",
        True,
    ),
    (
        "agn.norm",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            agn={"type": "composable", "all_params": FIXED, "norm": "independent"},
        ),
        "agn_norm",
        "independent",
    ),
    (
        "foreground.ebmv_mw",
        dict(sfh={"type": "dpl", "all_params": FIXED}, foreground={"ebmv_mw": 0.07}),
        "foreground_ebmv_mw",
        0.07,
    ),
    (
        "foreground.rv",
        dict(
            sfh={"type": "dpl", "all_params": FIXED},
            foreground={"ebmv_mw": 0.07, "rv": 2.9},
        ),
        "foreground_rv",
        2.9,
    ),
]


def _same(a, b) -> bool:
    """Value equality that tolerates array-valued settings (bin_edges_gyr)."""
    if isinstance(a, (list, tuple)) or hasattr(a, "shape"):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


@pytest.mark.parametrize("case_id, kwargs, attr, expected", CASES, ids=[c[0] for c in CASES])
class TestStructuralSettingsSurviveRoundtrip:
    def test_parse_stores_the_setting(self, case_id, kwargs, attr, expected):
        """Guard the fixture: a case that never set the value proves nothing.

        If the grammar stops routing a key to this attribute, the round-trip
        test below would compare ``default == default`` and pass vacuously.
        """
        spec = parse_groups(**kwargs)
        assert _same(getattr(spec, attr, None), expected), (
            f"{case_id}: parse_groups did not store the value — the round-trip "
            f"assertion below would be vacuous"
        )

    def test_setting_survives_to_groups(self, case_id, kwargs, attr, expected):
        spec = parse_groups(**kwargs)
        rebuilt = parse_groups(**spec.to_groups())
        assert _same(getattr(rebuilt, attr, None), expected), (
            f"{case_id}: to_groups() dropped the setting — it silently reverted "
            f"to {getattr(rebuilt, attr, None)!r} instead of {expected!r}"
        )


def test_patchy_igm_reparses_instead_of_raising():
    """#964's sharpest edge: the emitted dict was not merely lossy, it was invalid.

    Patchy reionization publishes ``bubble_mpc`` / ``x_HI`` into the ``igm``
    group, but those keys are only legal when ``patchy`` is on. Emitting the
    parameters without the toggle produced a dict that ``parse_groups``
    rejected outright.
    """
    spec = parse_groups(
        sfh={"type": "dpl", "all_params": FIXED},
        igm={"type": "inoue", "patchy": True},
    )
    groups = spec.to_groups()
    assert groups["igm"].get("patchy") is True, (
        f"the toggle that legalizes the patchy params is missing: {groups['igm']}"
    )
    parse_groups(**groups)  # must not raise


def test_default_spec_grows_no_spurious_groups():
    """Emitting only non-defaults keeps the round-trip dict quiet.

    An always-emit rule would force ``foreground={...}`` and ``met={...}``
    onto every spec, which breaks call sites that diff to_groups() output.
    """
    groups = parse_groups(sfh={"type": "dpl", "all_params": FIXED}).to_groups()
    assert "foreground" not in groups
    assert "age_kernel" not in groups.get("sfh", {})
    assert "met_mode" not in groups.get("stellar", {})


def test_every_structural_key_has_a_roundtrip_rule():
    """The anti-drift guard: a new structural key must declare how it returns.

    ``_GROUP_STRUCTURAL_KEYS`` says what ``parse_groups`` *accepts*;
    ``_STRUCTURAL_ROUNDTRIP`` says how ``to_groups()`` gives it *back*. When
    those two were independent descriptions of one set they drifted, and five
    of eight groups lost their settings. This asserts every accepted key is
    accounted for by exactly one mechanism.
    """
    # Handled by _extract_group_type and the wildcard analyzer, not the table.
    meta_keys = {"type", "*"}
    # Dust attenuation laws stay hand-written in _add_structural_settings:
    # law/law_bc/law_diff are an explicit XOR (never a default comparison:
    # the emit collapses to shared 'law' when both screens agree, else the
    # law_bc/law_diff pair), the per-component overrides live in a flattened
    # dict, and two booleans are stored as a float cutoff.
    hand_written = {
        "law",
        "law_bc",
        "law_diff",
        "law_neb",
        "lyman_cutoff",
        "lyc_absorb_all",
        "eb_include_lyc",
    } | {
        f"{stem}_{comp}"
        for stem in ("slope", "bump_strength", "delta", "Rv")
        for comp in ("bc", "diff", "neb")
    }

    missing: list[str] = []
    for group, keys in _GROUP_STRUCTURAL_KEYS.items():
        covered = {e.key for e in _STRUCTURAL_ROUNDTRIP.get(group, ())}
        for key in keys:
            if key in meta_keys or key in covered:
                continue
            # A key naming a nested group (dust.emission, agn.torus, igm.dla)
            # is emitted by the per-group walk, not by a structural rule.
            if f"{group}.{key}" in _GROUP_STRUCTURAL_KEYS:
                continue
            if group == "dust" and key in hand_written:
                continue
            missing.append(f"{group}.{key}")

    assert not missing, (
        "structural keys parse_groups accepts but to_groups() has no rule for: "
        f"{sorted(missing)}. Add an entry to _STRUCTURAL_ROUNDTRIP (or, if the "
        "emit is genuinely not a default comparison, to the hand-written set in "
        "_add_structural_settings and to this test's allowlist)."
    )


def test_roundtrip_table_targets_real_attributes():
    """Every table entry must name an attribute Parameters actually has.

    A typo'd attribute (``dust_geometry`` for ``dust_wg00_geometry``) reads as
    the default forever, so the key would silently never emit — the same
    failure the table exists to prevent, one layer down.
    """
    spec = parse_groups(sfh={"type": "dpl", "all_params": FIXED})
    unknown = [
        f"{group}.{entry.key} -> {entry.attr}"
        for group, entries in _STRUCTURAL_ROUNDTRIP.items()
        for entry in entries
        if not hasattr(spec, entry.attr)
    ]
    assert not unknown, f"_STRUCTURAL_ROUNDTRIP names nonexistent attributes: {unknown}"
