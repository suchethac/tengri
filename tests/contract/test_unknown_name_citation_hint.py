# SPDX-License-Identifier: BSD-3-Clause
"""Tests for citation key recognition in unknown-name errors.

When a user provides a citation key (like 'charlot_fall2000') as a type or
parameter name, error messages should recognize it as a known citation key and
explain which registry entry it cites, rather than falling back to difflib
suggestions alone.
"""

import re

import pytest

from tengri.citations import associations as _assoc
from tengri.citations.resolve import (
    _REGISTRY_NAMESPACE_TABLES,
    NAME_TO_BIBKEY,
    citation_key_hint,
    registry_names_for_citation_key,
)
from tengri.parameters import Fixed
from tengri.parameters.groups import _names_accepted_anywhere, parse_groups

pytestmark = pytest.mark.contract


def _every_routed_site_menu() -> dict[str, frozenset[str]]:
    """The 22 routed validators' own ``valid_names``, kept separate (not
    unioned) so a sweep can check the "valid at site" branch names only
    values that site itself accepts, independent of
    :func:`_names_accepted_anywhere`'s union (#2429 opus review round 2,
    item 1's exhaustive sweep). Built from the same menu-deriving functions
    the validators call, never a hand list.
    """
    import tengri.parameters.groups as g
    from tengri.components.agn.unified import AGN_MODELS, monolithic_agn_model_names
    from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS
    from tengri.components.stellar.sfh.met_registry import MET_REGISTRY

    menus = {
        "SFH type": g._valid_sfh_types(),
        "dust law": g._valid_dust_laws(),
        "dust_emission type": g._valid_dust_emission_types(),
        "dust_attenuation type": frozenset(g._VALID_DUST_TYPES),
        "nebular type": g._valid_nebular_types(),
        "shock type": frozenset(g._VALID_SHOCK_TYPES),
        "IGM type": g._valid_igm_types(),
        "X-ray type": g._valid_xray_types(),
        "radio sf type": frozenset(SF_RADIO_MODELS),
        "radio agn type": frozenset(AGN_RADIO_MODELS),
        "dust law (agn atten)": frozenset(g._VALID_AGN_ATTEN_LAWS),
        "metallicity mode": frozenset(MET_REGISTRY.keys()),
        "group key": frozenset(k for k in g._GROUP_STRUCTURAL_KEYS if "." not in k),
        "AGN model": monolithic_agn_model_names() | set(AGN_MODELS) | {"none"},
        "foreground law": frozenset(g._VALID_FOREGROUND_LAWS),
    }
    for category in ("disc", "torus", "nlr", "blr", "feii", "atten"):
        menus[f"agn.{category} type"] = g._agn_block_types(category)
    return menus


def _every_citation_key() -> list[str]:
    """Every distinct BibTeX key referenced anywhere the reverse lookup
    sweeps: :data:`NAME_TO_BIBKEY`'s values and every value list in the
    tables :data:`_REGISTRY_NAMESPACE_TABLES` names."""
    keys: set[str] = set(NAME_TO_BIBKEY.values())
    for attr, _kind in _REGISTRY_NAMESPACE_TABLES:
        table = getattr(_assoc, attr)
        for v in table.values():
            if isinstance(v, list):
                keys.update(v)
    return sorted(keys)


class TestCitationKeyRecognition:
    """When user provides a citation key as a name, error explains the registry entry."""

    def test_dust_law_citation_key_single_component_law(self):
        """Providing 'charlot_fall2000' (citation key for power_law) should explain it."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "charlot_fall2000"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg
        # The exact remedy, not merely a substring that would pass on any
        # mention of "law" (#2429 opus review round 2 item 7).
        assert "Use law='power_law'." in error_msg

    def test_dust_law_citation_key_two_component_law_bc(self):
        """Providing 'charlot_fall2000' as law_bc on two_component should explain it."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "charlot_fall2000",
                    "law_diff": "calzetti",
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg
        # The exact remedy, not merely a substring that would pass on any
        # mention of "law_bc" (#2429 opus review round 2 item 7).
        assert "Use law_bc='power_law'." in error_msg

    def test_dust_law_typo_unchanged(self):
        """A non-citation typo like 'calzeti' should still get difflib suggestion."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzeti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Should still have the difflib suggestion
        assert "Did you mean:" in error_msg
        assert "calzetti" in error_msg

    def test_dust_emission_citation_key(self):
        """Citation key for dust_emission type should be recognized."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                dust_emission={"type": "calzetti2000"},  # Citation key for "calzetti" (dust law)
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        # calzetti2000 is citation key for calzetti which is a dust law, not dust_emission

    def test_nebular_citation_key(self):
        """A citation key given at the wrong site (nebular type) is recognized.

        'byler2017' is the citation key NEBULAR_BACKEND_CITATIONS records for
        the 'cb19_grid' and 'baked_in' nebular backends -- not itself a valid
        neb.type, so this exercises the "not valid at this site" branch
        unconditionally (#2429 opus review L4: the prior version used
        'cue_wrong', which is not a citation key at all, so the citation
        branch never ran and the assertion passed vacuously).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                neb={"type": "byler2017"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()

    def test_citation_key_not_valid_at_site(self):
        """When citation key is for a different group, explain that clearly."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                neb={"type": "charlot_fall2000"},  # Citation key for power_law, not a nebular type
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Should say it's a citation key for power_law (dust law)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg
        # The exact "not valid" sentence, not a substring that would pass on
        # any mention of "not" or "dust" (#2429 opus review round 2 item 7).
        assert "power_law (dust law), not a valid nebular type." in error_msg

    def test_sfh_type_citation_key(self):
        """Citation key for SFH type is recognized, and lists both names it cites.

        'bagpipes' is the SFH_CITATIONS key for both 'dpl' and 'delayed', so
        this doubles as the "cites several registry names, lists all of them"
        case (#2429 opus review L4: the prior version nested this assertion
        under ``if "citation key" in error_msg.lower()``, so it passed
        vacuously whether or not the citation branch ever ran).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "bagpipes"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        assert "dpl" in error_msg
        assert "delayed" in error_msg

    def test_reverse_lookup_basic(self):
        """Unit test: registry_names_for_citation_key returns correct names."""
        # charlot_fall2000 -> power_law
        names = registry_names_for_citation_key("charlot_fall2000")
        assert "power_law" in names

    def test_reverse_lookup_multiple_names(self):
        """Unit test: citation key mapping to multiple registry names."""
        # skirtor citation key should map to multiple names
        names = registry_names_for_citation_key("skirtor")
        assert len(names) > 0
        # Should include the main skirtor name
        assert "skirtor" in names

    def test_reverse_lookup_unknown_key(self):
        """Unit test: unknown citation key returns empty tuple."""
        names = registry_names_for_citation_key("unknown_key_xyz_9999")
        assert names == ()

    def test_reverse_lookup_case_insensitive(self):
        """Unit test: reverse lookup is case-insensitive."""
        names_lower = registry_names_for_citation_key("charlot_fall2000")
        names_upper = registry_names_for_citation_key("CHARLOT_FALL2000")
        assert names_lower == names_upper

    def test_self_citing_name_not_reported_as_citation_key(self):
        """M1: a name that cites only itself gives no citation note.

        'cue' cites itself (NAME_TO_BIBKEY['cue'] == 'cue'); there is no
        *other* spelling to redirect to, so calling it out as "a citation
        key for cue" would be a tautology, and (worse) would have suppressed
        the ordinary difflib fallback for nothing (#2429 opus review M1).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                xray={"type": "cue"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" not in error_msg.lower()

    def test_non_registry_catalog_token_not_treated_as_citation_key(self):
        """M2: SSP-provenance and other non-registry catalog tokens are excluded.

        'basti' is an SSP isochrone token; the alias table also spells it
        'bsti' for the common misspelling, but that table is a filename-token
        catalog, not a registry namespace any group's grammar accepts, so
        'basti' given as an SFH type must fall back to the ordinary difflib
        suggestion rather than being reported as "a citation key for bsti".
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "basti"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" not in error_msg.lower()
        assert "bsti" not in error_msg
        assert "Did you mean:" in error_msg

    def test_reverse_lookup_excludes_provenance_and_alias_tables(self):
        """M2: SSP code/isochrone/library, IMF, and photometry-convention
        tables are excluded from the reverse lookup -- their keys are
        filename tokens and misspelling aliases, never a value any group's
        grammar accepts.
        """
        assert registry_names_for_citation_key("basti") == ()
        assert registry_names_for_citation_key("chabrier2003") == ()
        assert registry_names_for_citation_key("bessell2012") == ()

    def test_reverse_lookup_still_includes_registry_namespace_tables(self):
        """Positive control for the exclusion above: a genuine registry
        namespace (AGN nlr/blr) still reverse-resolves."""
        names = registry_names_for_citation_key("buchner2024")
        assert "grahsp" in names

    def test_top_level_group_key_citation_key(self):
        """M3: the top-level 'Unknown group key' validator also recognizes
        citation keys now, not just the per-group type checks."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                charlot_fall2000={"type": "dpl"},  # not a group name; a citation key
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "Unknown group key" in error_msg
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg

    def test_agn_atten_law_citation_key(self):
        """M3: agn['atten']['law'] gets the same citation-key hint as the
        byte-identical string at dust_attenuation (previously difflib-only).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                agn={"type": "composable", "atten": {"law": "charlot_fall2000"}},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg

    def test_citation_key_with_three_valid_names_shows_parenthetical(self):
        """L5: with three or more valid names, the '(or another from: ...)'
        aside lists them -- unlike the two-name case, where the lead
        sentence already names both, and the aside would be redundant.
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                xray={"type": "lehmer2016"},  # cites simple, yang20, and lopez24
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        for name in ("simple", "yang20", "lopez24"):
            assert name in error_msg
        assert "or another from" in error_msg

    def test_citation_key_with_two_valid_names_omits_parenthetical(self):
        """L5: exactly two valid names is not "or another from" territory."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "bagpipes"},  # cites exactly 'dpl' and 'delayed'
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "or another from" not in error_msg
        # The exact remedy sentence (#2429 opus review round 2 item 7).
        assert "Use type='dpl' or type='delayed'." in error_msg

    def test_sfh_composition_site_names_composition(self):
        """H1: a non-citation typo inside the sfh 'type' list still says so.

        Pins the exact wording, unlike the scalar-site message, so a
        regression that drops the composition context (or the caller
        keyword) is caught (#2429 opus review round 2 item 7: H1 had no
        pinning assert -- the test file passed on the round-1 source even
        after this wording regressed).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": ["dpl", "xyzbad"]},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "in composition" in error_msg
        # keyword=None here (#2429 opus review round 2 item 4): no remedy
        # would tell the reader to replace the whole list with a scalar.
        assert "Use type=" not in error_msg

    def test_met_mode_error_names_the_spelling(self):
        """H2: the metallicity-mode error names the ``met={'type': ...}`` spelling.

        Pins the exact wording (#2429 opus review round 2 item 7: H2 had no
        pinning assert).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                met={"type": "tabl"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "met={'type': ...}" in error_msg

    def test_dust_emission_shows_three_suggestions(self):
        """H3: the dust_emission site shows up to three difflib suggestions.

        'dael2014' is close to three real dust_emission types; the round-1
        source silently capped this site back to the default two
        (#2429 opus review round 2 item 7: H3 had no pinning assert).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                dust_emission={"type": "dael2014"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        for name in ("dale2014", "draine_li2014", "dl14"):
            assert name in error_msg

    def test_check_dict_keys_citation_hint_has_a_space(self):
        """H4/L6: the ``_check_dict_keys`` citation hint has a leading space.

        The round-1 regression ran the closing quote of the group name
        straight into "That is a citation key" with no space; pins the
        space and the L6 remedy suppression together (#2429 opus review
        round 2 item 7).
        """
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "charlot_fall2000": 1,
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "'. That is" in error_msg
        # L6: a dict *key* name has no "Use key='...'" remedy to show.
        assert "Use key=" not in error_msg


class TestAcceptedAnywhereSweep:
    """Item 1 (round 2): a citation-key hint may only name a value some
    grammar validator accepts.

    Exhaustive over every citation key in the reverse-lookup universe times
    every routed site's own menu: no message may name a value that site's
    own validator rejects (the "valid at this site" branch is restricted to
    ``valid_names`` by construction, so this is really checking the "not
    valid here" branch never smuggles in a name from outside its own site
    either), and no message may name a value *no* validator accepts
    anywhere (the accepted-anywhere filter).
    """

    def test_no_hint_names_a_value_outside_the_accepted_anywhere_union(self):
        """Every citation key, queried with no valid_names at all (forcing
        the "not valid here" branch whenever it resolves to anything), must
        never name something outside :func:`_names_accepted_anywhere`."""
        accepted = _names_accepted_anywhere()
        checked = 0
        for key in _every_citation_key():
            hint = citation_key_hint(key, [], kind="x", keyword="type", accepted_anywhere=accepted)
            checked += 1
            if not hint:
                continue
            match = re.search(r"citation key for ([^(),.]+)", hint)
            assert match is not None, f"unexpected hint shape for {key!r}: {hint!r}"
            for name in (n.strip() for n in match.group(1).split(",")):
                assert name in accepted, (
                    f"{key!r} names {name!r}, which no grammar validator accepts: {hint!r}"
                )
        assert checked >= 50, "sweep did not iterate the expected citation-key universe"

    def test_no_hint_at_any_site_names_a_value_that_site_rejects(self):
        """Every citation key, at every routed site's own menu, must never
        show a remedy naming a value outside *that site's* accepted names."""
        accepted = _names_accepted_anywhere()
        menus = _every_routed_site_menu()
        keys = _every_citation_key()
        checked = 0
        for kind, valid_names in menus.items():
            for key in keys:
                hint = citation_key_hint(
                    key,
                    list(valid_names),
                    kind=kind,
                    keyword="type",
                    accepted_anywhere=accepted,
                )
                checked += 1
                if not hint or "Use type=" not in hint:
                    continue
                for remedy_name in re.findall(r"type='([^']+)'", hint):
                    assert remedy_name in valid_names, (
                        f"{key!r} at site {kind!r} suggests {remedy_name!r}, "
                        f"which that site rejects: {hint!r}"
                    )
        assert checked == len(menus) * len(keys)

    def test_byler2017_at_neb_names_an_accepted_selector(self):
        """The concrete "good" case: byler2017 (cited by the cb19_grid and
        baked_in nebular backends) names a selector neb.type actually
        accepts, not the internal backend spelling."""
        menus = _every_routed_site_menu()
        hint = citation_key_hint(
            "byler2017",
            list(menus["nebular type"]),
            kind="nebular type",
            keyword="type",
            accepted_anywhere=_names_accepted_anywhere(),
        )
        assert "cb19" in hint or "ssp" in hint

    def test_buchner2024_at_agn_names_grahsp(self):
        """The concrete "good" case named in the ruling: buchner2024 (the
        GRAHSP citation) names 'grahsp' at a site that accepts it."""
        menus = _every_routed_site_menu()
        hint = citation_key_hint(
            "buchner2024",
            list(menus["agn.torus type"]),
            kind="agn.torus type",
            keyword="type",
            accepted_anywhere=_names_accepted_anywhere(),
        )
        assert "grahsp" in hint
