# SPDX-License-Identifier: BSD-3-Clause
"""Tests for citation key recognition in unknown-name errors.

When a user provides a citation key (like 'charlot_fall2000') as a type or
parameter name, error messages should recognize it as a known citation key and
explain which registry entry it cites, rather than falling back to difflib
suggestions alone.
"""

import pytest

from tengri.citations.resolve import registry_names_for_citation_key
from tengri.parameters import Fixed
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


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
        # The error should use the keyword the caller used
        assert "law=" in error_msg or "law" in error_msg

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
        # Should mention the keyword used
        assert "law_bc" in error_msg or "law" in error_msg

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
        # Should say it's not valid for nebular type
        assert "not" in error_msg.lower() or "dust" in error_msg.lower()

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
