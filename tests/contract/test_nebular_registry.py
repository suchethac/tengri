# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the nebular runtime registry (closes #331 — nebular).

Mirrors :mod:`tests.contract.test_xray_radio_igm_registry`. Locks the
canonical entries, validator-derived-from-registry property, dict-shape
parity with the sibling listers, and the round-trip through
:func:`parse_groups`.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.parameters.groups import (
    Fixed,
    _valid_nebular_types,
    parse_groups,
)

_EXPECTED_KEYS = {"name", "kind", "status", "citation", "short_doc", "use"}


class TestNebularRegistry:
    def test_canonical_entries_present(self):
        from tengri.components.nebular import NEBULAR_MODELS

        assert {"none", "ssp", "cue", "cloudy", "cb19"} <= set(NEBULAR_MODELS.keys())

    def test_validator_derived_from_registry(self):
        from tengri.components.nebular import NEBULAR_MODELS

        assert _valid_nebular_types() == frozenset(NEBULAR_MODELS.keys())

    def test_list_nebular_backends_shape(self):
        rows = tengri.list_nebular_backends()
        names = {r["name"] for r in rows}
        assert {"none", "ssp", "cue", "cloudy", "cb19"} <= names
        for row in rows:
            missing = _EXPECTED_KEYS - row.keys()
            assert not missing, f"missing keys: {missing} in {row!r}"
            assert row["kind"] == "nebular_backend"
            assert row["status"] in {"production", "experimental", "demo", "deprecated"}

    def test_status_filter(self):
        # All current entries are production, so a real status that none of them
        # carries returns nothing. This used to ask for `status="deprecated"`,
        # which is not a status tengri assigns anywhere — since #1679 that is
        # refused rather than answered with an empty list, because an empty list
        # could not be told apart from a typo.
        production = tengri.list_nebular_backends(status="production")
        assert {"none", "ssp", "cue", "cloudy", "cb19"} <= {r["name"] for r in production}
        assert tengri.list_nebular_backends(status="unvalidated") == []

    def test_status_filter_refuses_a_status_no_menu_uses(self):
        """A typo must not read as "there are none" (#1679).

        The example used to be ``status="deprecated"``, chosen because tengri
        assigned it nowhere. It stopped being unassigned the moment a menu row
        got that status, and this test then failed for a reason that had
        nothing to do with the rule it pins. Use a misspelling, which cannot
        become real, and assert that premise instead of assuming it.
        """
        from tengri.registry import _menu_vocabulary

        typo = "producton"
        assert typo not in set(_menu_vocabulary("status")), (
            f"{typo!r} has become a real status; pick another misspelling."
        )
        with pytest.raises(ValueError, match="is not a status any menu uses"):
            tengri.list_nebular_backends(status=typo)

    def test_parse_groups_accepts_standalone_variants(self):
        # ``cloudy`` requires an external grid path (see
        # ``parameters.Parameters._raise_missing_grid_path``) and so
        # can't be exercised in a pure validator-round-trip test
        # without fixturing the grid HDF5. The validator-shape
        # contract is covered by ``test_validator_derived_from_registry``.
        for variant in ("none", "ssp", "cue", "cb19"):
            params = parse_groups(neb={"type": variant}, redshift=Fixed(0.1))
            assert params is not None, variant

    def test_parse_groups_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown nebular type"):
            parse_groups(neb={"type": "not-a-real-backend"}, redshift=Fixed(0.1))

    def test_parse_groups_suggests_close_match(self):
        """Suggestion machinery still works against the derived set."""
        with pytest.raises(ValueError, match="Did you mean"):
            parse_groups(neb={"type": "cue1"}, redshift=Fixed(0.1))


class TestNebularRegistryParity:
    """The five list_*_models / list_nebular_backends listers must all
    share the same row shape so consumers can iterate them uniformly."""

    def test_row_shape_matches_xray_listers(self):
        # Locked against XRAY_MODELS (added in #355) since that's the
        # canonical registry-derived lister shape.
        xray_keys = set(tengri.list_xray_models()[0])
        neb_keys = set(tengri.list_nebular_backends()[0])
        assert xray_keys == neb_keys
