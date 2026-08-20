# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the X-ray / radio / IGM runtime registries (#355).

Mirrors the structural contracts that ``test_agn_registry.py`` and
``test_dust_emission_registry.py`` enforce on the sibling registries:

* :data:`XRAY_MODELS` / :data:`RADIO_MODELS` / :data:`IGM_MODELS` are
  populated at import time and contain at least the canonical entries.
* The validators in ``parameters/groups.py`` derive their accepted-type
  set from the registry — adding an entry in the registry surfaces in
  :func:`parse_groups` automatically.
* :func:`tengri.list_xray_models` / ``list_radio_models`` /
  ``list_igm_models`` return the same dict shape as
  :func:`tengri.list_agn_models`.
* :meth:`SEDModel.build` accepts the ``xray=``, ``radio=``, ``igm=``
  kwargs end-to-end.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.parameters.groups import (
    Fixed,
    _valid_igm_types,
    _valid_radio_types,
    _valid_xray_types,
    parse_groups,
)

pytestmark = pytest.mark.contract

_EXPECTED_KEYS = {"name", "kind", "status", "citation", "short_doc", "use"}


def _check_entry_shape(entry: dict, expected_kind: str) -> None:
    """Every list_* row carries the same dict shape — assert it once here."""
    missing = _EXPECTED_KEYS - entry.keys()
    assert not missing, f"missing keys: {missing} in {entry!r}"
    assert entry["kind"] == expected_kind
    assert isinstance(entry["name"], str) and entry["name"]
    assert entry["status"] in {"production", "experimental", "demo", "deprecated"}


class TestXRayRegistry:
    def test_canonical_entries_present(self):
        from tengri.components.xray import XRAY_MODELS

        assert "none" in XRAY_MODELS
        assert "simple" in XRAY_MODELS

    def test_validator_derived_from_registry(self):
        """Verify the grammar menu includes both function models and SEDModelComponent models.

        Issue #1120: the menu must include SEDModelComponent entries
        (xray_aird, agn_xray_corona) so they are discoverable to users.
        Without this, only the legacy function-based models (simple, none)
        would be in the menu, making the component variants silently unreachable.
        """
        from tengri.components.xray import XRAY_MODELS
        from tengri.forward.component_factory import _REGISTRY

        # Function-based models (the old path)
        function_models = frozenset(XRAY_MODELS.keys())

        # SEDModelComponent models (the new additions)
        component_models = frozenset(name for name in _REGISTRY if "xray_" in name)

        # The menu should be the union of both
        expected = function_models | component_models
        assert _valid_xray_types() == expected

    def test_list_xray_models_shape(self):
        rows = tengri.list_xray_models()
        names = {r["name"] for r in rows}
        assert {"none", "simple"} <= names
        for row in rows:
            _check_entry_shape(row, expected_kind="xray_model")

    def test_status_filter(self):
        production = tengri.list_xray_models(status="production")
        assert {"none", "simple"} <= {r["name"] for r in production}
        # A real status that no X-ray model carries is a well-formed question
        # with an empty answer. This used to be spelled `status="deprecated"`,
        # which is not a status tengri assigns anywhere — and since #1679 that
        # is refused rather than answered with a silent empty list, because an
        # empty list could not be told apart from a typo.
        assert tengri.list_xray_models(status="unvalidated") == []

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
            tengri.list_xray_models(status=typo)

    def test_parse_groups_accepts_simple(self):
        params = parse_groups(xray={"type": "simple"}, redshift=Fixed(0.1))
        assert params.xray is True

    def test_parse_groups_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown X-ray type"):
            parse_groups(xray={"type": "definitely-not-a-real-model"}, redshift=Fixed(0.1))


class TestRadioRegistry:
    def test_canonical_entries_present(self):
        from tengri.components.radio import RADIO_MODELS

        assert "none" in RADIO_MODELS
        assert "condon92" in RADIO_MODELS

    def test_validator_derived_from_registry(self):
        """Verify the grammar menu includes both function models and SEDModelComponent models.

        Issue #1120: the menu must include SEDModelComponent entries
        (radio_powerlaw, radio_dpl) so they are discoverable to users.
        Without this, only the legacy function-based models (condon92, none)
        would be in the menu, making the component variants silently unreachable.
        """
        from tengri.components.radio import RADIO_MODELS
        from tengri.forward.component_factory import _REGISTRY

        # Function-based models (the old path)
        function_models = frozenset(RADIO_MODELS.keys())

        # SEDModelComponent models (the new additions)
        component_models = frozenset(name for name in _REGISTRY if name.startswith("radio_"))

        # The menu should be the union of both
        expected = function_models | component_models
        assert _valid_radio_types() == expected

    def test_list_radio_models_shape(self):
        rows = tengri.list_radio_models()
        names = {r["name"] for r in rows}
        assert {"none", "condon92"} <= names
        for row in rows:
            _check_entry_shape(row, expected_kind="radio_model")

    def test_parse_groups_accepts_condon92(self):
        params = parse_groups(
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            redshift=Fixed(0.1),
        )
        assert params.radio is True

    def test_parse_groups_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown radio agn type"):
            parse_groups(radio={"agn": {"type": "ska-2030"}}, redshift=Fixed(0.1))


class TestIGMRegistry:
    def test_canonical_entries_present(self):
        from tengri.components.igm import IGM_MODELS

        # All four canonical variants + the 'inoue' alias for 'inoue14'
        # required for #344 backwards-compat.
        assert {"none", "madau", "inoue14", "inoue"} <= set(IGM_MODELS.keys())

    def test_validator_derived_from_registry(self):
        from tengri.components.igm import IGM_MODELS

        assert _valid_igm_types() == frozenset(IGM_MODELS.keys())

    def test_list_igm_models_shape(self):
        rows = tengri.list_igm_models()
        names = {r["name"] for r in rows}
        assert {"none", "madau", "inoue14"} <= names
        for row in rows:
            _check_entry_shape(row, expected_kind="igm_model")

    def test_parse_groups_propagates_model(self):
        # The #344 regression: igm.type must reach Parameters.igm_model.
        params = parse_groups(igm={"type": "madau"}, redshift=Fixed(0.1))
        assert params.apply_igm is True
        assert params.igm_model == "madau"

    def test_parse_groups_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown IGM type"):
            parse_groups(igm={"type": "lyman-something"}, redshift=Fixed(0.1))


class TestRegistryParityWithSFH:
    """Lock the dict shape against the existing list_sfh_models contract."""

    def test_xray_row_keys_match_sfh_row_keys(self):
        sfh_keys = set(tengri.list_sfh_models()[0]) - {"params", "param_details"}
        xray_keys = set(tengri.list_xray_models()[0])
        # X-ray rows must at minimum carry every key the SFH row does
        # (the inverse is fine — SFH adds `params` / `param_details`).
        assert sfh_keys <= xray_keys
