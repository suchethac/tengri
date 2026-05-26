# SPDX-License-Identifier: BSD-3-Clause
"""Parity contract: dict-grammar validator must accept every SFH in the registry.

The validator path (``parameters.groups._translate_sfh``) and the
auto-generated builder layer (``tengri.builders.sfh``) used to maintain
two parallel lists of accepted ``sfh.type`` values — the validator's
hand-maintained ``_VALID_SFH_TYPES`` and ``SFH_REGISTRY``. PR #324
surfaced the drift footgun (three new registry entries were silently
rejected by the validator). Per ADR-0005 / ADR-0008, the registry is the
canonical source; this test pins the invariant that the validator's
accepted set is exactly ``SFH_REGISTRY.keys()`` so the divergence cannot
re-emerge.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_valid_sfh_types_mirrors_registry_keys() -> None:
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY
    from tengri.parameters.groups import _valid_sfh_types

    assert set(_valid_sfh_types()) == set(SFH_REGISTRY.keys()), (
        "groups._valid_sfh_types() drifted from SFH_REGISTRY. See ADR-0005 / "
        "ADR-0008: the registry is the canonical source; the validator must "
        "derive from it, not duplicate its keys."
    )


def test_translate_sfh_accepts_every_registered_name() -> None:
    """Every registry key should round-trip through the dict-grammar parser.

    Catches the failure mode where ``_valid_sfh_types`` itself is correct
    but a separate hand-maintained list (e.g. an old hard-coded tuple in
    ``_translate_sfh``) silently rejects a registered name.
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY
    from tengri.parameters.groups import _translate_sfh

    # Compositor names ("burst", "field") are valid but live in their own
    # composition slot — _translate_sfh accepts them via the same allowlist
    # so they're fine to include.
    for name in SFH_REGISTRY:
        result: dict = {}
        # Should not raise. We only care that the validator does not reject
        # the name; downstream resolution (parameter buckets, etc.) is
        # exercised by other tests.
        _translate_sfh({"type": name}, result)
        assert result["mean_sfh_type"] == name


def test_validator_still_rejects_unknown_names_with_suggestions() -> None:
    """The 'did you mean' UX must survive the registry-derived refactor."""
    from tengri.parameters.groups import _translate_sfh

    with pytest.raises(ValueError, match="Unknown SFH type 'continuty'"):
        _translate_sfh({"type": "continuty"}, {})  # typo for 'continuity'
