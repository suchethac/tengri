# SPDX-License-Identifier: BSD-3-Clause
"""Parity contract: dict-grammar validator must accept every SFH in the registry.

The validator path (``parameters.groups._translate_sfh``) and the
auto-generated builder layer (``tengri.builders.sfh``) used to maintain
two parallel lists of accepted ``sfh.type`` values — the validator's
hand-maintained ``_VALID_SFH_TYPES`` and ``SFH_REGISTRY``. PR #324
surfaced the drift footgun (three new registry entries were silently
rejected by the validator). Per ADR-0005 / ADR-0008, the registry is the
canonical source; this test pins the invariant that the validator's
accepted set is exactly ``SFH_REGISTRY.keys()`` *minus* the explicitly
not-yet-validated names (``UNVALIDATED_SFH_TYPES``), so advertised types
always forward-model and the drift cannot re-emerge.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_valid_sfh_types_mirrors_registry_minus_unvalidated() -> None:
    from tengri.components.stellar.sfh.registry import (
        SFH_REGISTRY,
        UNVALIDATED_SFH_TYPES,
    )
    from tengri.parameters.groups import _valid_sfh_types

    assert set(_valid_sfh_types()) == set(SFH_REGISTRY.keys()) - set(UNVALIDATED_SFH_TYPES), (
        "groups._valid_sfh_types() must be SFH_REGISTRY.keys() minus "
        "UNVALIDATED_SFH_TYPES. See ADR-0005 / ADR-0008: the registry is the "
        "canonical catalog; the validator advertises only forward-validated types."
    )


def test_translate_sfh_accepts_every_validated_name_and_gates_the_rest() -> None:
    """Every validated registry key round-trips; unvalidated names raise clearly.

    Catches both the old drift footgun (a registered name silently rejected)
    and the advertised-but-unusable footgun (a name accepted by the grammar
    that then raises NotImplementedError at predict time).
    """
    from tengri.components.stellar.sfh.registry import (
        SFH_REGISTRY,
        UNVALIDATED_SFH_TYPES,
    )
    from tengri.parameters.groups import _translate_sfh

    for name in SFH_REGISTRY:
        if name in UNVALIDATED_SFH_TYPES:
            with pytest.raises(ValueError, match="not yet validated"):
                _translate_sfh({"type": name}, {})
            continue
        result: dict = {}
        _translate_sfh({"type": name}, result)
        assert result["mean_sfh_type"] == name


def test_validator_still_rejects_unknown_names_with_suggestions() -> None:
    """The 'did you mean' UX must survive the registry-derived refactor."""
    from tengri.parameters.groups import _translate_sfh

    with pytest.raises(ValueError, match="Unknown SFH type 'continuty'"):
        _translate_sfh({"type": "continuty"}, {})  # typo for 'continuity'
