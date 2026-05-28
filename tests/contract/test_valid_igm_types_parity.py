# SPDX-License-Identifier: BSD-3-Clause
"""Parity contract: IGM validator + dispatch + builder share one source.

Sister to ``test_valid_sfh_types_parity.py`` and
``test_valid_dust_types_parity.py``. Before this refactor the dict-grammar
validator accepted ``"inoue14"`` while the SEDModel ``_init_igm`` dispatch
only accepted ``"inoue"`` — same drift footgun, different domain. The
registry (`IGM_TRANSMISSION_MODELS`) is now the canonical source, with a
small `_IGM_ALIASES` map for back-compat; both surfaces consume it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_valid_igm_types_covers_registry_and_aliases_and_none() -> None:
    from tengri.components.igm.igm import _IGM_ALIASES, IGM_TRANSMISSION_MODELS
    from tengri.parameters.groups import _valid_igm_types

    expected = set(IGM_TRANSMISSION_MODELS.keys()) | set(_IGM_ALIASES.keys()) | {"none"}
    assert set(_valid_igm_types()) == expected


def test_translate_igm_accepts_canonical_and_alias_names() -> None:
    """Both ``inoue14`` (canonical) and ``inoue`` (alias) must validate."""
    from tengri.parameters.groups import _translate_igm

    for name in ("inoue14", "inoue", "madau", "none"):
        result: dict = {}
        _translate_igm({"type": name}, result)
        assert "apply_igm" in result


def test_translate_igm_still_rejects_unknown_with_suggestion() -> None:
    from tengri.parameters.groups import _translate_igm

    with pytest.raises(ValueError, match="Unknown IGM type 'inou'"):
        _translate_igm({"type": "inou"}, {})  # typo


def test_resolve_igm_model_canonical_and_alias_return_same_function() -> None:
    """``"inoue"`` is a back-compat alias for ``"inoue14"`` — both must
    resolve to the same Inoue+2014 transmission function."""
    from tengri.components.igm import resolve_igm_model

    assert resolve_igm_model("inoue14") is resolve_igm_model("inoue")


def test_resolve_igm_model_rejects_unknown_name() -> None:
    from tengri.components.igm import resolve_igm_model

    with pytest.raises(ValueError, match="Unknown IGM model 'foo'"):
        resolve_igm_model("foo")


def test_igm_builder_factories_skip_aliases() -> None:
    """``builders.igm`` only exposes canonical model names plus ``none``.

    ``inoue`` is a validator-side alias and intentionally does not get a
    separate factory in the user-facing namespace.
    """
    from tengri import builders
    from tengri.components.igm import IGM_TRANSMISSION_MODELS

    expected = set(IGM_TRANSMISSION_MODELS.keys()) | {"none"}
    assert set(builders.igm.available()) == expected
