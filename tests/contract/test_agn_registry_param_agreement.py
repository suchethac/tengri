# SPDX-License-Identifier: BSD-3-Clause
"""RULING R13: the monolithic and composable AGN torus registries must agree.

Two independent registries can dispatch the identical torus physics: the
monolithic ``component_factory._REGISTRY`` (a bare ``SEDModelComponent``
subclass, e.g. ``agn=SKIRTORTorus(...)``) and the composable
``AGN_BLOCKS['torus']`` (``agn={'torus': {'type': ...}}``). Before R17
(Task 16) they silently disagreed for ``'skirtor'``: the class declared
``agn_band_frac`` for its own covering fraction while the composable block
(and six OTHER composable torus blocks) declared the identical quantity
``agn_torus_frac`` -- two spellings for one physical parameter, reachable
through two different names depending which path a caller used.
``tengri.parameters.groups.check_agn_torus_registry_agreement`` is the live
guard against that class of drift recurring; this module exercises it two
ways -- a monkeypatched synthetic disagreement (proving the raise fires and
names both sides) and a live check over the real, currently-shared strings
(green only because R17 fixed the one real disagreement it used to catch).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri.config.exceptions import ConfigError
from tengri.parameters.groups import (
    _TORUS_REGISTRY_ALLOWED_DEVIATIONS,
    check_agn_torus_registry_agreement,
)


def test_live_registries_agree():
    """Green after R17 (item 6): the real, currently-shared torus type
    strings (cat3d_wind, silva04, skirtor) no longer disagree."""
    check_agn_torus_registry_agreement()  # must not raise


def test_shared_strings_are_non_trivial():
    """Guard the guard: the live check above must not be vacuously green
    because there is nothing shared to compare."""
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS
    from tengri.forward.component_factory import _REGISTRY

    shared = set(_REGISTRY) & set(AGN_BLOCKS.get("torus", {}))
    assert shared, "no torus type strings are shared between the two registries"
    assert "skirtor" in shared, "the exact case R17 fixed must still be exercised"


def test_synthetic_disagreement_raises_config_error(monkeypatch):
    """A monkeypatched, fabricated disagreement must raise ConfigError naming
    both sides' parameter sets -- proving the raise path actually fires,
    not just that the current registries happen to agree."""
    from dataclasses import dataclass

    from tengri.components.agn._params import PARAMS as _AGN_PARAMS
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS
    from tengri.forward.component_factory import _REGISTRY
    from tengri.protocols.component import declared_prior

    real_torus_blocks = dict(AGN_BLOCKS["torus"])

    # A synthetic class declaring a parameter the real "skirtor" composable
    # block does NOT (a fabricated name no real physics uses), so the two
    # sides disagree on purpose.
    _fake_prior = declared_prior(_AGN_PARAMS, "agn_torus_frac")

    @dataclass
    class _Decl:
        name: str
        prior: object

    @dataclass(frozen=True)
    class _FakeSkirtorTorus:
        def declared_parameters(self):
            return [_Decl(name="agn_totally_fabricated_disagreement", prior=_fake_prior)]

    fake_registry = dict(_REGISTRY)
    fake_registry["skirtor"] = _FakeSkirtorTorus

    # check_agn_torus_registry_agreement does a lazy
    # `from tengri.forward.component_factory import _REGISTRY` inside the
    # function body -- patch the SOURCE module's attribute so that fresh
    # import sees the fake registry.
    import tengri.forward.component_factory as component_factory_mod

    monkeypatch.setattr(component_factory_mod, "_REGISTRY", fake_registry)

    with pytest.raises(ConfigError) as excinfo:
        check_agn_torus_registry_agreement()

    msg = str(excinfo.value)
    assert "skirtor" in msg
    assert "agn_totally_fabricated_disagreement" in msg
    # Real torus blocks untouched -- prove the patch was additive/isolated.
    assert AGN_BLOCKS["torus"] == real_torus_blocks


def test_allowed_deviations_are_documented_and_minimal():
    """Every entry in the allow-list names a real, currently-shared torus
    type -- an entry for a name that has drifted out of the shared set
    would be silently unused, exactly the "allowlist entries are guard bug
    reports" trap: it should shrink to nothing if the registries are ever
    fully reconciled, not accumulate un-noticed cruft."""
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS
    from tengri.forward.component_factory import _REGISTRY

    shared = set(_REGISTRY) & set(AGN_BLOCKS.get("torus", {}))
    for name in _TORUS_REGISTRY_ALLOWED_DEVIATIONS:
        assert name in shared, (
            f"_TORUS_REGISTRY_ALLOWED_DEVIATIONS[{name!r}] names a torus type no "
            f"longer shared between the two registries -- stale entry, remove it"
        )
