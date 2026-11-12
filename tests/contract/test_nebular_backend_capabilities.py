# SPDX-License-Identifier: BSD-3-Clause
"""The class-level capabilities of each nebular backend equal what the component computes.

Each backend class answers, without an instance, for its ``config.backend`` string:
the parameters it contributes, the ``state.derived`` keys it reads, whether the
component publishes ``sed_shock``, whether ``split_sed`` accepts it, and whether
``publish_line_fraction`` is allowed. The component still computes the same answers
from the string; this file pins the two against each other for all seven names.

The name-to-class map is written out by hand, so it is the independent side.
"""

from __future__ import annotations

import dataclasses

import pytest

from tengri.components.nebular.baked_in import BakedInBackend
from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.component import (
    NebularSEDComponent,
    NebularSEDComponentConfig,
)
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.gutkin16 import Gutkin16Backend
from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
from tengri.components.nebular.shock import ShockBackend
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.contract

_BACKEND_CLASS_BY_CONFIG = {
    "baked_in": BakedInBackend,
    "cloudy_grid": CloudyGridBackend,
    "cb19": CB19Backend,
    "gutkin16": Gutkin16Backend,
    "mappings": MappingsPhotoStellarBackend,
    "cue": CueBackend,
    "shock": ShockBackend,
}
_NAMES = sorted(_BACKEND_CLASS_BY_CONFIG)


def _component(name: str) -> NebularSEDComponent:
    """Build the component for a config string, with no backend instance."""
    return NebularSEDComponent(config=NebularSEDComponentConfig(backend=name))


def _split_sed_refused(name: str) -> bool:
    """Whether ``split_sed`` raises its refusal for this config string."""
    try:
        _component(name).split_sed(None, {}, None)
    except ValueError as exc:
        return "has no continuum + line-catalog split" in str(exc)
    except Exception:
        return False
    return False


def _line_fraction_allowed(name: str) -> bool:
    """Whether ``publish_line_fraction=True`` constructs for this config string."""
    try:
        NebularSEDComponentConfig(backend=name, publish_line_fraction=True, dig_active=False)
    except ValueError as exc:
        assert "publish_line_fraction is only available" in str(exc)
        return False
    return True


@pytest.mark.parametrize("name", _NAMES)
def test_config_name_is_the_config_string(name):
    assert _BACKEND_CLASS_BY_CONFIG[name].config_name == name


@pytest.mark.parametrize("name", _NAMES)
def test_parameters_equal_component_declaration(name):
    declared = _component(name).declared_parameters()
    own = _BACKEND_CLASS_BY_CONFIG[name].parameters()
    assert [p.name for p in own] == [p.name for p in declared]
    assert [p.prior for p in own] == [p.prior for p in declared]
    assert [declared_default(own, p.name) for p in own] == [
        declared_default(declared, p.name) for p in declared
    ]


@pytest.mark.parametrize("name", _NAMES)
def test_reads_equal_component_inputs(name):
    inputs = tuple(key.name for key in _component(name).inputs())
    assert _BACKEND_CLASS_BY_CONFIG[name].reads == inputs


@pytest.mark.parametrize("name", _NAMES)
def test_publishes_shock_equals_component_outputs(name):
    publishes = any(key.name == "sed_shock" for key in _component(name).outputs())
    assert _BACKEND_CLASS_BY_CONFIG[name].publishes_shock is publishes


@pytest.mark.parametrize("name", _NAMES)
def test_supports_split_sed_equals_refusal(name):
    supported = not _split_sed_refused(name)
    assert _BACKEND_CLASS_BY_CONFIG[name].supports_split_sed is supported


@pytest.mark.parametrize("name", _NAMES)
def test_line_fraction_policy_equals_post_init(name):
    allowed = _line_fraction_allowed(name)
    assert _BACKEND_CLASS_BY_CONFIG[name].line_fraction_policy is allowed


def test_only_cue_allows_line_fraction_and_only_shock_publishes_shock():
    allowed = [n for n in _NAMES if _BACKEND_CLASS_BY_CONFIG[n].line_fraction_policy]
    shock = [n for n in _NAMES if _BACKEND_CLASS_BY_CONFIG[n].publishes_shock]
    assert allowed == ["cue"]
    assert shock == ["shock"]


def test_shock_capabilities_are_not_dataclass_fields():
    """ShockBackend is a dataclass; its capability answers must not become fields."""
    field_names = {f.name for f in dataclasses.fields(ShockBackend)}
    assert not {"config_name", "reads", "publishes_shock", "supports_split_sed"} & field_names
