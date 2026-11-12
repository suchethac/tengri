# SPDX-License-Identifier: BSD-3-Clause
"""One registry from config name to nebular backend class.

The class-level ``config_name`` is the only place a backend's ``config.backend``
string is written. These tests pin the derived views against the literals they
replace, and check that an unknown name is refused at construction.
"""

from __future__ import annotations

import pytest

from tengri.components.nebular import NEBULAR_MODELS
from tengri.components.nebular._models import (
    nebular_backend_classes,
    nebular_backends_with_freefree_continuum,
    nebular_photoionized_backends,
    resolve_nebular_backend,
)
from tengri.components.nebular.baked_in import BakedInBackend
from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.component import NebularSEDComponentConfig
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.gutkin16 import Gutkin16Backend
from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
from tengri.components.nebular.shock import ShockBackend
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.contract

_SEVEN_NAMES = frozenset(
    {"baked_in", "cloudy_grid", "cb19", "gutkin16", "mappings", "cue", "shock"}
)


def test_mapping_has_exactly_the_seven_config_names():
    assert set(nebular_backend_classes()) == _SEVEN_NAMES


def test_mapping_points_each_name_at_its_class():
    assert nebular_backend_classes() == {
        "baked_in": BakedInBackend,
        "cloudy_grid": CloudyGridBackend,
        "cb19": CB19Backend,
        "gutkin16": Gutkin16Backend,
        "mappings": MappingsPhotoStellarBackend,
        "cue": CueBackend,
        "shock": ShockBackend,
    }


def test_unknown_name_raises_config_error_naming_the_valid_set():
    with pytest.raises(ConfigError) as exc:
        resolve_nebular_backend("not_a_backend")
    message = str(exc.value)
    for name in sorted(_SEVEN_NAMES):
        assert name in message


def test_unknown_name_suggests_the_close_match():
    with pytest.raises(ConfigError, match=r"Did you mean: cloudy_grid\?"):
        resolve_nebular_backend("cloudy")


def test_config_refuses_unknown_name_at_construction():
    with pytest.raises(ConfigError, match="cloudy_grid"):
        NebularSEDComponentConfig(backend="cloudy")


def test_freefree_set_equals_the_literal_it_replaces():
    assert nebular_backends_with_freefree_continuum() == frozenset(
        {"cue", "cloudy_grid", "gutkin16"}
    )


def test_photoionized_set_equals_the_literal_it_replaces():
    assert nebular_photoionized_backends() == frozenset(
        {"cue", "cloudy_grid", "cb19", "gutkin16", "mappings"}
    )


def test_grammar_names_lower_to_the_expected_config_backend():
    assert {name: entry.config_backend for name, entry in NEBULAR_MODELS.items()} == {
        "none": None,
        "ssp": "baked_in",
        "cue": "cue",
        "cloudy": "cloudy_grid",
        "cb19": "cb19",
        "gutkin16": "gutkin16",
        "mappings": "mappings",
        "mappings_agn": None,
    }


def test_every_registered_config_backend_resolves():
    for name, entry in NEBULAR_MODELS.items():
        if entry.config_backend is None:
            continue
        assert resolve_nebular_backend(entry.config_backend) is not None, name
