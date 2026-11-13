# SPDX-License-Identifier: BSD-3-Clause
"""The model builds its nebular backend through the registry and reads the name from it.

The grammar mode picks a backend class, the class builds the instance, and the
instance reports the ``config.backend`` string the component is configured with.
These tests pin each step against a hand-written table, so a wrong name in the
registry or on a class fails here rather than as a silent fall-through.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.components.nebular import NEBULAR_MODELS
from tengri.components.nebular._models import (
    build_nebular_backend,
    nebular_backend_class_for_mode,
    nebular_backend_classes,
)
from tengri.components.nebular._shared import NebularContinuumFallback
from tengri.components.nebular.baked_in import BakedInBackend
from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.gutkin16 import Gutkin16Backend
from tengri.components.nebular.mappings_photo import (
    MappingsPhotoAGNBackend,
    MappingsPhotoStellarBackend,
)
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.contract

_BANDS = ["des_g", "des_r"]

#: Grammar ``nebular_mode`` -> the class its build constructs. Hand-written.
_MODE_CLASS = {
    "off": BakedInBackend,
    "ssp": BakedInBackend,
    "cue": CueBackend,
    "cloudy": CloudyGridBackend,
    "cb19": CB19Backend,
    "gutkin16": Gutkin16Backend,
    "mappings": MappingsPhotoStellarBackend,
    "mappings_agn": MappingsPhotoAGNBackend,
}

#: Grammar modes whose model builds on this machine: mode, SSP fixture, config string, class.
_BUILDABLE = (
    pytest.param("ssp", "ssp_data_wne", "baked_in", BakedInBackend, id="ssp"),
    pytest.param("cue", "ssp_data_fsps", "cue", CueBackend, id="cue"),
    pytest.param("gutkin16", "ssp_data_fsps", "gutkin16", Gutkin16Backend, id="gutkin16"),
)


def _build(ssp, neb):
    """Build a photometric model with the given nebular group."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Fixed(0.0),
            "tau_bc": Fixed(0.0),
        },
        neb=neb,
        redshift=Fixed(0.1),
    )


def _nebular_component(model):
    """The nebular component of a built model's chain."""
    return next(c for c in model._build_component_chain() if isinstance(c, NebularSEDComponent))


@pytest.mark.parametrize("mode, fixture, config, cls", _BUILDABLE)
def test_built_model_names_its_backend(request, mode, fixture, config, cls):
    neb = {"type": "ssp"} if mode == "ssp" else {"type": mode, "all_params": Fixed(DEFAULT)}
    model = _build(request.getfixturevalue(fixture), neb)

    component = _nebular_component(model)
    assert component.config.backend == config
    assert component.backend_class is cls
    assert type(component.backend) is cls
    assert type(model._nebular_backend) is cls


@pytest.mark.parametrize("mode, cls", list(_MODE_CLASS.items()))
def test_every_grammar_mode_selects_its_class(mode, cls):
    assert nebular_backend_class_for_mode(mode) is cls


def test_registered_mode_entries_point_at_the_expected_classes():
    for mode, cls in _MODE_CLASS.items():
        name = "ssp" if mode == "off" else mode
        assert NEBULAR_MODELS[name].callable is cls


@pytest.mark.parametrize("mode", ["not_a_mode", "none", "shock", ""])
def test_unknown_mode_raises_config_error_naming_the_valid_modes(mode):
    with pytest.raises(ConfigError) as exc:
        nebular_backend_class_for_mode(mode)
    message = str(exc.value)
    for valid in ("ssp", "cue", "cloudy", "cb19", "gutkin16", "mappings", "mappings_agn", "off"):
        assert valid in message


def test_build_refuses_an_unknown_mode_before_reading_the_spec():
    spec = SimpleNamespace(nebular_mode="not_a_mode")
    with pytest.raises(ConfigError, match="not_a_mode"):
        build_nebular_backend(spec, None)


def test_agn_backend_names_no_component_string():
    assert MappingsPhotoAGNBackend.config_name == "mappings_agn"
    assert "mappings_agn" not in nebular_backend_classes()
    assert NEBULAR_MODELS["mappings_agn"].config_backend is None


def test_fallback_names_the_backend_it_wraps():
    primary = SimpleNamespace(config_name="cb19")
    wrapper = NebularContinuumFallback(primary, fallback_mode="warn")
    assert wrapper.config_name == primary.config_name == "cb19"


def test_fallback_name_follows_a_real_backend():
    inner = BakedInBackend(ionizing_source_warning="suppress")
    wrapper = NebularContinuumFallback(inner, fallback_mode="warn")
    assert wrapper.config_name == inner.config_name == "baked_in"
