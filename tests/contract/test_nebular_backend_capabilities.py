# SPDX-License-Identifier: BSD-3-Clause
"""The class-level capabilities of each nebular backend equal what the component computes.

Each backend class answers, without an instance, for its ``config.backend`` string:
the parameters it contributes, the ``state.derived`` keys it reads, whether ``split_sed``
accepts it, and whether ``publish_line_fraction`` is allowed. The component still computes
the same answers from the string; this file pins the two against each other for all six
names.

The name-to-class map is written out by hand, so it is the independent side.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import NebularBackendDefaults
from tengri.components.nebular.baked_in import BakedInBackend, BakedInNebularWarning
from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.component import (
    NebularSEDComponent,
    NebularSEDComponentConfig,
)
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.gutkin16 import Gutkin16Backend
from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend

pytestmark = pytest.mark.contract

_BACKEND_CLASS_BY_CONFIG = {
    "baked_in": BakedInBackend,
    "cloudy_grid": CloudyGridBackend,
    "cb19": CB19Backend,
    "gutkin16": Gutkin16Backend,
    "mappings": MappingsPhotoStellarBackend,
    "cue": CueBackend,
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


# Hand-written expectations per config string. The component reads these from the
# backend classes now, so comparing it with the classes would test the code against itself.
_STD = ("neb_logU", "neb_logZ_gas", "neb_fesc", "neb_fesc_lya", "neb_fdust_frac")
_CUE_EXTRA = (
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
    "gas_logn",
    "gas_logno",
    "gas_logco",
)
_PARAM_NAMES = {
    "baked_in": (),
    "cloudy_grid": _STD,
    "cb19": _STD,
    "gutkin16": _STD,
    "mappings": _STD,
    "cue": _STD + _CUE_EXTRA,
}
_READS = {
    "baked_in": (),
    "cloudy_grid": ("lnu_age", "ssp_ages_yr", "age_weights"),
    "cb19": ("lnu_age", "ssp_ages_yr"),
    "gutkin16": ("lnu_age", "ssp_ages_yr"),
    "mappings": ("lnu_age", "ssp_ages_yr"),
    "cue": ("lnu_age", "ssp_ages_yr"),
}
_ADDITIVE = {n: n != "baked_in" for n in _NAMES}
_REQUIRES_INSTANCE = {n: n != "baked_in" for n in _NAMES}
_CACHES_FILTERS = {n: n != "baked_in" for n in _NAMES}


@pytest.mark.parametrize("name", _NAMES)
def test_parameters_names_are_hand_written(name):
    declared = _component(name).declared_parameters()
    assert tuple(p.name for p in declared) == _PARAM_NAMES[name]
    assert [p.name for p in _BACKEND_CLASS_BY_CONFIG[name].parameters()] == list(
        _PARAM_NAMES[name]
    )


@pytest.mark.parametrize("name", _NAMES)
def test_reads_names_are_hand_written(name):
    inputs = tuple(key.name for key in _component(name).inputs())
    assert _BACKEND_CLASS_BY_CONFIG[name].reads == _READS[name] == inputs


@pytest.mark.parametrize("name", _NAMES)
def test_supports_split_sed_equals_refusal(name):
    supported = not _split_sed_refused(name)
    assert _BACKEND_CLASS_BY_CONFIG[name].supports_split_sed is supported


@pytest.mark.parametrize("name", _NAMES)
def test_line_fraction_policy_is_hand_written(name):
    allowed = _line_fraction_allowed(name)
    assert _BACKEND_CLASS_BY_CONFIG[name].line_fraction_policy is (name == "cue")
    assert allowed is (name == "cue")


def test_only_cue_allows_line_fraction():
    allowed = [n for n in _NAMES if _BACKEND_CLASS_BY_CONFIG[n].line_fraction_policy]
    assert allowed == ["cue"]


@pytest.mark.parametrize("name", _NAMES)
def test_additive_is_hand_written(name):
    assert _BACKEND_CLASS_BY_CONFIG[name].additive is _ADDITIVE[name]


@pytest.mark.parametrize("name", _NAMES)
def test_requires_instance_is_hand_written(name):
    assert _BACKEND_CLASS_BY_CONFIG[name].requires_instance is _REQUIRES_INSTANCE[name]


@pytest.mark.parametrize("name", _NAMES)
def test_caches_filters_is_hand_written(name):
    assert _BACKEND_CLASS_BY_CONFIG[name].caches_filters is _CACHES_FILTERS[name]


@pytest.mark.parametrize("name", [n for n in _NAMES if n != "cue"])
def test_only_cue_passes_line_catalog_keywords(name):
    """Only Cue passes line-catalog keywords; every other backend passes none."""
    cls = _BACKEND_CLASS_BY_CONFIG[name]
    assert cls.line_catalog_kwargs(full_catalog=True) == {}
    assert cls.line_catalog_kwargs(full_catalog=False) == {}


def test_cue_line_catalog_kwargs_is_the_inverse_of_full_catalog():
    assert CueBackend.line_catalog_kwargs(full_catalog=True) == {"cloudyfsps_only": False}
    assert CueBackend.line_catalog_kwargs(full_catalog=False) == {"cloudyfsps_only": True}


def test_precompute_check_constructs_baked_in_with_the_suppress_flag():
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        BakedInBackend.precompute_check(suppress_warning=True)
    assert not [w for w in quiet if issubclass(w.category, BakedInNebularWarning)]
    with pytest.warns(BakedInNebularWarning):
        BakedInBackend.precompute_check(suppress_warning=False)


@pytest.mark.parametrize("name", [n for n in _NAMES if n != "baked_in"])
def test_instance_backends_inherit_the_empty_precompute_check(name):
    """Only the baked-in backend builds at precompute, so the others define no override."""
    assert "precompute_check" not in _BACKEND_CLASS_BY_CONFIG[name].__dict__


def _state(**derived):
    return SimpleNamespace(derived=derived)


def test_default_call_kwargs_refuses_a_missing_population():
    class _Stub(NebularBackendDefaults):
        config_name = "cloudy_grid"

    with pytest.raises(ValueError, match="CLOUDY_GRID backend requires"):
        _Stub().call_kwargs(_state(ssp_ages_yr=jnp.array([1.0])), {}, {"ssp_wave": 1.0})


def test_default_call_kwargs_adds_weights_and_log_ages():
    class _Stub(NebularBackendDefaults):
        config_name = "cloudy_grid"

    call = _Stub().call_kwargs(
        _state(age_weights=jnp.array([2.0, 3.0]), ssp_ages_yr=jnp.array([1e6, 1e7])),
        {},
        {"ssp_wave": 5.0},
    )
    assert call["ssp_wave"] == 5.0
    np.testing.assert_allclose(call["ssp_weights"], [2.0, 3.0])
    np.testing.assert_allclose(call["ssp_log_ages_yr"], [6.0, 7.0])


def test_cue_call_kwargs_prefers_the_ssp_population_and_forwards_overrides():
    state = _state(
        age_weights=jnp.array([1.0]), ssp_ages_yr=jnp.array([1e6]), log_nion=jnp.array(9.0)
    )
    call = CueBackend.call_kwargs(None, state, {"gas_logn": 2.0}, {"ssp_wave": 4.0})
    assert call["ssp_wave"] == 4.0
    assert call["gas_logn"] == 2.0
    np.testing.assert_allclose(call["ssp_weights"], [1.0])
    np.testing.assert_allclose(call["ssp_log_ages_yr"], [6.0])
    assert "gas_logqion" not in call


def test_cue_call_kwargs_uses_the_qion_shortcut_when_no_population_is_published():
    call = CueBackend.call_kwargs(None, _state(log_nion=jnp.array(-1.0)), {}, {"ssp_wave": 4.0})
    np.testing.assert_allclose(call["gas_logqion"], 0.0)
    assert call["ssp_weights"] is None and call["ssp_log_ages_yr"] is None


def test_cue_extra_derived_publishes_neb_log_no_from_effective_log_no():
    class _Fake:
        def effective_log_no(self, **kwargs):
            return kwargs

    out = CueBackend.extra_derived(_Fake(), {"neb_logZ_gas": 1.5}, {"gas_logno": 0.25})
    assert set(out) == {"neb_log_no"}
    assert out["neb_log_no"]["neb_logZ_gas"] == 1.5
    assert float(out["neb_log_no"]["gas_logno"]) == 0.25
    assert "gas_logz" not in out["neb_log_no"]


def test_default_extra_derived_publishes_nothing():
    class _Stub(NebularBackendDefaults):
        config_name = "gutkin16"

    assert _Stub().extra_derived({}, {}) == {}


def test_default_split_nebular_has_a_zero_continuum():
    class _Stub(NebularBackendDefaults):
        config_name = "gutkin16"

        def predict_nebular_line_luminosities(self, **kwargs):
            return jnp.array([1216.0]), jnp.array([3.0])

    cont, line_wave, line_lum = _Stub().split_nebular({"ssp_wave": jnp.ones(4)}, None)
    np.testing.assert_array_equal(cont, np.zeros(4))
    np.testing.assert_array_equal(line_wave, [1216.0])
    np.testing.assert_array_equal(line_lum, [3.0])


def test_every_backend_a_build_can_hand_the_component_answers_its_method_set():
    """The component calls these on the backend instance, so each constructible class has them.

    The list is written out by hand: the six registered classes, the MAPPINGS AGN backend
    (built for ``neb={'type': 'mappings_agn'}`` under the ``mappings`` config name) and the
    continuum fallback, which forwards attribute lookups to the backend it wraps.
    """
    from tengri.components.nebular._shared import NebularContinuumFallback
    from tengri.components.nebular.baked_in import BakedInBackend
    from tengri.components.nebular.cloudy_cb19 import CB19Backend
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.components.nebular.cue import CueBackend
    from tengri.components.nebular.gutkin16 import Gutkin16Backend
    from tengri.components.nebular.mappings_photo import (
        MappingsPhotoAGNBackend,
        MappingsPhotoStellarBackend,
    )

    methods = ("call_kwargs", "split_nebular", "extra_derived", "line_catalog_kwargs")
    classes = (
        BakedInBackend,
        CloudyGridBackend,
        CueBackend,
        CB19Backend,
        Gutkin16Backend,
        MappingsPhotoStellarBackend,
        MappingsPhotoAGNBackend,
    )
    for cls in classes:
        for name in methods:
            assert callable(getattr(cls, name, None)), f"{cls.__name__}.{name}"
    assert "__getattr__" in vars(NebularContinuumFallback)


def test_component_parameter_lists_do_not_replace_the_registry_records():
    """The registry record of a nebular name is the one ``_params.py`` owns.

    The component's own list restates ``neb_logU`` and its siblings with other
    priors and no ``free_prior``. Exported from
    ``_params.py`` they became the registry record of each name, so
    ``all_params: FREE`` left ``neb_fesc`` pinned and the registry default of
    ``shock_log_lhalpha`` read 40.0 instead of 41.0.
    """
    from tengri.components.nebular._component_params import NEBULAR_STANDARD_PARAMS
    from tengri.parameters.registry import registry

    records = registry()
    restated = [d.name for d in NEBULAR_STANDARD_PARAMS]
    owners = {name: records[name].group for name in restated}
    assert set(owners.values()) == {"PARAMS"}, owners
    assert records["neb_fesc"].free_prior is not None
    assert records["shock_log_lhalpha"].prior.value == 41.0


def test_the_defaults_declare_but_do_not_define_the_line_catalog_method():
    """A backend publishes lines exactly when it defines the line-catalog method.

    The forward model reads ``hasattr(backend, "predict_nebular_line_luminosities")``
    as that capability, so the shared defaults must not supply the method: with
    one, ``BakedInBackend`` would read as publishing a line catalog.
    """
    from tengri.components.nebular._shared import NebularBackendDefaults
    from tengri.components.nebular.baked_in import BakedInBackend
    from tengri.components.nebular.cue import CueBackend

    assert not hasattr(NebularBackendDefaults, "predict_nebular_line_luminosities")
    assert not hasattr(BakedInBackend, "predict_nebular_line_luminosities")
    assert hasattr(CueBackend, "predict_nebular_line_luminosities")
