# SPDX-License-Identifier: BSD-3-Clause
"""The :class:`SubModel` ``name`` is a stable identifier, not a constructor knob.

``SubModel`` documents ``name`` as a "Stable identifier for diagnostics" with
fixed examples (``"sed"``, ``"spatial"``, ``"spatial_sed"``). Three of the four
implementations enforced that by refusing ``name=``; two of those three still
*declared* it as an ordinary dataclass field, and the fourth accepted it and let
a caller overwrite the identifier outright.

The declaration is what leaks. ``@dataclass`` builds ``__init__`` with an
internal ``_set_new_attribute`` that refuses to overwrite a hand-written
``__init__``, so a hand-written constructor silently wins while
``dataclasses.fields()`` goes on reporting ``init=True``. Nothing reconciles the
two, and ``PopulationSEDModel``'s docstring — which *is* the published API
reference, since ``docs/api/*.rst`` are autodoc stubs — listed ``name`` under
``Parameters`` with a default, i.e. told users to pass the one kwarg that raises.

These tests pin the rule over a discovered census rather than the three classes
that happened to be wrong, so a fourth implementation cannot reintroduce it.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import pytest

import tengri
from tengri.protocols.submodel import SubModel

pytestmark = pytest.mark.contract


def _submodel_classes() -> list[type]:
    """Every concrete class in ``tengri`` shaped like a :class:`SubModel`."""
    for mod in pkgutil.walk_packages(tengri.__path__, prefix="tengri."):
        try:
            importlib.import_module(mod.name)
        except Exception:  # optional deps, data-gated backends
            continue

    import sys

    found: dict[str, type] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("tengri") or module is None:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not (getattr(obj, "__module__", "") or "").startswith("tengri"):
                continue
            if getattr(obj, "_is_protocol", False):
                continue
            if not all(hasattr(obj, a) for a in ("name", "run", "declared_parameters")):
                continue
            found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return [found[k] for k in sorted(found)]


def _ids(classes: list[type]) -> list[str]:
    return [c.__qualname__ for c in classes]


SUBMODELS = _submodel_classes()


def test_census_is_not_empty() -> None:
    """A census that silently collapses to zero would pass every test below."""
    assert len(SUBMODELS) >= 4, (
        f"expected at least the four known SubModels, found {_ids(SUBMODELS)}. "
        "A shrinking census makes the assertions below vacuous."
    )
    assert "SEDModel" in _ids(SUBMODELS)


@pytest.mark.parametrize("cls", SUBMODELS, ids=_ids(SUBMODELS))
def test_name_is_not_a_constructor_parameter(cls: type) -> None:
    """No SubModel accepts ``name=``; the protocol calls the identifier stable."""
    params = inspect.signature(cls.__init__).parameters
    assert "name" not in params, (
        f"{cls.__qualname__}.__init__ accepts name=, so a caller can overwrite the "
        f"identifier SubModel promises is stable. Declare it "
        f"`field(default=..., init=False)`."
    )


@pytest.mark.parametrize("cls", SUBMODELS, ids=_ids(SUBMODELS))
def test_name_field_declaration_matches_the_constructor(cls: type) -> None:
    """``dataclasses.fields()`` must not advertise a kwarg ``__init__`` refuses.

    This is the assertion that would have caught the original defect: the
    constructors were already right, and only the declaration lied.
    """
    if not dataclasses.is_dataclass(cls):
        pytest.skip(f"{cls.__qualname__} is not a dataclass")
    field = next((f for f in dataclasses.fields(cls) if f.name == "name"), None)
    if field is None:
        pytest.skip(f"{cls.__qualname__} has no `name` dataclass field")
    accepted = set(inspect.signature(cls.__init__).parameters)
    assert field.init is False or "name" in accepted, (
        f"{cls.__qualname__} declares `name` with init=True but its __init__ does "
        f"not accept it. dataclasses.fields() and the class docstring both "
        f"advertise a parameter that raises TypeError."
    )


@pytest.mark.parametrize("cls", SUBMODELS, ids=_ids(SUBMODELS))
def test_name_is_documented_as_an_attribute_not_a_parameter(cls: type) -> None:
    """A docstring ``Parameters`` entry is an instruction to pass that kwarg.

    Docstrings in ``src/`` are the published API reference, so listing a
    read-only identifier there hands the user a call that raises.
    """
    doc = cls.__doc__ or ""
    if "Parameters" not in doc:
        pytest.skip(f"{cls.__qualname__} has no Parameters section")
    section = doc.split("Parameters", 1)[1]
    for terminator in ("Attributes", "Returns", "Raises", "Notes", "Examples"):
        section = section.split(terminator, 1)[0]
    assert "name :" not in section, (
        f"{cls.__qualname__} documents `name` under Parameters, but its "
        f"__init__ refuses it. Move it to an Attributes section."
    )


@pytest.mark.parametrize("cls", SUBMODELS, ids=_ids(SUBMODELS))
def test_name_still_resolves_and_is_non_empty(cls: type) -> None:
    """Making the field ``init=False`` must not leave the attribute unset.

    Instances read ``name`` off the class attribute that ``@dataclass`` leaves
    behind for a plain default, so the explicit ``object.__setattr__`` in the
    hand-written constructors was redundant. This pins that it stays true.
    """
    value = getattr(cls, "name", None)
    assert isinstance(value, str) and value, (
        f"{cls.__qualname__}.name is {value!r}; SubModel requires a stable string identifier."
    )


def test_the_four_known_identifiers_are_unchanged() -> None:
    """The values are part of the diagnostics contract, so pin them explicitly."""
    from tengri.forward.population_sed_model import PopulationSEDModel
    from tengri.forward.sed_model import SEDModel
    from tengri.forward.spatial_model import SpatialModel, SpatialSEDModel

    assert SEDModel.name == "sed"
    assert SpatialModel.name == "spatial"
    assert SpatialSEDModel.name == "spatial_sed"
    assert PopulationSEDModel.name == "population_sed_model"


def test_instances_report_the_identifier_without_an_explicit_setattr() -> None:
    """Construct the two hand-written-``__init__`` cases and read ``name`` back."""
    import jax.numpy as jnp

    from tengri.forward.population_sed_model import PopulationSEDModel
    from tengri.forward.spatial_model import SpatialModel, SpatialSEDModel

    grid = (jnp.zeros((4, 4)), jnp.zeros((4, 4)))
    assert SpatialModel(components=(), grid_kpc=grid).name == "spatial"
    assert SpatialSEDModel(sed=None, spatial=None).name == "spatial_sed"
    population = PopulationSEDModel(sed=None, galaxies=[{"flux_obs": [1.0], "noise": [0.1]}])
    assert population.name == "population_sed_model"


def test_isinstance_submodel_still_holds() -> None:
    """``init=False`` must not disturb the runtime-checkable Protocol match."""
    import jax.numpy as jnp

    from tengri.forward.spatial_model import SpatialModel

    grid = (jnp.zeros((4, 4)), jnp.zeros((4, 4)))
    assert isinstance(SpatialModel(components=(), grid_kpc=grid), SubModel)
