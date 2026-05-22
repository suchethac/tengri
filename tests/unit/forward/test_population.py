"""Tests for the Population dataclass (forward-model architecture §5)."""

from __future__ import annotations

import dataclasses

import pytest

from tengri.forward.population import Population


class _DummySubModel:
    name = "dummy"

    def declared_parameters(self):
        return []

    def run(self, state, params):
        return state


def test_population_is_frozen() -> None:
    pop = Population(name="default", sed=_DummySubModel(), spatial=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pop.name = "other"  # type: ignore[misc]


def test_population_holds_sed_and_optional_spatial() -> None:
    sub = _DummySubModel()
    pop = Population(name="bulge", sed=sub, spatial=None)
    assert pop.name == "bulge"
    assert pop.sed is sub
    assert pop.spatial is None


def test_population_spatial_defaults_to_none() -> None:
    pop = Population(name="x", sed=_DummySubModel())
    assert pop.spatial is None


def test_population_name_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Population(name="", sed=_DummySubModel())
