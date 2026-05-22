"""Tests for SpatialModelComponent base class behaviour."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import pytest

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import ForwardState


class _DummyProfile(SpatialModelComponent):
    name = "dummy_profile"
    parameter_prefix = "spatial_"

    radius = Uniform(0.1, 10.0, description="radius", units="kpc")

    reads: ClassVar[dict[str, str]] = {}
    publishes: ClassVar[dict[str, str]] = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        r = jnp.sqrt(x**2 + y**2)
        profile = jnp.exp(-r / p["radius"])
        return profile, {}


@pytest.fixture
def dummy_profile():
    return _DummyProfile()


@pytest.fixture
def state_with_grid():
    x = jnp.linspace(-5, 5, 10)
    y = jnp.linspace(-5, 5, 10)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    return state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))


def test_auto_discovers_distribution_attrs_as_free_params(dummy_profile) -> None:
    declared = dummy_profile.declared_parameters()
    assert {d.name for d in declared} == {"spatial_radius"}
    assert declared[0].units == "kpc"


def test_apply_writes_profile_to_state_derived(dummy_profile, state_with_grid) -> None:
    params = {"spatial_radius": jnp.float64(2.0)}
    out = dummy_profile.apply(state_with_grid, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (10, 10)
    assert profile[5, 5] > profile[0, 0]


def test_apply_strips_prefix_before_calling_predict(dummy_profile, state_with_grid) -> None:
    params = {"spatial_radius": jnp.float64(2.0)}
    dummy_profile.apply(state_with_grid, params)


def test_apply_raises_when_grid_missing(dummy_profile) -> None:
    state = ForwardState(wave=jnp.zeros(1))
    params = {"spatial_radius": jnp.float64(2.0)}
    with pytest.raises(KeyError, match="spatial_grid_xy_kpc"):
        dummy_profile.apply(state, params)
