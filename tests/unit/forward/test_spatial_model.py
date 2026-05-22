"""Tests for SpatialModel and SpatialSEDModel sub-models."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial.exponential import Exponential
from tengri.components.spatial.sersic import Sersic
from tengri.forward.spatial_model import SpatialModel, SpatialSEDModel
from tengri.protocols import SubModel
from tengri.protocols.component import ForwardState


@pytest.fixture
def grid_state():
    x = jnp.linspace(-5, 5, 16)
    y = jnp.linspace(-5, 5, 16)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    return state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))


def test_spatial_model_satisfies_submodel(grid_state) -> None:
    model = SpatialModel(components=[Sersic()])
    assert isinstance(model, SubModel)
    assert model.name == "spatial"


def test_spatial_model_declares_aggregated_params() -> None:
    model = SpatialModel(components=[Sersic()])
    declared = model.declared_parameters()
    names = {d.name for d in declared}
    assert names == {
        "spatial_re_kpc",
        "spatial_n",
        "spatial_axis_ratio",
        "spatial_pa_deg",
    }


def test_spatial_model_run_writes_profile(grid_state) -> None:
    model = SpatialModel(components=[Sersic()])
    params = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_n": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    out = model.run(grid_state, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (16, 16)
    assert profile[8, 8] > profile[0, 0]


def test_spatial_model_threads_through_components(grid_state) -> None:
    """Last component wins on spatial_profile_2d (current single-publisher rule).

    Sersic and Exponential both publish ``spatial_profile_2d``; running
    them in sequence means the second one's output is what state.derived
    carries at the end. (Multi-component composition that ADDS profiles
    is the BulgeDisk path — a separate later component.)
    """
    sersic_only = SpatialModel(components=[Sersic()]).run(
        grid_state,
        {
            "spatial_re_kpc": jnp.float64(1.0),
            "spatial_n": jnp.float64(4.0),
            "spatial_axis_ratio": jnp.float64(1.0),
            "spatial_pa_deg": jnp.float64(0.0),
            "spatial_rd_kpc": jnp.float64(1.0),
        },
    )
    chained = SpatialModel(components=[Sersic(), Exponential()]).run(
        grid_state,
        {
            "spatial_re_kpc": jnp.float64(1.0),
            "spatial_n": jnp.float64(4.0),
            "spatial_axis_ratio": jnp.float64(1.0),
            "spatial_pa_deg": jnp.float64(0.0),
            "spatial_rd_kpc": jnp.float64(1.0),
        },
    )
    # Last (Exponential) overwrites the profile
    chained_profile = chained.derived["spatial_profile_2d"]
    assert not jnp.allclose(chained_profile, sersic_only.derived["spatial_profile_2d"])


def test_spatial_sed_model_satisfies_submodel(grid_state) -> None:
    """SpatialSEDModel composes any two SubModel-shaped objects."""
    sed_like = SpatialModel(components=[Sersic()])  # double-duty: also a SubModel
    spatial = SpatialModel(components=[Exponential()])
    model = SpatialSEDModel(sed=sed_like, spatial=spatial)
    assert isinstance(model, SubModel)
    assert model.name == "spatial_sed"


def test_spatial_sed_model_declared_parameters_is_union() -> None:
    sed_like = SpatialModel(components=[Sersic()])
    spatial = SpatialModel(components=[Exponential()])
    model = SpatialSEDModel(sed=sed_like, spatial=spatial)
    declared = model.declared_parameters()
    names = [d.name for d in declared]
    # Sersic params (from sed_like) come first, then Exponential (from spatial)
    assert names[:4] == [
        "spatial_re_kpc",
        "spatial_n",
        "spatial_axis_ratio",
        "spatial_pa_deg",
    ]
    assert "spatial_rd_kpc" in names
