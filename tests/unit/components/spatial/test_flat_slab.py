"""Numeric smoke tests for the FlatSlab uniform-aperture profile."""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.spatial.flat_slab import FlatSlab
from tengri.protocols.component import ForwardState
from tengri.protocols.spatial import SpatialComponent


def test_flat_slab_satisfies_protocol() -> None:
    assert isinstance(FlatSlab(), SpatialComponent)


def test_flat_slab_inside_near_unity_outside_near_zero() -> None:
    x = jnp.linspace(-5, 5, 50)
    y = jnp.linspace(-5, 5, 50)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    state = state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))

    params = {"spatial_radius_kpc": jnp.float64(2.0)}
    profile = FlatSlab().apply(state, params).derived["spatial_profile_2d"]

    # Center (r=0) inside the disk
    assert profile[25, 25] > 0.9
    # Corner (r ≈ 7 kpc) well outside
    assert profile[0, 0] < 0.01
