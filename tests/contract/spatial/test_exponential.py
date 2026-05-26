# SPDX-License-Identifier: BSD-3-Clause
"""Numeric smoke tests for the Exponential disk profile."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial.exponential import Exponential
from tengri.protocols.component import ForwardState
from tengri.protocols.spatial import SpatialComponent

pytestmark = pytest.mark.contract


def test_exponential_satisfies_protocol() -> None:
    assert isinstance(Exponential(), SpatialComponent)


def test_exponential_publishes_profile_falling_with_radius() -> None:
    x = jnp.linspace(-5, 5, 20)
    y = jnp.linspace(-5, 5, 20)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    state = state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))

    params = {
        "spatial_rd_kpc": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    out = Exponential().apply(state, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (20, 20)
    assert profile[10, 10] > profile[0, 0]
