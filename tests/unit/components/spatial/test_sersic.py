"""Numeric smoke tests for the Sersic spatial profile."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.spatial.sersic import Sersic
from tengri.protocols.component import ForwardState
from tengri.protocols.spatial import SpatialComponent


@pytest.fixture
def sersic():
    return Sersic()


@pytest.fixture
def state_with_grid():
    x = jnp.linspace(-5, 5, 20)
    y = jnp.linspace(-5, 5, 20)
    xx, yy = jnp.meshgrid(x, y)
    state = ForwardState(wave=jnp.zeros(1))
    return state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=(xx, yy)))


def test_sersic_satisfies_spatial_component_protocol(sersic) -> None:
    assert isinstance(sersic, SpatialComponent)


def test_sersic_publishes_profile_with_central_peak(sersic, state_with_grid) -> None:
    params = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_n": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    out = sersic.apply(state_with_grid, params)
    profile = out.derived["spatial_profile_2d"]
    assert profile.shape == (20, 20)
    assert profile[10, 10] > profile[0, 0]


def test_sersic_n_4_has_higher_central_peak_than_n_1(sersic, state_with_grid) -> None:
    """n=4 (de Vaucouleurs) has a much sharper central peak than n=1 (exponential).

    The Sersic profile is normalised so that ``r_e`` encloses half the
    light; the surface brightness at r=0 is ``exp(b_n)``, which is
    ~2147 for n=4 vs ~5.4 for n=1 — a factor of ~400 difference.
    """
    base = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    p_dv = {**base, "spatial_n": jnp.float64(4.0)}
    p_exp = {**base, "spatial_n": jnp.float64(1.0)}

    profile_dv = sersic.apply(state_with_grid, p_dv).derived["spatial_profile_2d"]
    profile_exp = sersic.apply(state_with_grid, p_exp).derived["spatial_profile_2d"]

    # Near the centre (pixel 10 of 20 on a [-5, 5] grid → r ≈ 0.37 kpc),
    # the de Vaucouleurs profile sits well above the exponential.
    assert profile_dv[10, 10] > profile_exp[10, 10]


def test_sersic_circular_axis_ratio_gives_axisymmetric_profile(sersic, state_with_grid) -> None:
    params = {
        "spatial_re_kpc": jnp.float64(1.0),
        "spatial_n": jnp.float64(1.0),
        "spatial_axis_ratio": jnp.float64(1.0),
        "spatial_pa_deg": jnp.float64(0.0),
    }
    profile = sersic.apply(state_with_grid, params).derived["spatial_profile_2d"]
    assert jnp.allclose(profile, profile[::-1, ::-1], rtol=1e-10)
