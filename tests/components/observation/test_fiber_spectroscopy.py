"""Tests for FiberSpectroscopyObservation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest

from tengri.observation.fiber_spectroscopy import FiberSpectroscopyObservation
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.bounds


@dataclass(frozen=True)
class _StubObservation:
    """Minimal observation that returns a fixed dict for testing."""

    fixed_output: dict[str, Any]
    name: str = "stub"

    def predict(self, state, params):
        return dict(self.fixed_output)


@pytest.fixture
def state_with_profile():
    """ForwardState with a small Gaussian-like spatial profile centred on origin."""
    axis = jnp.linspace(-2.0, 2.0, 32)
    xx, yy = jnp.meshgrid(axis, axis)
    profile = jnp.exp(-jnp.sqrt(xx**2 + yy**2) / 1.0)  # exp disk, scale 1 kpc
    state = ForwardState(wave=jnp.zeros(1))
    return state.with_(
        derived=state.derived.with_(
            spatial_grid_xy_kpc=(xx, yy),
            spatial_profile_2d=profile,
        )
    )


@pytest.fixture
def state_no_profile():
    return ForwardState(wave=jnp.zeros(1))


def test_passes_through_when_no_spec_key(state_with_profile) -> None:
    """If the wrapped observation has no spec_fnu, no scaling happens."""
    base = _StubObservation(fixed_output={"phot_fnu": jnp.array([1.0, 2.0, 3.0])})
    obs = FiberSpectroscopyObservation(observation=base, fiber_radius_arcsec=1.0)
    pred = obs.predict(state_with_profile, {"redshift": jnp.float64(0.05)})
    assert "phot_fnu" in pred
    assert "spec_fnu" not in pred


def test_passes_through_when_no_spatial_profile(state_no_profile) -> None:
    """If the state has no spatial profile, no scaling happens (flat-slab fallback)."""
    spec = jnp.array([10.0, 20.0, 30.0])
    base = _StubObservation(fixed_output={"spec_fnu": spec})
    obs = FiberSpectroscopyObservation(observation=base, fiber_radius_arcsec=1.0)
    pred = obs.predict(state_no_profile, {"redshift": jnp.float64(0.05)})
    assert jnp.allclose(pred["spec_fnu"], spec)


def test_scales_spec_by_aperture_fraction(state_with_profile) -> None:
    """Small aperture captures less than the full spectrum."""
    spec = jnp.ones(10)
    base = _StubObservation(fixed_output={"spec_fnu": spec})
    obs = FiberSpectroscopyObservation(
        observation=base,
        fiber_radius_arcsec=1.0,  # ~1 kpc at z=0.05
    )
    pred = obs.predict(state_with_profile, {"redshift": jnp.float64(0.05)})
    # All entries are scaled by the same scalar
    assert pred["spec_fnu"].shape == spec.shape
    scaling = float(pred["spec_fnu"][0] / spec[0])
    assert 0.0 < scaling < 1.0


def test_larger_aperture_captures_more(state_with_profile) -> None:
    spec = jnp.ones(5)
    base = _StubObservation(fixed_output={"spec_fnu": spec})

    obs_small = FiberSpectroscopyObservation(observation=base, fiber_radius_arcsec=0.5)
    obs_large = FiberSpectroscopyObservation(observation=base, fiber_radius_arcsec=3.0)

    z = {"redshift": jnp.float64(0.05)}
    small_pred = obs_small.predict(state_with_profile, z)
    large_pred = obs_large.predict(state_with_profile, z)

    assert float(large_pred["spec_fnu"][0]) > float(small_pred["spec_fnu"][0])


def test_photometry_unchanged_when_present(state_with_profile) -> None:
    """phot_fnu (total-flux convention) is not scaled by the fiber aperture."""
    phot = jnp.array([100.0, 200.0])
    spec = jnp.ones(3)
    base = _StubObservation(fixed_output={"phot_fnu": phot, "spec_fnu": spec})
    obs = FiberSpectroscopyObservation(observation=base, fiber_radius_arcsec=1.0)
    pred = obs.predict(state_with_profile, {"redshift": jnp.float64(0.05)})
    assert jnp.allclose(pred["phot_fnu"], phot)
    # Spec is scaled
    assert not jnp.allclose(pred["spec_fnu"], spec)
