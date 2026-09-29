# SPDX-License-Identifier: BSD-3-Clause
"""jit/grad-finiteness test for FiberSpectroscopyObservation.

Redshift is traced inside every fitter (jax.jit / jax.grad over the
posterior), so FiberSpectroscopyObservation.predict must never
concretize it. This file is kept separate from
test_fiber_spectroscopy.py, whose file-level `pytestmark` is
`bounds`: stacking a `gradient`-marked test under a `bounds`
`pytestmark` would make `pytest -m bounds` silently select a
jit/grad test, violating the "exactly one taxonomy marker" rule in
tests/TESTING.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from tengri.observation.fiber_spectroscopy import FiberSpectroscopyObservation
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.gradient


@dataclass(frozen=True)
class _StubObservation:
    """Minimal observation that returns a fixed dict for testing."""

    fixed_output: dict[str, Any]
    name: str = "stub"

    def predict(self, state, params):
        return dict(self.fixed_output)


@pytest.fixture
def state_with_profile():
    """ForwardState with a small Gaussian-like spatial profile centered on origin."""
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


def test_predict_is_jittable_in_redshift(state_with_profile) -> None:
    """jax.grad w.r.t. redshift through jax.jit(predict) is finite.

    Redshift is traced inside every fitter; predict must not call
    float() on it. Differentiates spec_fnu[0] w.r.t. redshift and
    asserts the gradient is finite.
    """
    spec = jnp.ones(10)
    base = _StubObservation(fixed_output={"spec_fnu": spec})
    obs = FiberSpectroscopyObservation(
        observation=base, fiber_radius_arcsec=1.0, fiber_center_arcsec=(0.3, -0.2)
    )
    f = jax.jit(lambda z: obs.predict(state_with_profile, {"redshift": z})["spec_fnu"])
    eager = obs.predict(state_with_profile, {"redshift": jnp.float64(0.05)})["spec_fnu"]
    assert jnp.allclose(f(jnp.float64(0.05)), eager, rtol=1e-12)
    g = jax.grad(lambda z: f(z)[0])(jnp.float64(0.05))
    assert jnp.isfinite(g), "predict's redshift gradient must not be NaN/inf under jit"
    assert g != 0.0, (
        "redshift moves the fiber center in kpc (arcsec_to_kpc) and therefore the "
        "aperture-weighted flux; a gradient of exactly zero here would mean the "
        "traced fiber center silently stopped influencing the output"
    )
