# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fiber aperture-fraction utilities."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.observation.fiber_aperture import (
    aperture_fraction,
    arcsec_to_kpc,
    circular_aperture_mask,
)
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds


@pytest.fixture
def grid_2kpc():
    """64x64 grid spanning +/-2 kpc per axis (fine enough for sharp masks)."""
    axis = jnp.linspace(-2.0, 2.0, 64)
    xx, yy = jnp.meshgrid(axis, axis)
    return xx, yy


def test_arcsec_to_kpc_z_dependence() -> None:
    """1 arcsec maps to monotonically increasing kpc up to z~1.7, then decreases.

    Angular diameter distance peaks around z~1.7 in Lambda-CDM.
    """
    sizes = jnp.array([float(arcsec_to_kpc(1.0, z)) for z in (0.01, 0.1, 1.0, 1.7)])
    # Strictly increasing from z=0.01 to z=1.7
    assert sizes[1] > sizes[0]
    assert sizes[2] > sizes[1]
    assert sizes[3] > sizes[2]


def test_arcsec_to_kpc_units_make_sense() -> None:
    """At z=0.05, 1 arcsec is ~1 kpc."""
    kpc = float(arcsec_to_kpc(1.0, 0.05))
    assert 0.8 < kpc < 1.4


def test_circular_aperture_mask_inside_and_outside(grid_2kpc) -> None:
    """Mask is ~1 inside, ~0 outside the aperture."""
    mask = circular_aperture_mask(grid_2kpc, radius_kpc=1.0)
    assert mask.shape == (64, 64)
    # Center (x=0, y=0) is well inside
    assert float(mask[32, 32]) > 0.9
    # Corner (x≈-2, y≈-2; r≈2.83) is well outside r=1
    assert float(mask[0, 0]) < 0.01


def test_circular_aperture_mask_hard_edge_when_softness_zero(grid_2kpc) -> None:
    """softness=0 gives a strict 0/1 mask."""
    mask = circular_aperture_mask(grid_2kpc, radius_kpc=1.0, softness=0.0)
    unique = jnp.unique(mask)
    # Only two unique values (0 and 1) for a hard edge
    assert set(float(v) for v in unique) == {0.0, 1.0}


def test_aperture_fraction_full_aperture_is_one(grid_2kpc) -> None:
    """When aperture covers the whole grid, the fraction is ~1."""
    profile = jnp.exp(-jnp.sqrt(grid_2kpc[0] ** 2 + grid_2kpc[1] ** 2))
    frac = aperture_fraction(profile, grid_2kpc, radius_kpc=10.0)
    assert float(frac) > 0.99


def test_aperture_fraction_tiny_aperture_is_small(grid_2kpc) -> None:
    """Small aperture relative to the profile scale-length captures little."""
    # Profile with scale length 1 kpc; aperture 0.05 kpc (very small)
    profile = jnp.exp(-jnp.sqrt(grid_2kpc[0] ** 2 + grid_2kpc[1] ** 2) / 1.0)
    frac = aperture_fraction(profile, grid_2kpc, radius_kpc=0.05)
    assert float(frac) < 0.01


def test_aperture_fraction_monotonic_in_radius(grid_2kpc) -> None:
    """Larger aperture captures more light."""
    profile = jnp.exp(-jnp.sqrt(grid_2kpc[0] ** 2 + grid_2kpc[1] ** 2))
    fracs = [float(aperture_fraction(profile, grid_2kpc, radius_kpc=r)) for r in (0.5, 1.0, 1.5)]
    assert fracs[0] < fracs[1] < fracs[2]


def test_aperture_fraction_differentiable_in_radius(grid_2kpc) -> None:
    """Gradient w.r.t. radius is finite and positive (more radius → more flux)."""
    profile = jnp.exp(-jnp.sqrt(grid_2kpc[0] ** 2 + grid_2kpc[1] ** 2))

    def frac_of_r(r):
        return aperture_fraction(profile, grid_2kpc, radius_kpc=r)

    dfrac = assert_grad_matches_fd(frac_of_r, jnp.float64(1.0))
    assert jnp.isfinite(dfrac)
    assert float(dfrac) > 0.0


def test_aperture_fraction_offset_center(grid_2kpc) -> None:
    """An aperture far off-center captures less of the central profile."""
    profile = jnp.exp(-jnp.sqrt(grid_2kpc[0] ** 2 + grid_2kpc[1] ** 2))
    frac_centered = aperture_fraction(profile, grid_2kpc, radius_kpc=0.5)
    frac_offset = aperture_fraction(profile, grid_2kpc, radius_kpc=0.5, center_kpc=(1.5, 0.0))
    assert frac_centered > frac_offset
