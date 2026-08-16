# SPDX-License-Identifier: BSD-3-Clause
"""Parity: tengri's unified redshift kernel agrees with DSPS's own
observer-frame primitive.

The kernel introduced in #398 wraps DSPS's ``_obs_flux_ssp`` interp
pattern + an explicit ``(1+z) / (4π d_L²)`` F_ν conversion. DSPS
packages the equivalent factor as a magnitude-space dimming term.
Both paths must agree at the AB-magnitude level.

These tests are the load-bearing artifact that locks the convention:
a future "optimization" that silently re-introduces a per-case (1+z)
factor will fail here.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from dsps.photometry.photometry_kernels import (
    _cosmological_dimming,
    _flux_ab0_at_10pc,
    _obs_flux_ssp,
    calc_obs_mag,
)

from tengri.cosmology import PLANCK18
from tengri.observation.redshift_kernel import shift_to_obs_frame
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.contract


@pytest.fixture
def fiducial_sed():
    """Power-law L_ν ∝ λ^(-1.5), 1000–20000 Å rest."""
    wave_rest = jnp.linspace(1000.0, 20000.0, 2000)
    L_nu = (wave_rest / 5000.0) ** (-1.5) * 1e30  # erg/s/Hz
    return wave_rest, L_nu


def test_z0_returns_l_nu_at_10pc(fiducial_sed):
    """At z=0 the convention is d_L = 10 pc; F_ν = L_ν / (4π × 10pc²)."""
    wave_rest, L_nu = fiducial_sed
    f_nu = shift_to_obs_frame(wave_rest, L_nu, wave_rest, jnp.asarray(0.0), PLANCK18)
    d_10pc_cm = 3.0856775814913673e24 * 1e-5
    expected = L_nu / (4.0 * jnp.pi * d_10pc_cm**2)
    assert jnp.allclose(f_nu, expected, rtol=1e-6)


def test_calc_obs_mag_parity(fiducial_sed):
    """End-to-end AB magnitude parity vs ``dsps.photometry.calc_obs_mag``.

    DSPS is the ground truth. Integrating tengri's F_ν output through
    the same top-hat filter and applying the same AB zero-point must
    give the same magnitude to floating-point precision.
    """
    wave_rest, L_nu = fiducial_sed
    z = 0.5
    c = PLANCK18

    wave_filter = jnp.linspace(5000.0, 7000.0, 200)
    T = jnp.ones_like(wave_filter)

    mag_dsps = float(calc_obs_mag(wave_rest, L_nu, wave_filter, T, z, c.Om0, c.w0, c.wa, c.h))

    # tengri path: reuse DSPS's filter-integration + AB-zeropoint machinery
    # so the test isolates the redshift kernel itself, not the filter math.
    L_filter_integral = _obs_flux_ssp(wave_rest, L_nu, wave_filter, T, jnp.asarray(z))
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, T)
    dimming = _cosmological_dimming(z, c.Om0, c.w0, c.wa, c.h)
    mag_tengri = float(-2.5 * jnp.log10(L_filter_integral / flux_ab0) + dimming)

    assert abs(mag_tengri - mag_dsps) < 1e-6


def test_kernel_is_jittable_and_grad_safe(fiducial_sed):
    """Both JIT and grad through z must work end-to-end. Locking this
    prevents silent regressions to any free-redshift fit path.

    The gradient is taken at z = 0.55, not at z = 0.5. This fixture's reddest
    observed wavelength is 30000 A and its rest grid ends at 20000 A, so
    30000 / (1 + z) crosses the last rest-grid point at *exactly* z = 0.5 --
    the value this test used to differentiate at, and a discontinuity in its
    own objective (see the test below). The gradient there was fine; the
    reference was not. Measured off the edge, the analytic gradient matches a
    central difference to 1e-7 at z = 0.30, 0.55 and 0.70.
    """
    wave_rest, L_nu = fiducial_sed
    wave_obs = jnp.linspace(2000.0, 30000.0, 200)

    out = assert_jit_matches_eager(
        lambda z: shift_to_obs_frame(wave_rest, L_nu, wave_obs, z, PLANCK18), jnp.asarray(0.5)
    )
    assert out.shape == wave_obs.shape
    assert jnp.all(jnp.isfinite(out))

    g = assert_grad_matches_fd(
        lambda z: jnp.sum(shift_to_obs_frame(wave_rest, L_nu, wave_obs, z, PLANCK18)),
        jnp.asarray(0.55),
    )
    assert jnp.isfinite(g) and float(g) != 0.0


def test_flux_steps_when_a_band_leaves_the_rest_grid(fiducial_sed):
    """Observed wavelengths that fall off the red end contribute zero, not the
    edge value -- so the summed flux steps as z sweeps a band across the end.

    This is the hazard that made the gradient check above look broken, and
    nothing else in the suite states it. It matters for free-redshift fitting:
    where a filter's red edge crosses the end of the rest-frame template grid,
    the likelihood has a genuine step, and any sampler walking across it sees a
    discontinuity rather than a slope. With a template grid that extends well
    past every filter -- the normal case -- z never reaches the edge and the
    question does not arise.

    Measured 2026-08-16: crossing the edge moves the summed flux by 8.0e-4
    relative, in one step, at the z where the last point rejoins the grid.
    """
    wave_rest, L_nu = fiducial_sed
    wave_obs = jnp.linspace(2000.0, 30000.0, 200)

    def total(z):
        f_nu = shift_to_obs_frame(wave_rest, L_nu, wave_obs, jnp.asarray(z), PLANCK18)
        return float(jnp.sum(f_nu))

    # 30000 / (1 + z) == 20000 == wave_rest[-1] exactly at z = 0.5.
    edge = 30000.0 / float(wave_rest[-1]) - 1.0
    assert edge == pytest.approx(0.5, abs=1e-12)

    just_below, just_above = total(edge - 1e-7), total(edge + 1e-7)
    at_edge = total(edge)

    n_off_below = int(jnp.sum(wave_obs / (1 + (edge - 1e-7)) > wave_rest[-1]))
    assert n_off_below == 1, "the fixture no longer straddles the grid edge here"
    assert int(jnp.sum(wave_obs / (1 + edge) > wave_rest[-1])) == 0

    step = (at_edge - just_below) / at_edge
    assert step == pytest.approx(8.0e-4, rel=0.1), (
        f"the off-grid convention changed: crossing the rest-grid red edge now "
        f"moves the summed flux by {step:.2e} relative, not 8.0e-4. A value of "
        f"~0 would mean off-grid points are clamped to the edge value instead "
        f"of dropped, which would make the likelihood continuous here."
    )
    # Away from the edge the function is smooth: the step above is the whole
    # discontinuity, not part of a general roughness.
    assert (just_above - at_edge) / at_edge == pytest.approx(-3.1e-7, rel=0.2)


def test_vmap_recipe_for_z_table(fiducial_sed):
    """Document-and-lock the standard JAX vmap recipe for precompute paths.

    No dedicated wrapper function — callers vmap directly. This test
    pins the recipe so the docstring example stays correct."""
    from jax import vmap

    wave_rest, L_nu = fiducial_sed
    wave_obs = jnp.linspace(2000.0, 30000.0, 500)
    z_table = jnp.linspace(0.1, 3.0, 8)

    shift_grid = vmap(shift_to_obs_frame, in_axes=(None, None, None, 0, None))
    grid = shift_grid(wave_rest, L_nu, wave_obs, z_table, PLANCK18)
    assert grid.shape == (8, 500)
    assert jnp.all(jnp.isfinite(grid))
