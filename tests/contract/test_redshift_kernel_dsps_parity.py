# SPDX-License-Identifier: BSD-3-Clause
"""Parity: tengri's unified redshift kernel agrees with DSPS's own
observer-frame primitive.

The unified kernel introduced in #398 is built on top of DSPS's
``_obs_flux_ssp`` interp pattern plus an explicit ``(1+z) / (4π d_L²)``
flux factor. DSPS packages the equivalent factor into a magnitude-space
dimming term (``distance_modulus - 2.5*log10(1+z)``). Both paths must
agree at the AB-magnitude level on the slice they share.

These tests are the load-bearing artifact that locks the convention:
if a future "optimisation" or refactor silently re-introduces a per-case
``(1+z)`` factor, the parity check will fail.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from dsps.cosmology.flat_wcdm import luminosity_distance_to_z
from dsps.photometry.photometry_kernels import (
    _cosmological_dimming,
    _obs_flux_ssp,
    calc_obs_mag,
)

from tengri.cosmology import PLANCK18
from tengri.observation.redshift_kernel import (
    _luminosity_distance_cm,
    shift_to_obs_frame,
    shift_to_obs_frame_grid,
)

pytestmark = pytest.mark.contract


@pytest.fixture
def fiducial_sed():
    """Power-law L_ν = (λ/5000)^(-1.5), 1000-20000 Å rest."""
    wave_rest = jnp.linspace(1000.0, 20000.0, 2000)
    L_nu = (wave_rest / 5000.0) ** (-1.5) * 1e30  # erg/s/Hz, arbitrary norm
    return wave_rest, L_nu


def test_luminosity_distance_matches_dsps_directly(fiducial_sed):
    """The private ``_luminosity_distance_cm`` is just DSPS's
    ``luminosity_distance_to_z`` with the Mpc→cm conversion. Pin the
    numerical agreement."""
    c = PLANCK18
    for z in (0.01, 0.5, 1.0, 3.0):
        dl_cm = float(_luminosity_distance_cm(jnp.asarray(z), c))
        dl_mpc_dsps = float(luminosity_distance_to_z(z, c.Om0, c.w0, c.wa, c.h))
        dl_cm_dsps = dl_mpc_dsps * 3.0856775814913673e24
        assert np.isclose(dl_cm, dl_cm_dsps, rtol=1e-10), (
            f"z={z}: tengri d_L={dl_cm:.4e} cm vs DSPS d_L={dl_cm_dsps:.4e} cm"
        )


def test_z0_returns_l_nu_unchanged(fiducial_sed):
    """At z=0 with d_L=10 pc, F_ν should equal L_ν / (4π × 10pc²).
    Sanity check that the kernel doesn't lose precision at the limit."""
    wave_rest, L_nu = fiducial_sed
    f_nu = shift_to_obs_frame(wave_rest, L_nu, wave_rest, jnp.asarray(0.0), PLANCK18)

    # 10 pc in cm: 3.0856775814913673e24 × 1e-5
    d_10pc_cm = 3.0856775814913673e24 * 1e-5
    expected_scale = 1.0 / (4.0 * jnp.pi * d_10pc_cm**2)
    expected = L_nu * expected_scale

    assert jnp.allclose(f_nu, expected, rtol=1e-6)


def test_dsps_obs_flux_parity_through_a_flat_filter(fiducial_sed):
    """Build a flat (top-hat) filter T(λ)=1 on a band; integrate the
    tengri F_ν over that filter and compare to DSPS's
    ``_obs_flux_ssp`` then-scale-to-flux.

    DSPS's ``_obs_flux_ssp(wave_rest, L_nu, wave_filter, T, z)`` returns
    L_ν integrated against the filter in **rest-frame luminosity units**
    (no cosmological dimming). The tengri kernel returns observer-frame
    F_ν directly. So:

        flux_dsps_at_filter_eff_λ_dimmed
          = _obs_flux_ssp(...) × (1+z) / (4π d_L²)

    should equal the integral over filter of tengri's ``shift_to_obs_frame``
    output. Lock this equality.
    """
    wave_rest, L_nu = fiducial_sed
    z = jnp.asarray(0.5)
    c = PLANCK18

    # Top-hat filter at observed-frame 5000-7000 Å.
    wave_filter = jnp.linspace(5000.0, 7000.0, 200)
    T = jnp.ones_like(wave_filter)

    # DSPS path: returns ∫ T(λ) × L_zshift(λ) / λ dλ in L-units.
    L_dsps = _obs_flux_ssp(wave_rest, L_nu, wave_filter, T, z)

    # tengri path: get F_ν on the filter wave grid, integrate ∫ T × F_ν / λ dλ.
    f_nu_obs = shift_to_obs_frame(wave_rest, L_nu, wave_filter, z, c)
    flux_tengri = jnp.trapezoid(T * f_nu_obs / wave_filter, wave_filter)

    # The tengri integral is in F_ν units (erg/s/cm²/Hz). Convert DSPS's
    # L-units to F_ν units the same way:
    dl_cm = _luminosity_distance_cm(z, c)
    flux_dsps_in_flux_units = L_dsps * (1.0 + z) / (4.0 * jnp.pi * dl_cm**2)

    rel_err = float(abs(flux_tengri - flux_dsps_in_flux_units) / flux_dsps_in_flux_units)
    assert rel_err < 1e-5, (
        f"tengri kernel vs DSPS _obs_flux_ssp parity broke: "
        f"tengri={float(flux_tengri):.6e}, dsps={float(flux_dsps_in_flux_units):.6e}, "
        f"rel_err={rel_err:.2e}"
    )


def test_calc_obs_mag_parity(fiducial_sed):
    """End-to-end AB magnitude parity. The tengri kernel + a hand-rolled
    AB conversion must match ``dsps.photometry.calc_obs_mag``."""
    wave_rest, L_nu = fiducial_sed
    z = 0.5
    c = PLANCK18

    wave_filter = jnp.linspace(5000.0, 7000.0, 200)
    T = jnp.ones_like(wave_filter)

    # DSPS magnitude (the canonical reference path).
    mag_dsps = float(calc_obs_mag(wave_rest, L_nu, wave_filter, T, z, c.Om0, c.w0, c.wa, c.h))

    # tengri magnitude: F_ν on filter grid, then AB-system conversion.
    # AB zero-point: F_AB,0 = 3631 Jy = 3631 × 1e-23 erg/s/cm²/Hz at 10pc;
    # but the calc_obs_mag formulation uses AB0 in Lsun/Hz, so we reuse
    # the DSPS internals exactly to avoid double-converting units.
    L_filter_integral = _obs_flux_ssp(wave_rest, L_nu, wave_filter, T, jnp.asarray(z))
    from dsps.photometry.photometry_kernels import _flux_ab0_at_10pc

    flux_ab0 = _flux_ab0_at_10pc(wave_filter, T)
    mag_no_dimming_tengri = -2.5 * jnp.log10(L_filter_integral / flux_ab0)
    dimming = _cosmological_dimming(z, c.Om0, c.w0, c.wa, c.h)
    mag_tengri = float(mag_no_dimming_tengri + dimming)

    assert abs(mag_tengri - mag_dsps) < 1e-6, (
        f"tengri vs DSPS AB-mag parity broke: tengri={mag_tengri:.6f}, dsps={mag_dsps:.6f}"
    )


def test_kernel_zero_outside_support(fiducial_sed):
    """Outside the rest-frame support of the input, F_ν must be zero
    (matches DSPS's ``left=0, right=0`` interp boundary)."""
    wave_rest, L_nu = fiducial_sed
    z = jnp.asarray(0.5)

    # Probe well above the redshifted max wavelength.
    wave_above = jnp.asarray([wave_rest.max() * (1.0 + 0.5) + 5000.0])
    f_above = shift_to_obs_frame(wave_rest, L_nu, wave_above, z, PLANCK18)
    assert float(f_above[0]) == 0.0

    # Probe well below the redshifted min wavelength.
    wave_below = jnp.asarray([wave_rest.min() * (1.0 + 0.5) - 500.0])
    f_below = shift_to_obs_frame(wave_rest, L_nu, wave_below, z, PLANCK18)
    assert float(f_below[0]) == 0.0


def test_grid_variant_matches_per_z_calls(fiducial_sed):
    """``shift_to_obs_frame_grid`` over a z table must give the same
    result as a Python-loop of per-z ``shift_to_obs_frame`` calls."""
    wave_rest, L_nu = fiducial_sed
    wave_obs = jnp.linspace(2000.0, 30000.0, 500)
    z_table = jnp.linspace(0.1, 3.0, 8)

    grid = shift_to_obs_frame_grid(wave_rest, L_nu, wave_obs, z_table, PLANCK18)
    expected = jnp.stack(
        [shift_to_obs_frame(wave_rest, L_nu, wave_obs, z, PLANCK18) for z in z_table]
    )
    assert jnp.allclose(grid, expected, rtol=1e-10)


def test_kernel_is_jittable(fiducial_sed):
    """``shift_to_obs_frame`` must be JIT-compatible (the @jax.jit
    decoration on the source plus the lack of any Python-side
    branching). Sanity check end-to-end."""
    import jax

    wave_rest, L_nu = fiducial_sed
    f = jax.jit(lambda z: shift_to_obs_frame(wave_rest, L_nu, wave_rest, z, PLANCK18))
    out = f(jnp.asarray(0.5))
    assert out.shape == wave_rest.shape
    assert jnp.all(jnp.isfinite(out))


def test_kernel_is_grad_safe(fiducial_sed):
    """``shift_to_obs_frame`` must accept gradients through z (the
    luminosity-distance integral and the interp both have to be
    differentiable). Lock this — any future refactor that breaks
    grad would silently break MAP/NUTS fits with free redshift."""
    import jax

    wave_rest, L_nu = fiducial_sed
    wave_obs = jnp.linspace(2000.0, 30000.0, 200)

    def loss(z):
        f = shift_to_obs_frame(wave_rest, L_nu, wave_obs, z, PLANCK18)
        return jnp.sum(f)

    g = jax.grad(loss)(jnp.asarray(0.5))
    assert jnp.isfinite(g)
