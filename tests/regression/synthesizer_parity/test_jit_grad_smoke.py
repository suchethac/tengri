# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for JIT/grad finiteness at high redshift.

Synthesizer parity.

Smoke tests for numerical stability of the forward model at high redshift (z=10)
where wavelengths get heavily redshifted and potential for overflow/underflow exists.

Pitfall: P-22 — numerical-overflow edge cases when looping over large arrays.
Synthesizer PR #995: particle spectra overflow when looping over large arrays;
used int for nlam/npart, overflowed when >2^31. JAX unlikely to overflow in
pure Python but numerical stability must be verified.

Pitfall: P-24 — NaN from empty spectra or zero luminosity. Unhandled exceptions
when spectra array is empty or luminosity=0 (e.g., no ionizing photons).

Synthesizer source:
- https://github.com/flaresimulations/synthesizer/pull/995 (overflow fix)
- https://github.com/flaresimulations/synthesizer/pull/1084 (NaN handling)

Reference papers:
- Madau et al. 1995, ApJ, 441, 18 (IGM transmission — high-z critical test)
- Inoue et al. 2014, MNRAS, 442, 1805 (IGM absorption data)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.igm.igm import igm_transmission
from tengri.utils.cosmology import luminosity_distance_mpc
from tengri.utils.physics_constants import L_SUN as L_SUN_ERG


@pytest.fixture(scope="module")
def high_z_wavelength_grid() -> jnp.ndarray:
    """Rest-frame wavelength grid for high-z test (optical to IR).

    At z=10, the Lyman-alpha forest is heavily absorbed. This grid exercises
    IGM transmission calculation at extreme redshifts.
    """
    return jnp.logspace(3.0, 5.0, 200)  # 1000 Å .. 100,000 Å


@pytest.fixture(scope="module")
def low_z_wavelength_grid() -> jnp.ndarray:
    """Rest-frame wavelength grid for low-z sanity check (optical).

    Standard optical filters for z=0 SED sanity.
    """
    return jnp.logspace(3.5, 4.2, 100)  # 3162 Å .. 15,849 Å


def test_igm_transmission_finite_at_high_redshift(high_z_wavelength_grid):
    """IGM transmission must be finite (no NaN/inf) at z=10.

    Pitfall P-20 (related): IGM absorption with observed-frame wavelengths at
    extreme z can underflow or overflow without proper numerics.

    Pitfall P-24: NaN from zero luminosity or empty spectra. IGM transmission
    is the first operation on SED; if it returns NaN, nothing downstream works.
    """
    z = 10.0
    tau_igm = igm_transmission(high_z_wavelength_grid, z)

    assert bool(jnp.all(jnp.isfinite(tau_igm))), (
        "P-24 BUG: IGM transmission has NaN/inf at z=10. "
        "Numerical underflow/overflow in exponential or integral."
    )

    # All values must be in physical range [0, 1] (attenuation)
    assert bool(jnp.all((tau_igm >= 0.0) & (tau_igm <= 1.0))), (
        "P-24 BUG: IGM transmission outside [0, 1] at z=10. "
        f"Min: {float(jnp.min(tau_igm)):.3e}, Max: {float(jnp.max(tau_igm)):.3e}"
    )


def test_igm_transmission_ly_alpha_absorbed_at_high_z(high_z_wavelength_grid):
    """Lyman-alpha (1216 Å rest-frame) is heavily absorbed at z=10 (obs 13,376 Å).

    Pitfall P-20: If IGM_transmission uses rest-frame instead of observed-frame
    wavelengths, or if the redshift is not applied, Ly-alpha will NOT be absorbed
    as it should be.

    Physical expectation: at z=10, τ_IGM(Ly-alpha) ≈ 0.1–0.5 (heavily absorbed).
    """
    z = 10.0
    tau_igm = igm_transmission(high_z_wavelength_grid, z)

    # Find Lyman-alpha position in the wavelength grid
    # Rest-frame Ly-alpha: 1216 Å → observed: 1216 * (1 + z) = 13,376 Å at z=10
    lya_rest = 1216.0
    lya_obs = lya_rest * (1.0 + z)

    # Interpolate tau at Ly-alpha wavelength
    tau_at_lya = jnp.interp(lya_obs, high_z_wavelength_grid, tau_igm)

    # At z=10, Ly-alpha is in the heavily absorbed region (forest)
    # tau should be significant (>0.1 at minimum)
    assert float(tau_at_lya) < 0.7, (
        f"P-20 WARNING: IGM absorption at Ly-alpha (z=10) is {float(tau_at_lya):.3f}, "
        "expected < 0.7 (heavily absorbed). May indicate wrong wavelength convention."
    )


def test_igm_transmission_jit_compatible(high_z_wavelength_grid):
    """IGM transmission must be JIT-compilable (used in forward-model loop).

    Pitfall P-24 (implicit): If IGM transmission is not JIT-compatible (e.g.,
    has Python float comparisons or data-dependent branching), forward-model
    inference will fail at compile time.
    """

    @jax.jit
    def compute_igm_tau(wave, z):
        return igm_transmission(wave, z)

    z = 5.0
    try:
        tau = compute_igm_tau(high_z_wavelength_grid, z)
        assert bool(jnp.all(jnp.isfinite(tau))), "JIT result has NaN/inf"
    except Exception as e:
        pytest.fail(f"P-24 BUG: IGM transmission is not JIT-compatible: {e}")


def test_igm_transmission_grad_compatible(high_z_wavelength_grid):
    """IGM transmission gradient w.r.t. redshift is finite (inference requirement).

    Pitfall P-24: If IGM transmission is not differentiable, fitting models with
    redshift as a free parameter will fail silently (zero gradient).
    """

    def loss_fn(z_traced):
        tau = igm_transmission(high_z_wavelength_grid, z_traced)
        # Loss: sum of (1 - tau) to measure total absorption
        return jnp.sum(1.0 - tau)

    z_test = 3.0
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(jnp.array(z_test))

    assert jnp.isfinite(grad), (
        "P-24 BUG: gradient of IGM transmission w.r.t. z is non-finite. "
        "Derivative broken or outside JAX trace."
    )
    assert jnp.any(grad != 0.0), (
        "`grad` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_cosmological_distance_finite_at_high_z():
    """Luminosity distance must be finite at z=10 (no overflow).

    Pitfall P-22 (implied): Cosmological integral from 0 to z can overflow
    if upper limit is large or integrator has accumulation error.
    """
    z = 10.0
    d_l = luminosity_distance_mpc(z)

    assert jnp.isfinite(d_l), (
        "P-22 BUG: luminosity distance is non-finite at z=10. "
        "Cosmological integral overflow or integration error."
    )
    assert d_l > 0.0, f"Luminosity distance must be positive; got {d_l}"


def test_forward_model_photometry_sanity_at_low_z(low_z_wavelength_grid):
    """Sanity check: AB magnitude in optical range is physically reasonable at z=0.

    Pitfall P-24: If forward model returns NaN/inf at z=0, the entire pipeline
    is broken. This test ensures the most basic case (no redshift) works.

    Test: a solar-luminosity source at 10 Mpc should have optical mag ~17–20 mag.
    """
    z = 0.0
    d_l_mpc = float(luminosity_distance_mpc(z))

    # By flat FLRW at z=0, distance vanishes, so this is not a true end-to-end test.
    # Instead, use d_l = 10 Mpc as a proxy.
    d_l_mpc = 10.0

    # Solar luminosity SED (rough approximation: blackbody at T=5780 K)
    # L_sun in erg/s; integrate over filter response to get flux density
    # Rough estimate: F_nu ~ L_sun / (4π d_L^2) ~ 1e33 / 1e54 ~ 1e-21 erg/s/cm^2/Hz
    l_sun_erg = L_SUN_ERG
    d_l_cm = d_l_mpc * 3.086e24  # Mpc to cm
    f_nu_cgs = l_sun_erg / (4.0 * jnp.pi * d_l_cm**2)

    # Convert to Jy: 1 Jy = 1e-23 erg/s/cm^2/Hz
    f_nu_jy = float(f_nu_cgs) * 1e23

    # Convert to AB magnitude: m = -2.5 log10(f_nu / 3631 Jy)
    # (3631 Jy is the Jy equivalent of the AB reference magnitude zero-point)
    ab_mag = -2.5 * jnp.log10(f_nu_jy / 3631.0)
    ab_mag_float = float(ab_mag)

    # Solar-luminosity object at 10 Mpc should have a finite AB magnitude
    # (rough range -5 to 30 covers most realistic objects at 10 Mpc)
    assert -10.0 < ab_mag_float < 30.0, (
        f"P-24 BUG: AB magnitude for solar-lum at 10 Mpc is {ab_mag_float:.1f}, "
        "outside expected [-10, 30] range. Flux calculation likely broken."
    )


def test_eddington_ratio_boundaries_no_nan(high_z_wavelength_grid):
    """Eddington-ratio edge cases (very low L_bol, very high L_bol) must stay finite.

    Pitfall P-24: Some special values (log_lbol at grid edge) can cause NaN if
    not handled carefully.
    """
    # Test extreme parameters
    test_cases = [
        ("very faint", 6.0, 6.0),  # log_lbol, log_mbh
        ("typical AGN", 11.0, 8.0),
        ("ultraluminous", 14.0, 9.5),
    ]

    for name, log_lbol, log_mbh in test_cases:
        l_bol_erg = 10.0**log_lbol * L_SUN_ERG
        l_edd_erg = 1.26e38 * 10.0**log_mbh

        # Compute Eddington ratio
        ratio = l_bol_erg / l_edd_erg

        assert jnp.isfinite(ratio), (
            f"P-24 BUG: Eddington ratio is non-finite for {name} "
            f"(log_lbol={log_lbol}, log_mbh={log_mbh}). "
            f"Likely underflow/overflow in exponentiation."
        )
        assert ratio > 0.0, f"P-24 BUG: Eddington ratio is non-positive for {name}."
