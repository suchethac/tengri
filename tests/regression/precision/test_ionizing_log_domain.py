# SPDX-License-Identifier: BSD-3-Clause
"""Test log-domain ionizing-photon integral for float32 safety (issue #1206).

Validates that the core Q_H integral is computed in log-domain to prevent
float32 overflow (Q_H ~ 1e56 photons/s exceeds float32 max ~3.4e38).
The log-domain formulation (_integrate_nion_log10) keeps all intermediates
within float32 range; the linear wrapper (_integrate_nion) provides the
user-facing contract.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.component import (
    _HI_LIMIT_AA,
    _integrate_nion,
    _integrate_nion_log10,
)
from tengri.utils.physics_constants import C_AA, H_PLANCK
from tengri.utils.scale import pow10

from .conftest import build_minimal_cue_model

pytestmark = pytest.mark.regression_bug


# FROZEN pre-log_nion reference (#1206) — exact copy of the original _integrate_nion
# from component.py:494-536, used to validate f64 exactness.
def _frozen_integrate_nion(sed_lnu: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Frozen reference implementation (pre-log_nion)."""
    nu = C_AA / wave
    nu_edge = C_AA / _HI_LIMIT_AA
    integrand = sed_lnu / (H_PLANCK * nu)
    ionizing_mask = wave < _HI_LIMIT_AA
    integrand_masked = jnp.where(ionizing_mask, integrand, 0.0)
    idx_below = jnp.argmax(jnp.where(ionizing_mask, jnp.arange(wave.shape[0]), -1))
    idx_above = idx_below + 1
    integrand_below = integrand[idx_below]
    # Boundary bin: subtract the trapezoid triangle, add the true rectangle.
    triangle_overcount = 0.5 * integrand_below * jnp.abs(nu[idx_below] - nu[idx_above])
    rectangle_correct = integrand_below * jnp.abs(nu[idx_below] - nu_edge)
    nion_bulk = jnp.abs(jnp.trapezoid(integrand_masked, nu))
    return nion_bulk - triangle_overcount + rectangle_correct


def test_f64_exactness_vs_frozen_reference(ssp_bare):
    """(1) Test f64 exactness of new wrapper vs frozen reference.

    Verifies that _integrate_nion (new wrapper) matches the frozen pre-log_nion
    implementation at rtol=1e-12 for several synthetic SEDs and one real SED.
    Also verifies _integrate_nion_log10 matches log10(frozen result) at atol=1e-12 (dex).
    """
    # Synthetic SED on a wave grid spanning 200-2000 Angstrom (~200 points)
    wave = jnp.linspace(200.0, 2000.0, 200)

    # Test 1a: flat SED
    sed_flat = jnp.ones_like(wave)
    ref_flat = _frozen_integrate_nion(sed_flat, wave)
    new_flat = _integrate_nion(sed_flat, wave)
    assert_allclose(new_flat, ref_flat, rtol=1e-12)

    log_new_flat = _integrate_nion_log10(sed_flat, wave)
    log_ref_flat = jnp.log10(ref_flat)
    assert_allclose(log_new_flat, log_ref_flat, atol=1e-12)

    # Test 1b: steep blue power law (proportional to nu^{-1.5} or lambda^{1.5})
    sed_blue = wave**1.5
    ref_blue = _frozen_integrate_nion(sed_blue, wave)
    new_blue = _integrate_nion(sed_blue, wave)
    assert_allclose(new_blue, ref_blue, rtol=1e-12)

    log_new_blue = _integrate_nion_log10(sed_blue, wave)
    log_ref_blue = jnp.log10(ref_blue)
    assert_allclose(log_new_blue, log_ref_blue, atol=1e-12)

    # Test 1c: steep red power law (proportional to nu^{1.5} or lambda^{-1.5})
    sed_red = wave ** (-1.5)
    ref_red = _frozen_integrate_nion(sed_red, wave)
    new_red = _integrate_nion(sed_red, wave)
    assert_allclose(new_red, ref_red, rtol=1e-12)

    log_new_red = _integrate_nion_log10(sed_red, wave)
    log_ref_red = jnp.log10(ref_red)
    assert_allclose(log_new_red, log_ref_red, atol=1e-12)

    # Test 1d: mass-scaled SED (1e42 scales the ionizing flux significantly)
    sed_scaled = 1e42 * sed_flat
    ref_scaled = _frozen_integrate_nion(sed_scaled, wave)
    new_scaled = _integrate_nion(sed_scaled, wave)
    assert_allclose(new_scaled, ref_scaled, rtol=1e-12)

    log_new_scaled = _integrate_nion_log10(sed_scaled, wave)
    log_ref_scaled = jnp.log10(ref_scaled)
    assert_allclose(log_new_scaled, log_ref_scaled, atol=1e-12)

    # Test 1e: real SED from minimal Cue model
    m64 = build_minimal_cue_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    state = m64.predict_state(p)
    sed_real = state.sed_intrinsic
    wave_real = state.wave

    ref_real = _frozen_integrate_nion(sed_real, wave_real)
    new_real = _integrate_nion(sed_real, wave_real)
    assert_allclose(new_real, ref_real, rtol=1e-12)

    log_new_real = _integrate_nion_log10(sed_real, wave_real)
    log_ref_real = jnp.log10(ref_real)
    assert_allclose(log_new_real, log_ref_real, atol=1e-12)


def test_pure_float32_finiteness_and_parity(ssp_bare):
    """(2) Test pure-float32 finiteness and parity.

    Compute a reference Q_H in f64, then recompute in pure f32 via
    jax.enable_x64(False) and verify finite and close (atol=5e-3 dex).
    Uses a synthetic O(1e-7) SED mimicking per-Msun tensordot magnitude.
    """
    # Reference in f64
    wave = jnp.linspace(200.0, 2000.0, 200)
    ell = 1e-7 * jnp.ones_like(wave)  # O(1e-7) normalized SED
    ref_log = _integrate_nion_log10(ell, wave, log10_scale=43.6)

    # Pure f32: recompute with f32 inputs
    with jax.enable_x64(False):
        ell_f32 = jnp.asarray(ell, dtype=jnp.float32)
        wave_f32 = jnp.asarray(wave, dtype=jnp.float32)
        result_f32 = _integrate_nion_log10(ell_f32, wave_f32, log10_scale=43.6)

    # Assert finite
    assert jnp.isfinite(result_f32), f"f32 result is non-finite: {result_f32}"

    # Assert close to reference (5e-3 dex = ~1.2% error in linear)
    ref_f64 = jnp.asarray(ref_log, dtype=jnp.float64)
    assert_allclose(float(result_f32), float(ref_f64), atol=5e-3)


def test_zero_ionizing_flux():
    """(3) Test zero ionizing flux handling.

    SED that is zero everywhere below 911.76 A (Lyman limit) but nonzero above.
    _integrate_nion_log10 should return -inf, _integrate_nion should return 0.0,
    and gradients should be finite (no NaN).
    """
    # Create a wave grid spanning the Lyman limit
    wave = jnp.linspace(200.0, 2000.0, 200)
    # SED is zero below 911.76 A, nonzero above
    sed = jnp.where(wave < _HI_LIMIT_AA, 0.0, 1.0)

    # Test log result
    log_result = _integrate_nion_log10(sed, wave)
    assert log_result == -jnp.inf, f"Expected -inf for zero ionizing flux, got {log_result}"

    # Test linear result
    linear_result = _integrate_nion(sed, wave)
    assert linear_result == 0.0, f"Expected 0.0 for zero ionizing flux, got {linear_result}"

    # Test gradient is finite (no NaN)
    grad_fn = jax.grad(lambda sed_: jnp.sum(_integrate_nion(sed_, wave)))
    grad_result = grad_fn(sed)
    assert jnp.all(jnp.isfinite(grad_result)), (
        f"Gradient contains non-finite values for zero ionizing flux: {grad_result}"
    )


def test_gradient_identity_with_total_mass():
    """(4) Test gradient identity: Q_H ∝ total_mass exactly.

    The stellar component's compute_log_nion should have gradient=1.0 w.r.t.
    sfh_dpl_log_total_mass because the CSP weights exclude total mass.
    """
    from tengri import DEFAULT, Fixed, SEDModel
    from tengri.components.stellar.component import StellarSEDComponent
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
    from tengri.observation import Observation, Photometry

    # Build minimal model to get the stellar component
    ssp_path = "data/fsps_prsc_miles_chabrier.h5"
    ssp = load_ssp_data(ssp_path)

    # Construct stellar component directly
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(1.0),
        approx=None,
        forward_dtype="float64",
    )

    # Get the stellar component from the model's internals
    # Mirror the accessor used in tests/components/spectroscopy/test_wne_window_lut_parity.py
    chain = model._build_component_chain() if hasattr(model, "_build_component_chain") else None
    stellar = next(c for c in chain if isinstance(c, StellarSEDComponent))

    # Sample and get params, override log_total_mass
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))

    # Gradient w.r.t. sfh_dpl_log_total_mass should be 1.0
    grad_fn = jax.grad(lambda lm: stellar.compute_log_nion({**p, "sfh_dpl_log_total_mass": lm}))
    gradient = grad_fn(10.0)

    assert_allclose(float(gradient), 1.0, rtol=1e-9)


def test_published_keys_in_derived():
    """(5) Test published keys: both log_nion and nion in derived state.

    Verify that m.predict_state(p).derived contains BOTH "log_nion" and "nion",
    with pow10(log_nion) == nion bitwise and abs(log10(nion) - log_nion) < 1e-12.
    """
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")

    from tengri import DEFAULT, Fixed, SEDModel
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(1.0),
        approx=None,
        forward_dtype="float64",
    )

    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(p)

    # Check both keys exist
    assert "log_nion" in state.derived, "log_nion not found in state.derived"
    assert "nion" in state.derived, "nion not found in state.derived"

    log_nion = state.derived["log_nion"]
    nion = state.derived["nion"]

    # Check pow10(log_nion) == nion (bitwise)
    nion_from_log = pow10(log_nion)
    np.testing.assert_equal(nion_from_log, nion, err_msg="pow10(log_nion) != nion")

    # Check abs(log10(nion) - log_nion) < 1e-12 (round-trip)
    if nion > 0:
        log10_nion = jnp.log10(nion)
        assert_allclose(log10_nion, log_nion, atol=1e-12)
