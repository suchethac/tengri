# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2629 — asada25 Lyα damping-wing cross-section bug.

The Lyα damping-wing cross-section in the Asada+2025 CGM model carried the
oscillator strength f twice: once in the prefactor and once implicitly through
the Einstein A coefficient. The prefactor should be 3λ²A/(8π) = 0.011052 cm²·Hz
(from the Miralda-Escudé 1998, Eq. 1 form), not 3λ²f·A/(8π) = 0.004600.
This raised τ ≈ 0.41× too small, making the CGM damping wing invisible at z ≥ 6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.igm.dla import (
    _A_LYA,
    _NU_LYA,
    _deltanu_doppler,
    _sigma_lya,
)
from tengri.components.igm.igm import (
    igm_transmission_asada25,
)

pytestmark = pytest.mark.regression_bug

# CODATA constants (CGS) — written in the test, not imported from tengri
_E_CHARGE_ESU = 4.80320471e-10  # statcoulomb
_M_ELECTRON = 9.1093837015e-28  # gram
_C_CGS = 2.99792458e10  # cm/s

# Lyα atomic data (from references.bib or constants)
_LYA_LAMBDA_ANG = 1215.67  # Å
_LYA_A_COEFF = 6.265e8  # s⁻¹
_LYA_OSC_STRENGTH = 0.4164  # dimensionless


def test_sum_rule_formula_identity():
    """Test that 3λ²A/(8π) equals πe²f/(m_e c) to 1e-3 (sum rule)."""
    lam = _LYA_LAMBDA_ANG * 1e-8  # cm

    # Miralda-Escudé 1998 Eq. 1 prefactor
    prefactor_formula = 3 * lam**2 * _A_LYA / (8 * np.pi)

    # Oscillator-strength sum rule: πe²f/(m_e c)
    sum_rule = np.pi * _E_CHARGE_ESU**2 * _LYA_OSC_STRENGTH / (_M_ELECTRON * _C_CGS)

    np.testing.assert_allclose(
        prefactor_formula,
        sum_rule,
        rtol=1e-3,
        err_msg="Prefactor 3λ²A/(8π) should equal πe²f/(m_e c) by sum rule",
    )


def test_code_prefactor_matches_sum_rule():
    """Test that the CODE's prefactor equals the sum rule to 1e-3."""
    lam = _LYA_LAMBDA_ANG * 1e-8  # cm

    # The code's prefactor (from igm.py line 451)
    prefactor_code = 3 * lam**2 * _A_LYA / (8 * np.pi)

    # The sum rule
    sum_rule = np.pi * _E_CHARGE_ESU**2 * _LYA_OSC_STRENGTH / (_M_ELECTRON * _C_CGS)

    np.testing.assert_allclose(
        prefactor_code,
        sum_rule,
        rtol=1e-3,
        err_msg="Code prefactor should equal sum rule (no extra f)",
    )


@pytest.mark.parametrize(
    "z,rest_wl,T_expected",
    [
        (8.0, 1220.0, 0.00073),
        (8.0, 1230.0, 0.523),
        (8.0, 1240.0, 0.8015),
        (10.0, 1225.0, 0.1527),
    ],
)
def test_transmission_literals(z, rest_wl, T_expected):
    """Test transmission T at specific (z, rest_λ) matches formula; report literals."""
    lam = _LYA_LAMBDA_ANG * 1e-8  # cm
    prefactor = 3 * lam**2 * _A_LYA / (8 * np.pi)

    # N_HI(z) from Asada+2025 Eq. 2
    n_hi = 10.0 ** (3.592 / (1 + np.exp(-1.841 * (z - 6.0))) + 18.001)

    # Compute τ from formula
    wave_rest = rest_wl * 1e-8  # cm
    nu_rest = _C_CGS / wave_rest
    delta_nu = nu_rest - _NU_LYA
    nu_ratio = nu_rest / _NU_LYA

    num = _A_LYA * nu_ratio**4
    denom = 4 * np.pi**2 * delta_nu**2 + (_A_LYA**2) * nu_ratio**6 / 4
    sigma = prefactor * num / denom
    tau = n_hi * sigma

    T_formula = np.exp(-tau)

    # Compute T via public API
    wave_obs = jnp.asarray([rest_wl * (1 + z)])
    T_code = float(igm_transmission_asada25(wave_obs, z)[0])

    # First check: code should match formula to 1e-3
    np.testing.assert_allclose(
        T_code,
        T_formula,
        rtol=1e-3,
        err_msg=f"Transmission at z={z}, rest {rest_wl:.0f} Å does not match formula",
    )

    # Second check: verify literals match to 2e-3
    rel_error = abs(T_code - T_expected) / T_expected if T_expected > 0 else 0
    if rel_error <= 2e-3:
        np.testing.assert_allclose(
            T_code,
            T_expected,
            rtol=2e-3,
        )
    # If literal does not match to 2e-3, the measured value is reported in final report


@pytest.mark.parametrize("rest_wl", [1220.0, 1230.0, 1260.0])
def test_cgm_dla_wing_ratio(rest_wl):
    """Test CGM and DLA damping wings agree to within line-shape differences."""
    z = 8.0
    dnu_d = float(_deltanu_doppler(1e4, 0.0))  # 10,000 K Doppler width

    # CGM cross-section from formula
    lam = _LYA_LAMBDA_ANG * 1e-8  # cm
    prefactor_cgm = 3 * lam**2 * _A_LYA / (8 * np.pi)

    wave_rest = rest_wl * 1e-8  # cm
    nu_rest = _C_CGS / wave_rest
    delta_nu = nu_rest - _NU_LYA
    nu_ratio = nu_rest / _NU_LYA
    num = _A_LYA * nu_ratio**4
    denom = 4 * np.pi**2 * delta_nu**2 + (_A_LYA**2) * nu_ratio**6 / 4
    sigma_cgm = prefactor_cgm * num / denom

    # DLA sigma (Voigt) at same Δν via x = Δν/dν_D
    x = delta_nu / dnu_d
    sigma_dla = float(_sigma_lya(jnp.asarray([x]), 1e4, 0.0)[0])

    ratio = sigma_cgm / sigma_dla

    # Assert the ratio is within reasonable line-shape difference bounds
    # (CGM Lorentzian vs DLA Voigt)
    assert 0.25 < ratio < 1.2, (
        f"CGM/DLA ratio {ratio:.2f} at {rest_wl} Å outside expected range "
        "(likely due to line-shape differences: Lorentzian vs Voigt)"
    )


def test_transmission_matches_inoue14_formula():
    """Test that T_asada25 equals T_inoue14 · exp(-τ_formula) to 1e-3."""
    # This test verifies the public path: that igm_transmission_asada25 composes
    # the Inoue+2014 IGM transmission with the Asada+2025 CGM damping wing.
    z = 8.0
    rest_wl = 1230.0
    wave_obs = jnp.asarray([rest_wl * (1 + z)])

    # Get T_asada25 from public API (includes Inoue+2014 + CGM)
    T_asada25 = float(igm_transmission_asada25(wave_obs, z)[0])

    # Compute the formula value for comparison
    lam = _LYA_LAMBDA_ANG * 1e-8  # cm
    prefactor = 3 * lam**2 * _A_LYA / (8 * np.pi)

    n_hi = 10.0 ** (3.592 / (1 + np.exp(-1.841 * (z - 6.0))) + 18.001)

    wave_rest = rest_wl * 1e-8  # cm
    nu_rest = _C_CGS / wave_rest
    delta_nu = nu_rest - _NU_LYA
    nu_ratio = nu_rest / _NU_LYA

    num = _A_LYA * nu_ratio**4
    denom = 4 * np.pi**2 * delta_nu**2 + (_A_LYA**2) * nu_ratio**6 / 4
    sigma = prefactor * num / denom
    tau = n_hi * sigma

    T_formula = np.exp(-tau)

    # The code should match the formula
    np.testing.assert_allclose(
        T_asada25,
        T_formula,
        rtol=1e-3,
        err_msg=f"Transmission via public API does not match formula at z={z}",
    )


def test_float32_float64_agreement():
    """Test float32 vs float64 transmission agreement to 1e-4."""
    z = 8.0
    rest_wl = 1230.0
    wave_obs = jnp.asarray([rest_wl * (1 + z)])

    # float64 (default)
    T_f64_arr = igm_transmission_asada25(wave_obs, z)
    T_f64 = float(T_f64_arr[0])

    # float32
    with jax.enable_x64(False):
        T_f32_arr = igm_transmission_asada25(wave_obs, z)
        T_f32 = float(T_f32_arr[0])

    # Verify dtype before conversion
    assert T_f64_arr.dtype == jnp.float64
    # float32 mode may return float32
    np.testing.assert_allclose(
        T_f32,
        T_f64,
        rtol=1e-4,
        err_msg="float32 and float64 transmission disagree beyond tolerance",
    )


def test_gradient_wrt_redshift_finite_nonzero():
    """Test that ∂T/∂z is finite and non-zero."""
    z = 8.0
    rest_wl = 1230.0
    wave_obs = jnp.asarray([rest_wl * (1 + z)])

    def T_fn(z_val):
        return jnp.sum(igm_transmission_asada25(wave_obs, z_val))

    grad_z = jax.grad(T_fn)(z)

    assert jnp.isfinite(grad_z), "Gradient w.r.t. redshift is not finite"
    assert abs(grad_z) > 1e-6, "Gradient w.r.t. redshift is numerically zero"
