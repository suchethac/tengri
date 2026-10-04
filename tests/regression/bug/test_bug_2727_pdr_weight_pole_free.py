# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2727: `_pdr_luminosity_weight` without pole window.

Bug: The power-law to single-U luminosity ratio R(α) had 1e-3-width windows at
α = 1, 2 selecting exact limit forms. Inside those windows R is constant in α,
yielding R(α=1±δ) = R(α=1) and R(α=2±δ) = R(α=2), with zero gradient and
discontinuous jumps at the window edges (±5-8e-3). This creates likelihood
plateaus and gradient discontinuities for free α parameters, making HMC
trajectories diverge and causing optimizers to stall.

Fix: Implement the closed form without any window:
    L = ln(umax / umin);  t = 1 − alpha;  s = 2 − alpha
    R = g(s*L) / g(t*L),   g(u) = expm1(u) / u,  g(0) = 1

Use a Taylor series for |u| < 1e-3 to avoid division by zero in gradients.
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from tengri.components.dust.emission_templates import _pdr_luminosity_weight as R
except ImportError:
    R = None

pytestmark = pytest.mark.regression_bug


def _exact_formula(umin, umax, alpha):
    """Reference formula using numpy float64."""
    x = float(umax) / float(umin)
    a = float(alpha)
    if abs(a - 1.0) < 1e-10:
        return (x - 1.0) / np.log(x)
    if abs(a - 2.0) < 1e-10:
        return x * np.log(x) / (x - 1.0)
    return ((1.0 - a) / (2.0 - a)) * ((x ** (2.0 - a)) - 1.0) / ((x ** (1.0 - a)) - 1.0)


@pytest.mark.skipif(R is None, reason="tengri not installed")
def test_pdr_weight_value_dense_grid():
    """Test R(umin, umax, alpha) against exact formula on dense α grid."""
    alpha_grid = np.arange(0.5, 3.01, 0.01)
    # Umin/Umax pairs from the issue
    test_cases = [
        (1.0, 1e7),
        (0.1, 1e7),
        (25.0, 1e7),
        (1.0, 1e6),
    ]

    for umin, umax in test_cases:
        for alpha in alpha_grid:
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected) if expected != 0 else 0
            assert rel_err < 1e-12, (
                f"R({umin}, {umax}, {alpha}) = {computed}, expected {expected}, "
                f"relative error {rel_err:.2e} > 1e-12"
            )


@pytest.mark.skipif(R is None, reason="tengri not installed")
def test_pdr_weight_value_near_poles():
    """Test R at pole ± small offsets (exact limit forms)."""
    pole_offsets = [1e-7, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 1e-2]
    test_cases = [
        (1.0, 1e7),
        (0.1, 1e7),
        (25.0, 1e7),
        (1.0, 1e6),
    ]

    for umin, umax in test_cases:
        # Pole at α = 1
        for offset in pole_offsets:
            alpha = 1.0 + offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            # Tolerance 1e-10 accounts for float64 rounding in log/expm1/division chain
            assert rel_err < 1e-10, (
                f"R(pole α=1): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )

        for offset in pole_offsets:
            alpha = 1.0 - offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            assert rel_err < 1e-10, (
                f"R(pole α=1−): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )

        # Pole at α = 2
        for offset in pole_offsets:
            alpha = 2.0 + offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            assert rel_err < 1e-10, (
                f"R(pole α=2): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )

        for offset in pole_offsets:
            alpha = 2.0 - offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            assert rel_err < 1e-10, (
                f"R(pole α=2−): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )


@pytest.mark.skipif(R is None, reason="tengri not installed")
def test_pdr_weight_gradient():
    """Test jax.grad(R) against central difference at α sampling points."""
    umin, umax = 1.0, 1e7
    alpha_points = [0.7, 0.9995, 1.0, 1.0005, 1.5, 1.9995, 2.0, 2.0005, 2.5]
    h = 1e-6

    for alpha in alpha_points:
        # JAX gradient
        jax_grad_val = float(jax.grad(lambda a: R(umin, umax, a))(alpha))

        # Central difference (high precision reference)
        fwd = _exact_formula(umin, umax, alpha + h)
        bwd = _exact_formula(umin, umax, alpha - h)
        central_diff = (fwd - bwd) / (2.0 * h)

        rel_err = abs(jax_grad_val - central_diff) / abs(central_diff) if central_diff != 0 else 0
        assert rel_err < 1e-6, (
            f"∂R/∂α at α={alpha}: jax.grad={jax_grad_val:.6e}, "
            f"central_diff={central_diff:.6e}, rel_err={rel_err:.2e}"
        )
        assert jax_grad_val != 0, (
            f"∂R/∂α at α={alpha} is zero: gradient is discontinuous or undefined"
        )


@pytest.mark.skipif(R is None, reason="tengri not installed")
def test_pdr_weight_continuity_across_old_windows():
    """Test continuity across the old 1e-3 window edges."""
    test_cases = [
        (1.0, 1e7),
        (0.1, 1e7),
        (25.0, 1e7),
        (1.0, 1e6),
    ]

    for umin, umax in test_cases:
        # Window edges at α = 1 ± 1e-3
        alpha_edge = 1.0 - 1e-3
        val_left = float(R(umin, umax, alpha_edge - 1e-9))
        val_right = float(R(umin, umax, alpha_edge + 1e-9))
        rel_jump = abs(val_left - val_right) / abs(val_right)
        assert rel_jump < 1e-7, (
            f"Jump at α=1−1e-3 edge: left={val_left}, right={val_right}, "
            f"relative={rel_jump:.2e}"
        )

        # Window edges at α = 2 ± 1e-3
        alpha_edge = 2.0 - 1e-3
        val_left = float(R(umin, umax, alpha_edge - 1e-9))
        val_right = float(R(umin, umax, alpha_edge + 1e-9))
        rel_jump = abs(val_left - val_right) / abs(val_right)
        assert rel_jump < 1e-7, (
            f"Jump at α=2−1e-3 edge: left={val_left}, right={val_right}, "
            f"relative={rel_jump:.2e}"
        )


@pytest.mark.skipif(R is None, reason="tengri not installed")
def test_pdr_weight_float32():
    """Test float32 precision: values within 4 ulp, gradient non-zero and finite."""
    alpha_test = [1.0, 2.0, 1.0005]
    umin, umax = 1.0, 1e7

    for alpha in alpha_test:
        # Compute in float64
        val_f64 = float(R(float(umin), float(umax), float(alpha)))

        # Compute in float32
        with jax.enable_x64(False):
            val_f32 = float(R(np.float32(umin), np.float32(umax), np.float32(alpha)))

        # ulp tolerance: 4 ulp of the float32 value
        ulp_tol = 4.0 * np.spacing(np.float32(abs(val_f64)))
        abs_err = abs(val_f32 - val_f64)
        assert abs_err < ulp_tol, (
            f"float32 value at α={alpha}: f32={val_f32}, f64={val_f64}, "
            f"error={abs_err:.2e} > 4ulp={ulp_tol:.2e}"
        )

        # Gradient must be finite and non-zero in float32
        with jax.enable_x64(False):
            grad_f32 = float(jax.grad(lambda a: R(np.float32(umin), np.float32(umax), a))(
                np.float32(alpha)
            ))
        assert np.isfinite(grad_f32), f"float32 gradient at α={alpha} is not finite"
        assert grad_f32 != 0, f"float32 gradient at α={alpha} is zero"


@pytest.mark.skipif(R is None, reason="tengri not installed")
def test_pdr_weight_sed_mix_outside_windows():
    """Test that SED mix does not move outside the old window regions."""
    # DL07 call at α = 2 exactly (emission_templates.py:242)
    umin_dl07_cases = [0.1, 1.0, 25.0]
    umax_dl07 = 1e6
    alpha_dl07 = 2.0

    for umin in umin_dl07_cases:
        computed = float(R(umin, umax_dl07, alpha_dl07))
        expected = _exact_formula(umin, umax_dl07, alpha_dl07)
        rel_err = abs(computed - expected) / abs(expected)
        assert rel_err < 1e-12, (
            f"DL07 (α=2): R({umin}, {umax_dl07}, {alpha_dl07}) should match exact form, "
            f"rel_err={rel_err:.2e}"
        )

    # DL14 call at α = 1.5 (emission_templates.py:470, a value inside the old windows)
    umin_dl14_cases = [0.1, 1.0, 25.0]
    umax_dl14 = 1e7
    alpha_dl14 = 1.5

    for umin in umin_dl14_cases:
        computed = float(R(umin, umax_dl14, alpha_dl14))
        expected = _exact_formula(umin, umax_dl14, alpha_dl14)
        rel_err = abs(computed - expected) / abs(expected)
        assert rel_err < 1e-12, (
            f"DL14 (α=1.5): R({umin}, {umax_dl14}, {alpha_dl14}) should match exact form, "
            f"rel_err={rel_err:.2e}"
        )
