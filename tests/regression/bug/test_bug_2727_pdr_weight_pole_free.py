# SPDX-License-Identifier: BSD-3-Clause
"""#2727: ``_pdr_luminosity_weight`` is the Draine & Li (2007) Eq. 33 ratio at every α.

R(α) = g((2 − α)L) / g((1 − α)L) with g(u) = expm1(u)/u, L = ln(U_max/U_min):
one closed form, continuous with a continuous gradient through α = 1 and 2.
The cells pin the value against a 50-digit reference, the gradient against a
central difference of that reference, continuity across α = 1 ± 1e-3 and
2 ± 1e-3, float32 agreement, and the two SED-mix calls (DL07 at α = 2, DL14).
"""

import jax
import mpmath
import numpy as np
import pytest

from tengri.components.dust.emission_templates import _pdr_luminosity_weight as R

pytestmark = pytest.mark.regression_bug


def _exact_formula(umin, umax, alpha):
    """Draine & Li (2007) Eq. 33 ratio at 50 significant digits.

    The textbook form ``(1-a)/(2-a) (x^(2-a)-1)/(x^(1-a)-1)`` cancels
    catastrophically within 1e-6 of either pole in float64 (``x^(1-a) - 1`` is
    ~1e-6 there), so the reference is evaluated in ``mpmath`` at 50 digits and
    rounded once; the limit forms are taken only at the exact poles.
    """
    with mpmath.workdps(50):
        x = mpmath.mpf(umax) / mpmath.mpf(umin)
        a = mpmath.mpf(alpha)
        if a == 1:
            value = (x - 1) / mpmath.log(x)
        elif a == 2:
            value = x * mpmath.log(x) / (x - 1)
        else:
            value = ((1 - a) / (2 - a)) * (x ** (2 - a) - 1) / (x ** (1 - a) - 1)
        return float(value)


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
            assert rel_err < 1e-12, (
                f"R(pole α=1): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )

        for offset in pole_offsets:
            alpha = 1.0 - offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            assert rel_err < 1e-12, (
                f"R(pole α=1−): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )

        # Pole at α = 2
        for offset in pole_offsets:
            alpha = 2.0 + offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            assert rel_err < 1e-12, (
                f"R(pole α=2): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )

        for offset in pole_offsets:
            alpha = 2.0 - offset
            computed = float(R(umin, umax, alpha))
            expected = _exact_formula(umin, umax, alpha)
            rel_err = abs(computed - expected) / abs(expected)
            assert rel_err < 1e-12, (
                f"R(pole α=2−): α={alpha} gives {computed}, expected {expected}, "
                f"rel_err {rel_err:.2e}"
            )


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

        assert np.isfinite(jax_grad_val), f"∂R/∂α at α={alpha} is not finite: {jax_grad_val}"
        assert central_diff != 0, f"reference ∂R/∂α at α={alpha} is zero: vacuous cell"
        rel_err = abs(jax_grad_val - central_diff) / abs(central_diff)
        assert rel_err < 1e-6, (
            f"∂R/∂α at α={alpha}: jax.grad={jax_grad_val:.6e}, "
            f"central_diff={central_diff:.6e}, rel_err={rel_err:.2e}"
        )
        assert jax_grad_val != 0, (
            f"∂R/∂α at α={alpha} is zero: the pole window of #2727 would hold R flat here"
        )


def test_pdr_weight_continuity_across_old_windows():
    """No step at the former window edges α = 1 ± 1e-3 and 2 ± 1e-3.

    R itself changes by ≈ 2δ·ln(U_max/U_min) relative across a step of 2δ in
    α, so continuity is stated as: the finite difference of R across each edge
    equals the finite difference of the 50-digit reference to 1e-12 of R. A
    step of the old size (5–8e-3) would miss by six orders of magnitude.
    """
    test_cases = [
        (1.0, 1e7),
        (0.1, 1e7),
        (25.0, 1e7),
        (1.0, 1e6),
    ]
    delta = 1e-9
    for umin, umax in test_cases:
        for edge in (1.0 - 1e-3, 1.0 + 1e-3, 2.0 - 1e-3, 2.0 + 1e-3):
            val_left = float(R(umin, umax, edge - delta))
            val_right = float(R(umin, umax, edge + delta))
            ref_left = _exact_formula(umin, umax, edge - delta)
            ref_right = _exact_formula(umin, umax, edge + delta)
            step_err = abs((val_right - val_left) - (ref_right - ref_left)) / abs(ref_right)
            assert step_err < 1e-12, (
                f"step across α={edge} (U_min={umin}): R changes by {val_right - val_left:.6e}, "
                f"the reference by {ref_right - ref_left:.6e}; mismatch {step_err:.2e} of R"
            )


def test_pdr_weight_float32():
    """float32: value within 8 ulp of float64, gradient finite and non-zero.

    The float32 chain is log, two multiplies, two expm1, two divisions and one
    series, each rounded once (≤ 0.5 ulp) and the series switch at |u| = 1e-3
    below float32 resolution: 8 ulp is the sum; measured ≤ 6.8 ulp at
    (U_min = 25, α = 1.0005), ≤ 2 ulp at U_min = 1.
    """
    alpha_test = [1.0, 2.0, 1.0005, 2.0005]
    cases = [(1.0, 1e7), (25.0, 1e7), (0.1, 1e6)]

    for (umin, umax), alpha in [(c, a) for c in cases for a in alpha_test]:
        # Compute in float64
        val_f64 = float(R(float(umin), float(umax), float(alpha)))

        # Compute in float32
        with jax.enable_x64(False):
            val_f32 = float(R(np.float32(umin), np.float32(umax), np.float32(alpha)))

        ulp_tol = 8.0 * np.spacing(np.float32(abs(val_f64)))
        abs_err = abs(val_f32 - val_f64)
        assert abs_err < ulp_tol, (
            f"float32 value at α={alpha}: f32={val_f32}, f64={val_f64}, "
            f"error={abs_err:.2e} > 8 ulp={ulp_tol:.2e} (U_min={umin})"
        )

        # Gradient must be finite and non-zero in float32
        with jax.enable_x64(False):
            grad_f32 = float(
                jax.grad(lambda a, lo=umin, hi=umax: R(np.float32(lo), np.float32(hi), a))(
                    np.float32(alpha)
                )
            )
        assert np.isfinite(grad_f32), f"float32 gradient at α={alpha} is not finite"
        assert grad_f32 != 0, f"float32 gradient at α={alpha} is zero"


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
