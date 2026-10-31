# SPDX-License-Identifier: BSD-3-Clause
"""Fitzpatrick (1999) Milky Way extinction law against its published tables.

Paper: Fitzpatrick 1999, PASP 111, 63 (DOI 10.1086/316293). Table 3 lists the
R = 3.1 cubic-spline anchors as A(lambda)/E(B-V); Table 4 gives the R-dependent
optical anchors. The UV branch is the Fitzpatrick & Massa (1990, ApJS 72, 163,
DOI 10.1086/191413) Eqs. (2)-(4), so the 2700 and 2600 Angstrom anchors check
the UV form and its coefficients together.

Tolerances: every tabulated Table 3 wavelength to 1% (relative); the UV joint at
2700 Angstrom continuous to 1e-3 in A(lambda)/E(B-V); the two UV anchors to
0.01 absolute; Table 4 optical anchors at R = 5 to 1e-6 relative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import fitzpatrick99, fitzpatrick99_unnormalized

pytestmark = pytest.mark.regression_paper

#: Table 3 (R = 3.1 anchors), wavelength in Angstrom and A(lambda)/E(B-V).
_TABLE3_WAVE_A = np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0, 2700.0, 2600.0])
_TABLE3_A_OVER_EBV = np.array([0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591])


def _a_over_ebv(wave_a: np.ndarray, rv: float) -> np.ndarray:
    """Paper-convention A(lambda)/E(B-V): the unnormalized curve times R_V."""
    return np.asarray(fitzpatrick99_unnormalized(jnp.asarray(wave_a), dust_Rv=rv)) * rv


def test_registered_law_is_unity_at_v_band_and_is_the_paper_curve_renormalized():
    """The registry contract is k(5500 A) = 1; the shape is the paper's curve over its V value."""
    wave_a = np.array([26500.0, 6000.0, 4110.0, 2600.0])
    v_value = float(np.asarray(fitzpatrick99_unnormalized(jnp.asarray([5500.0]), dust_Rv=3.1))[0])
    registered = np.asarray(fitzpatrick99(jnp.asarray(np.r_[wave_a, 5500.0]), dust_Rv=3.1))
    assert registered[-1] == pytest.approx(1.0, abs=1e-12)
    paper = np.asarray(fitzpatrick99_unnormalized(jnp.asarray(wave_a), dust_Rv=3.1))
    np.testing.assert_allclose(registered[:-1], paper / v_value, rtol=1e-12)


def test_table3_anchors_at_r31_to_one_percent():
    got = _a_over_ebv(_TABLE3_WAVE_A, 3.1)
    np.testing.assert_allclose(got, _TABLE3_A_OVER_EBV, rtol=0.01)


def test_uv_fm90_form_reproduces_2700_and_2600_anchors():
    got = _a_over_ebv(np.array([2700.0, 2600.0]), 3.1)
    np.testing.assert_allclose(got, [6.265, 6.591], atol=0.01)


def test_uv_optical_joint_is_continuous_at_2700():
    left = _a_over_ebv(np.array([2700.0 * (1.0 - 1e-6)]), 3.1)
    right = _a_over_ebv(np.array([2700.0 * (1.0 + 1e-6)]), 3.1)
    assert abs(left[0] - right[0]) < 1e-3


def test_optical_anchors_follow_table4_at_r5():
    rv = 5.0
    wave_a = np.array([6000.0, 5470.0, 4670.0, 4110.0])
    expected = np.array(
        [
            -0.426 + 1.0044 * rv,
            -0.050 + 1.0016 * rv,
            0.701 + 1.0016 * rv,
            1.208 + 1.0032 * rv - 0.00033 * rv**2,
        ]
    )
    np.testing.assert_allclose(_a_over_ebv(wave_a, rv), expected, rtol=1e-6)


def test_output_is_finite_and_positive_across_optical_to_uv():
    wave_a = np.linspace(1000.0, 30000.0, 400)
    k = np.asarray(fitzpatrick99(jnp.asarray(wave_a), dust_Rv=3.1))
    assert np.all(np.isfinite(k))
    assert np.all(k > 0.0)


def test_jit_matches_eager():
    wave_a = jnp.asarray(_TABLE3_WAVE_A)
    eager = np.asarray(fitzpatrick99(wave_a, dust_Rv=3.1))
    jitted = np.asarray(jax.jit(lambda w: fitzpatrick99(w, dust_Rv=3.1))(wave_a))
    np.testing.assert_allclose(jitted, eager, rtol=1e-12)


def test_gradient_with_respect_to_wavelength_is_finite_and_nonzero():
    grad = jax.grad(lambda w: fitzpatrick99(w, dust_Rv=3.1))(jnp.asarray(5000.0))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


@pytest.mark.parametrize("rv", [2.5, 5.0])
def test_gradient_with_respect_to_r_v_is_finite_and_nonzero(rv):
    grad = jax.grad(lambda r: jnp.sum(fitzpatrick99(jnp.asarray([4000.0, 2000.0]), dust_Rv=r)))(
        jnp.asarray(rv)
    )
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0
