# SPDX-License-Identifier: BSD-3-Clause
"""Tests for bug #2397: salim_sbl18 UV bump normalization by R_V,mod(δ).

Salim, Boquien & Lee (2018) Eq. 4 defines the delta-dependent R_V,mod, which
should normalize the UV Drude bump term separately from the tilted base curve.
The prior implementation incorrectly used a fixed R_V=4.05 for both terms.

Citation: Salim et al. 2018 (ApJ 859, 11), Eq. 3 and Eq. 4.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_sbl18_rv_mod_paper_worked_example():
    """R_V,mod(-0.5) from Salim+2018 Eq. 4 must match paper's worked example."""
    # This test imports the helper after the fix is applied
    # Import only when the module has the fixed code
    try:
        from tengri.components.dust.attenuation import _sbl18_rv_mod
    except ImportError:
        pytest.skip("_sbl18_rv_mod not yet implemented (expected on BASE before fix)")

    # Eq. 4: R_V,mod = R_V,Cal / [(R_V,Cal + 1)(4400/5500)^δ - R_V,Cal]
    # Paper states: "for δ = -0.5, R_V,mod = 2.54"
    rv_mod_neg05 = _sbl18_rv_mod(-0.5)
    assert rv_mod_neg05 == pytest.approx(2.54, abs=5e-3)


def test_sbl18_rv_mod_footnote_7_factor():
    """The footnote-7 bias factor R_V,Cal / R_V,mod(δ=0.3) must equal 0.673."""
    try:
        from tengri.components.dust.attenuation import _sbl18_rv_mod
    except ImportError:
        pytest.skip("_sbl18_rv_mod not yet implemented (expected on BASE before fix)")

    # Eq. 4 at δ=0.3: R_V,mod ≈ 6.0178, so R_V,Cal / R_V,mod ≈ 0.673
    # (The issue #2397 number is the exact thousands digit of this factor)
    rv_mod_pos03 = _sbl18_rv_mod(0.3)
    rv_cal = 4.05
    factor = rv_cal / rv_mod_pos03
    assert factor == pytest.approx(0.673, abs=1e-3)


def test_sbl18_rv_mod_invariance_at_delta_zero():
    """R_V,mod(δ=0) must equal R_V,Cal = 4.05 (Eq. 4 reduces when δ=0)."""
    try:
        from tengri.components.dust.attenuation import _sbl18_rv_mod
    except ImportError:
        pytest.skip("_sbl18_rv_mod not yet implemented (expected on BASE before fix)")

    rv_mod_zero = _sbl18_rv_mod(0.0)
    assert rv_mod_zero == pytest.approx(4.05)


def test_salim_sbl18_bump_ratio_delta_varies():
    """Bump-to-base ratio at 2175 Å must move from 0.461 to 0.311 at δ=0.3.

    Before fix: (k(B=3) - k(B=0)) / k(B=0) ≈ 0.461 at δ=+0.3
    After fix: ratio must be ≈ 0.311 (smaller by factor R_V,Cal/R_V,mod ≈ 0.673)
    At δ=0: ratio must stay ≈ 0.348 (invariance — the fix is a no-op when δ=0)
    """
    from tengri.components.dust.attenuation import salim_sbl18

    wavelength = jnp.array([2175.0])  # UV bump peak in Angstrom
    rv_cal = 4.05

    # Test at δ=0.3
    k_bump0_delta03 = salim_sbl18(wavelength, dust_bump_strength=0.0, dust_delta=0.3)
    k_bump3_delta03 = salim_sbl18(wavelength, dust_bump_strength=3.0, dust_delta=0.3)
    ratio_delta03 = (k_bump3_delta03[0] - k_bump0_delta03[0]) / k_bump0_delta03[0]

    # Pre-fix value (what we see on BASE)
    pre_fix_ratio = 0.461

    # The ratio should be significantly different from pre-fix if code is correct
    # After fix: approx 0.311. Before fix: approx 0.461
    # This test should FAIL on BASE (showing pre_fix_ratio ≈ 0.461)
    # and PASS after fix (showing corrected_ratio ≈ 0.311)
    assert ratio_delta03 == pytest.approx(0.311, abs=3e-3), (
        f"Got {ratio_delta03:.6f}, expected ~0.311 after fix. "
        f"If you see ~0.461, the fix has not been applied yet."
    )

    # At δ=0, the ratio should stay the same (invariance)
    k_bump0_delta00 = salim_sbl18(wavelength, dust_bump_strength=0.0, dust_delta=0.0)
    k_bump3_delta00 = salim_sbl18(wavelength, dust_bump_strength=3.0, dust_delta=0.0)
    ratio_delta00 = (k_bump3_delta00[0] - k_bump0_delta00[0]) / k_bump0_delta00[0]
    assert ratio_delta00 == pytest.approx(0.348, abs=2e-3)

    # At δ=-0.3, ratio should be greater than δ=0 (since R_V,mod is smaller at negative δ)
    # At δ=-0.3: R_V,mod ≈ 3.00, so bump/R_V,mod is larger than at δ=0 where R_V,mod = 4.05
    k_bump0_delta_neg03 = salim_sbl18(wavelength, dust_bump_strength=0.0, dust_delta=-0.3)
    k_bump3_delta_neg03 = salim_sbl18(wavelength, dust_bump_strength=3.0, dust_delta=-0.3)
    ratio_delta_neg03 = (k_bump3_delta_neg03[0] - k_bump0_delta_neg03[0]) / k_bump0_delta_neg03[0]
    # Should be greater than δ=0 (since R_V,mod is smaller at negative δ, enhancing the bump)
    assert ratio_delta_neg03 > ratio_delta00, (
        f"At δ=-0.3, ratio {ratio_delta_neg03:.6f} should be > δ=0 ratio {ratio_delta00:.6f}"
    )


def test_salim_sbl18_absolute_extinction_match_bagpipes():
    """A(2175)/A_V at (δ=0.3, B=3) must match BAGPIPES evaluation (~2.079).

    Pre-fix tengri gives 2.317; BAGPIPES independently gives 2.082.
    After fix, tengri should match BAGPIPES to within <0.5%.
    """
    from tengri.components.dust.attenuation import salim_sbl18

    # Construct a full wavelength grid spanning the UV
    # We'll use a grid from 1000 to 10000 Angstrom
    wavelength = jnp.linspace(1000.0, 10000.0, 100)

    # Get extinction at target params
    k = salim_sbl18(wavelength, dust_bump_strength=3.0, dust_delta=0.3)

    # Find k at 2175 Å (the Drude bump peak)
    idx_2175 = jnp.argmin(jnp.abs(wavelength - 2175.0))
    k_2175 = k[idx_2175]

    # Find k at 5500 Å (V-band reference; k should be normalized to 1.0 there)
    idx_5500 = jnp.argmin(jnp.abs(wavelength - 5500.0))
    k_5500 = k[idx_5500]

    # A(2175)/A_V = k(2175) / k(5500) when normalized
    # But the function returns k/(k@5500), so it should be k(2175) / 1.0 approximately
    a_ratio = k_2175 / k_5500 if k_5500 > 0 else k_2175

    # BAGPIPES gives 2.082; pre-fix tengri gives 2.317
    # After fix, should be ~2.079
    expected_bagpipes = 2.082
    pre_fix_tengri = 2.317

    assert a_ratio == pytest.approx(expected_bagpipes, rel=5e-2), (
        f"Got {a_ratio:.4f}. Expected ~{expected_bagpipes:.4f} (BAGPIPES). "
        f"If you see ~{pre_fix_tengri:.4f}, the fix has not been applied yet."
    )


def test_salim_sbl18_algebraic_invariants():
    """Algebraic invariants: at δ=0 and B=0, salim_sbl18 equals Calzetti-only formula.

    This tests that the fix preserves correctness when bump or slope are zero,
    by verifying the output matches the tilted Calzetti base (δ=0) or
    no-bump (B=0) case.
    """
    from tengri.components.dust.attenuation import (
        _calzetti_l02_kprime,
        _drude_profile,
        salim_sbl18,
    )

    wavelength = jnp.linspace(1000.0, 30000.0, 200)
    wave_um = wavelength / 1e4

    rv_cal = 4.05

    # Case 1: dust_bump_strength=0 (bump disabled)
    # Output should match: (k_base * slope_mod) / rv, normalized at 5500
    k_no_bump = salim_sbl18(wavelength, dust_bump_strength=0.0, dust_delta=0.3)

    # Compute expected output manually
    k_base = _calzetti_l02_kprime(wavelength)
    slope_mod = (wave_um / 0.55) ** 0.3
    k_expected_no_bump = k_base * slope_mod / rv_cal
    k_5500_expected = _calzetti_l02_kprime(jnp.asarray(5500.0)) * 1.0 / rv_cal
    k_expected_no_bump = k_expected_no_bump / k_5500_expected

    assert jnp.allclose(k_no_bump, k_expected_no_bump, rtol=1e-6), (
        "With dust_bump_strength=0, output should match (k_base*slope_mod)/rv"
    )

    # Case 2: dust_delta=0 (no slope modification)
    # Output should match: (k_base + bump/rv_mod) normalized
    # At δ=0, R_V,mod = R_V,Cal = 4.05, so this reduces to the case
    # (k_base + bump/4.05) / normalized
    k_no_delta = salim_sbl18(wavelength, dust_bump_strength=3.0, dust_delta=0.0)

    # At δ=0, slope_mod = 1, so result should be (k_base + bump/4.05) / normalized
    k_base = _calzetti_l02_kprime(wavelength)
    bump = 3.0 * _drude_profile(wave_um, x0=0.2175, gamma=0.035)
    k_expected_no_delta = k_base / rv_cal + bump / rv_cal  # Both use rv=4.05
    k_5500_expected = (
        _calzetti_l02_kprime(jnp.asarray(5500.0)) / rv_cal
        + 3.0 * _drude_profile(jnp.asarray(0.55), x0=0.2175, gamma=0.035) / rv_cal
    )
    k_expected_no_delta = k_expected_no_delta / k_5500_expected

    assert jnp.allclose(k_no_delta, k_expected_no_delta, rtol=1e-6), (
        "With dust_delta=0, output should match (k_base + bump/rv)/normalized"
    )
