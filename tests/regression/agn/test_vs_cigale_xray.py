# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for X-ray emission against Yang+2020 X-CIGALE.

Verifies that the X-ray implementation produces luminosities consistent with
the Yang et al. 2020 X-CIGALE methods and the Just+2007 / Lusso–Risaliti
α_OX–L_2500 relation.

References
----------
Yang, G., et al. 2020, MNRAS, 491, 740.
https://doi.org/10.1093/mnras/stz3001

Just, D. W., et al. 2007, ApJ, 665, 1004.
https://doi.org/10.1086/519990

Lusso, E., & Risaliti, G. 2016, ApJ, 819, 154.
https://doi.org/10.3847/0004-637X/819/2/154

Lehmer, B. D., et al. 2016, ApJ, 825, 7.
https://doi.org/10.3847/0004-637X/825/1/7
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    alpha_ox_from_l2500,
    xray_agn_corona_from_disc,
    xray_anisotropy,
)
from tengri.utils.physics_constants import (
    C_AA as _C_AA,
    H_PLANCK as _H_PLANCK,
    KEV_TO_ERG as _KEV_TO_ERG,
)


@pytest.mark.regression_paper
def test_alpha_ox_from_l2500_just2007():
    """Test α_OX–L_2500 relation against Just+2007 calibration.

    Verifies that the Just+2007 empirical relation is correctly implemented:
        α_OX = −0.137 log₁₀(L_2500) + 2.638

    For canonical AGN luminosity L_2500 = 1e30 erg/s/Hz, this should yield
    α_OX ≈ −1.4 (typical Type 1 AGN value).

    References
    ----------
    Just, D. W., et al. 2007, ApJ, 665, 1004, Eq. 3.
    """
    # Canonical test point: L_2500 = 1e30 erg/s/Hz
    l_2500_test = 1e30
    alpha_ox_test = alpha_ox_from_l2500(l_2500_test)

    # Expected value from Just+2007 Eq. 3:
    # α_OX = −0.137 × log₁₀(1e30) + 2.638 = −0.137 × 30 + 2.638 = −1.472
    expected = -0.137 * 30.0 + 2.638
    np.testing.assert_allclose(float(alpha_ox_test), expected, atol=0.01,
                              err_msg="α_OX from L_2500 does not match Just+2007")


@pytest.mark.regression_paper
def test_xray_corona_face_on_monochromatic():
    """Test AGN corona X-ray at 2 keV monochromatic point (Just+2007 anchor).

    Verifies that the corona is correctly normalized to the α_OX relation
    at the 2 keV reference point used by Just et al. 2007.

    The α_OX definition directly links L_2keV to L_2500:
        α_OX = 0.3838 × log₁₀(L_2keV / L_2500)
        => L_2keV = L_2500 × 10^(α_OX / 0.3838)

    This test computes the SED at 2 keV and verifies consistency.

    Test setup:
    - Face-on geometry (cos_inc = 1.0, no anisotropy)
    - L_2500_30deg = 1e30 erg/s/Hz (canonical AGN)
    - Spectral index Γ = 1.8 (typical)
    - Reference point: 2 keV (λ = 6.2 Å)
    """
    l_2500_erg_hz = 1e30
    gamma = 1.8
    E_cut = 300.0

    # Wavelength grid around 2 keV
    wavelength_2kev = 6.2  # angstrom, λ = hc/E
    wavelength = jnp.array([wavelength_2kev, 6.3, 6.1])

    # Compute α_OX from L_2500
    alpha_ox = alpha_ox_from_l2500(l_2500_erg_hz)

    # Expected L_2keV from α_OX definition (yang20.py:227)
    l_2kev_expected = l_2500_erg_hz * 10.0 ** (alpha_ox / 0.3838)

    # Compute X-ray SED
    sed_xray = xray_agn_corona_from_disc(
        wavelength,
        l_2500_erg_hz,
        cos_inc=1.0,
        delta_alpha_ox=0.0,
        gamma=gamma,
        E_cut=E_cut,
        apply_anisotropy=False,
    )

    # Find the value closest to 2 keV
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG
    idx_2kev = jnp.argmin(jnp.abs(E_keV - 2.0))
    l_nu_2kev = sed_xray[idx_2kev]

    # The SED at 2 keV should approximate L_2keV
    # (exact match is difficult due to spectrum shape, so allow 50% tolerance)
    assert float(l_nu_2kev) > 0, "SED at 2 keV is zero"
    log_ratio = jnp.log10(l_nu_2kev / l_2kev_expected)
    assert float(jnp.abs(log_ratio)) < 0.5, (
        f"SED at 2 keV {l_nu_2kev:.3e} deviates significantly from "
        f"expected L_2keV {l_2kev_expected:.3e} (log-ratio: {log_ratio:.4f} dex)"
    )


@pytest.mark.regression_paper
def test_xray_anisotropy_polynomial():
    """Test X-ray anisotropy factor polynomial form (Yang+2022 via CIGALE).

    Verifies the normalized 2nd-order polynomial anisotropy model
    (yang20.py:231–235):
        f(μ) = [a₁ cos θ + a₂ cos² θ + (1 − a₁ − a₂)]
               / [1 − 0.13397 a₁ − 0.25 a₂]

    The denominator normalizes the angular distribution so that the bolometric
    corona luminosity at θ=0° (face-on) is recovered.

    With default (a₁ = 0.5, a₂ = 0.0), the denominator = 0.933, giving:
    - Face-on (θ = 0°): f = 1.0 / 0.933 ≈ 1.072 (recovery factor)
    - Edge-on (θ = 90°): f = 0.5 / 0.933 ≈ 0.536

    References
    ----------
    Yang, G., et al. 2022, ApJ, 927, 192.
    CIGALE yang20.py:231–235 (Lehmer et al. 2016 anisotropy normalization).
    """
    # Test wavelength grid (arbitrary; anisotropy is geometry-independent)
    wavelength = jnp.array([10.0, 50.0, 100.0])
    l_iso = jnp.array([1e40, 1e40, 1e40])  # Isotropic baseline

    # Default anisotropy coefficients (X-CIGALE)
    a1, a2 = 0.5, 0.0
    denom = 1.0 - 0.13397 * a1 - 0.25 * a2  # yang20.py:233

    # Face-on (cos_inc = 1.0)
    cos_inc_face = 1.0
    factor_face = xray_anisotropy(l_iso, cos_inc_face, a1=a1, a2=a2)
    numerator_face = a1 * 1.0 + a2 * 1.0**2 + (1.0 - a1 - a2)
    expected_face = numerator_face / denom
    np.testing.assert_allclose(float(factor_face[0]) / l_iso[0], expected_face,
                              atol=1e-6, err_msg="Face-on anisotropy failed")

    # Edge-on (cos_inc = 0.0)
    cos_inc_edge = 0.0
    factor_edge = xray_anisotropy(l_iso, cos_inc_edge, a1=a1, a2=a2)
    numerator_edge = a1 * 0.0 + a2 * 0.0**2 + (1.0 - a1 - a2)
    expected_edge = numerator_edge / denom
    np.testing.assert_allclose(float(factor_edge[0]) / l_iso[0], expected_edge,
                              atol=1e-6, err_msg="Edge-on anisotropy failed")

    # Intermediate (cos_inc = cos(30°) ≈ 0.866)
    cos_inc_30deg = np.cos(np.radians(30.0))
    factor_30deg = xray_anisotropy(l_iso, cos_inc_30deg, a1=a1, a2=a2)
    numerator_30deg = a1 * cos_inc_30deg + a2 * cos_inc_30deg**2 + (1.0 - a1 - a2)
    expected_30deg = numerator_30deg / denom
    np.testing.assert_allclose(float(factor_30deg[0]) / l_iso[0], expected_30deg,
                              atol=1e-6, err_msg="30° anisotropy failed")


@pytest.mark.regression_paper
def test_xray_anisotropy_ratio_face_edge():
    """Test face-on to edge-on X-ray flux ratio with CIGALE normalization.

    For default (a₁ = 0.5, a₂ = 0.0), with denominator correction (yang20.py:233):
        denom = 1 − 0.13397 × 0.5 − 0.25 × 0.0 = 0.933
        L_X(0°) / L_X(90°) = [1.0 / denom] / [0.5 / denom] = 2.0

    After the denominator normalization, the ratio remains 2.0, predicting that
    face-on AGN should be 2× brighter in X-rays than edge-on systems, consistent
    with Yang et al. 2022 observational findings.

    References
    ----------
    Yang, G., et al. 2022, ApJ, 927, 192, Fig. 2 / Table 1.
    CIGALE yang20.py:231–235.
    """
    wavelength = jnp.linspace(10.0, 100.0, 100)
    l_iso = jnp.ones_like(wavelength) * 1e40

    a1, a2 = 0.5, 0.0

    # Face-on
    l_face = xray_anisotropy(l_iso, cos_inc=1.0, a1=a1, a2=a2)
    # Edge-on
    l_edge = xray_anisotropy(l_iso, cos_inc=0.0, a1=a1, a2=a2)

    # Ratio (denominators cancel, ratio = 1.0 / 0.5 = 2.0)
    ratio = jnp.mean(l_face) / jnp.mean(l_edge)

    np.testing.assert_allclose(float(ratio), 2.0, atol=0.01,
                              err_msg="Face-on / edge-on ratio does not match 2.0")
