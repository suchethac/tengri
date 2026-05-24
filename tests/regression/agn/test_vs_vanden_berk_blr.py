# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for BLR emission against Vanden Berk+2001 composite.

Verifies that the BLR implementation produces line ratios consistent with
the Vanden Berk et al. (2001) SDSS composite quasar spectrum.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blr import (
    compute_blr_sed,
    _BLR_LINES,
    _BLR_FWHM_KMS,
)
from tengri.components.agn._phys import gaussian_line_profile


@pytest.mark.regression_paper
def test_blr_vanden_berk_line_ratios():
    """Test BLR line ratios derived from Vanden Berk+2001 SDSS composite.

    Verifies that key emission-line ratios are consistent with expectations
    from the Vanden Berk et al. (2001) SDSS composite quasar spectrum.
    Pins to line pairs at: Lyα, C IV, Mg II, H-α, H-β, and computes their
    relative strengths to within ±5 % tolerance.

    References
    ----------
    Vanden Berk, M. A., et al. 2001, AJ, 122, 549.
    https://doi.org/10.1086/321167
    """
    # Test parameters
    l_disc_bol_erg = 1e45  # 1e12 L_sun (moderate AGN)
    covering_fraction = 0.1
    fwhm_kms = _BLR_FWHM_KMS  # 5000 km/s (broad lines)
    fe2_strength = 0.0  # Disable Fe II for this test

    # Wavelength grid (cover UV to optical)
    wavelength = jnp.logspace(np.log10(1000), np.log10(8000), 3000)

    # Compute BLR spectrum
    sed = compute_blr_sed(
        wavelength=wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        fwhm_kms=fwhm_kms,
        agn_fe2_strength=fe2_strength,
        line_efficiency=0.08,
    )

    # Expected relative line strengths (from _BLR_LINES table).
    # These values are taken directly from the _BLR_LINES array, which is
    # calibrated to the Vanden Berk et al. (2001) SDSS composite quasar spectrum
    # and represent the expected line strength ratios when the BLR receives
    # disc luminosity. The test verifies that the computed spectrum produces
    # these ratios (after convolution and integration).
    expected_strengths = {
        "lya": 1.40,      # Lyα 1215.7
        "civ": 0.28,      # C IV 1549
        "mgii": 0.33,     # Mg II 2800
        "hbeta": 1.00,    # H-β 4863 (reference)
        "halpha": 2.15,   # H-α 6563
    }

    # Line centers (vacuum wavelengths)
    line_centers = {
        "lya": 1215.7,
        "civ": 1549.0,
        "mgii": 2799.9,
        "hbeta": 4862.7,
        "halpha": 6562.8,
    }

    # Measure peak fluxes by integrating line profiles
    measured_strengths = {}

    for line_name, line_wl in line_centers.items():
        # Create a Gaussian profile centered at this line
        profile = gaussian_line_profile(wavelength, line_wl, fwhm_kms)
        # Integrate SED × profile to get line flux
        _c_aa = 2.99792458e18  # c in Angstrom/s
        nu = _c_aa / jnp.maximum(wavelength, 1.0)
        sort_idx = jnp.argsort(nu)
        line_flux = jnp.abs(
            jnp.trapezoid((sed * profile)[sort_idx], nu[sort_idx])
        )
        measured_strengths[line_name] = line_flux

    # Normalize to H-beta
    hbeta_flux = jnp.maximum(measured_strengths["hbeta"], 1e-30)
    for line_name in measured_strengths:
        measured_strengths[line_name] = measured_strengths[line_name] / hbeta_flux

    # Assert strengths within ±5 % of expected (scaled by H-beta ratio)
    hbeta_expected = expected_strengths["hbeta"]
    for line_name, expected_strength in expected_strengths.items():
        measured_strength = measured_strengths[line_name]
        expected_normalized = expected_strength / hbeta_expected
        relative_error = (
            jnp.abs(measured_strength - expected_normalized) / expected_normalized
        )
        assert (
            relative_error < 0.05
        ), (
            f"{line_name}: expected {expected_normalized:.3f}, got {measured_strength:.3f} "
            f"({100*relative_error:.1f}% error)"
        )


@pytest.mark.regression_paper
def test_blr_line_count():
    """Verify that BLR line list has ≥25 lines as per Vanden Berk+2001."""
    n_lines = _BLR_LINES.shape[0]
    assert (
        n_lines >= 25
    ), f"Expected ≥25 BLR lines, got {n_lines}. "
    "Update _BLR_LINES with full Vanden Berk Table 2."


@pytest.mark.regression_paper
def test_blr_covers_uv_optical():
    """Verify that BLR line list spans UV to optical wavelengths."""
    min_wave = jnp.min(_BLR_LINES[:, 0])
    max_wave = jnp.max(_BLR_LINES[:, 0])

    # Should cover at least Lyα to H-α
    assert min_wave < 1250, f"Minimum wavelength {min_wave} should be <1250 Å (Lyα region)"
    assert max_wave > 6000, f"Maximum wavelength {max_wave} should be >6000 Å (H-α region)"


if __name__ == "__main__":
    test_blr_vanden_berk_line_ratios()
    test_blr_line_count()
    test_blr_covers_uv_optical()
    print("All BLR regression tests passed!")
