# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for NLR emission against Richardson+2014 template.

Verifies that the NLR implementation produces line ratios consistent with
the Richardson et al. (2014) emission-line diagnostics used as the canonical
NLR template in tengri.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn._phys import gaussian_line_profile
from tengri.components.agn.nlr import (
    _NLR_FWHM_KMS,
    compute_nlr_sed,
    compute_nlr_sed_richardson2014,
)


@pytest.mark.xfail(
    reason=(
        "Methodology bug shared with test_blr_vanden_berk_line_ratios: integrating "
        "`sed × gaussian_profile` over frequency gives ∫L_ν × profile², not the line "
        "flux. Need to switch to narrow-band L_ν integration. Tracked separately."
    ),
    strict=False,
)
@pytest.mark.regression_paper
def test_richardson_nlr_line_ratios():
    """Test NLR line ratios against Richardson+2014 'a42' template.

    Verifies that key emission-line ratios match the Richardson et al. (2014)
    template within ±5 % tolerance. The test pins to a representative set of
    lines: [O III] 5007, [O III] 4959, Hβ, [N II] 6584, Hα.

    References
    ----------
    Richardson et al. 2014, MNRAS, 437, 2376. Table 3, column 'a42'.
    """
    # Test parameters
    l_disc_bol_erg = 1e45  # 1e12 L_sun
    covering_fraction = 0.1
    fwhm_kms = _NLR_FWHM_KMS  # 500 km/s (narrow lines)
    line_efficiency = 0.10

    # Wavelength grid (high resolution in optical region)
    wavelength = jnp.logspace(np.log10(3000), np.log10(10000), 2000)

    # Compute NLR spectrum
    sed = compute_nlr_sed_richardson2014(
        wavelength=wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        fwhm_kms=fwhm_kms,
        line_efficiency=line_efficiency,
    )

    # Extract peak fluxes near key lines by integrating Gaussian profiles
    # The line ratios from Richardson+2014 'a42' table (normalized to Hβ = 1).
    # These values are taken directly from the _RICHARDSON_FLUXES array in nlr.py,
    # which embeds the Richardson et al. (2014) emission-line template.
    expected_ratios = {
        "oiii_5007": 8.53,  # [O III] 5007 / Hβ
        "oiii_4959": 2.87,  # [O III] 4959 / Hβ (ratio 5007/4959 = 2.98)
        "hbeta": 1.00,  # H-beta (reference)
        "nii_6584": 2.13,  # [N II] 6584 / Hβ
        "halpha": 2.86,  # H-α / Hβ
    }

    # Line centers (vacuum wavelengths)
    line_centers = {
        "oiii_5007": 5008.0,
        "oiii_4959": 4960.0,
        "hbeta": 4863.0,
        "nii_6584": 6585.0,
        "halpha": 6564.0,
    }

    # Measure peak fluxes by computing line profiles and integrating
    measured_ratios = {}
    hbeta_flux = None

    for line_name, line_wl in line_centers.items():
        # Create a Gaussian profile centered at this line
        profile = gaussian_line_profile(wavelength, line_wl, fwhm_kms)
        # Integrate SED × profile to get line flux
        _c_aa = 2.99792458e18  # c in Angstrom/s
        nu = _c_aa / jnp.maximum(wavelength, 1.0)
        sort_idx = jnp.argsort(nu)
        line_flux = jnp.abs(jnp.trapezoid((sed * profile)[sort_idx], nu[sort_idx]))

        if line_name == "hbeta":
            hbeta_flux = line_flux
        measured_ratios[line_name] = line_flux

    # Compute ratios relative to Hβ
    hbeta_flux = jnp.maximum(hbeta_flux, 1e-30)
    for line_name in measured_ratios:
        measured_ratios[line_name] = measured_ratios[line_name] / hbeta_flux

    # Assert ratios within ±5 %
    for line_name, expected_ratio in expected_ratios.items():
        measured_ratio = measured_ratios[line_name]
        relative_error = jnp.abs(measured_ratio - expected_ratio) / expected_ratio
        assert relative_error < 0.05, (
            f"{line_name}: expected {expected_ratio:.2f}, got {measured_ratio:.2f} "
        )
        f"({100 * relative_error:.1f}% error)"


@pytest.mark.regression_paper
def test_nlr_delegate_to_richardson():
    """Verify that compute_nlr_sed delegates correctly to richardson2014 version."""
    wavelength = jnp.linspace(3000, 10000, 500)
    l_disc_bol_erg = 1e45
    covering_fraction = 0.1
    fwhm_kms = 500.0
    line_efficiency = 0.10

    sed_delegated = compute_nlr_sed(
        wavelength=wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        fwhm_kms=fwhm_kms,
        line_efficiency=line_efficiency,
    )

    sed_richardson = compute_nlr_sed_richardson2014(
        wavelength=wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        fwhm_kms=fwhm_kms,
        line_efficiency=line_efficiency,
    )

    # Should be identical (delegated version just calls Richardson directly)
    assert jnp.allclose(sed_delegated, sed_richardson, rtol=1e-6)


if __name__ == "__main__":
    import numpy as np

    test_richardson_nlr_line_ratios()
    test_nlr_delegate_to_richardson()
    print("All NLR regression tests passed!")
