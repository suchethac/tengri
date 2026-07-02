# SPDX-License-Identifier: BSD-3-Clause
"""Regression: Lopez+2024 IR-tied corona uses the Asmus+2015 α_IRX direction.

`xray_agn_corona_lopez24` derives the 2-10 keV corona from the nuclear 12 µm
luminosity via `α_IRX = log10(νLν(12µm) / L_X(2-10 keV))` (Asmus+2015,
Lopez+2024), matching CIGALE `lopez24.py`:

    l_agn_2to10keV = L_12um / 10**alpha_irx          # CIGALE lopez24.py:200

An earlier port had the ratio **inverted** (`L_X = 10**α_IRX · νLν`), which put
the X-ray a factor `10**(2·α_IRX) ≈ 4×` too high at the α_IRX = 0.3 default. This
guards the direction: at α_IRX = 0.3 the integrated corona L_X(2-10 keV) must be
≈ `νLν(12µm) / 10**0.3` (X-ray *below* the 12 µm), not above it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.xray.xray import xray_agn_corona_lopez24

pytestmark = [pytest.mark.regression_bug]

_C_AA = 2.998e18  # Å/s
_NU_12UM = _C_AA / 1.2e5  # 12 µm = 120000 Å


def _integrated_lx_2to10(nu_lnu_12um: float, alpha_irx: float) -> float:
    """Integrate the (unabsorbed, isotropic) lopez24 corona over 2-10 keV."""
    l_nu_12um = nu_lnu_12um / _NU_12UM  # νLν -> Lν [erg/s/Hz]
    e_kev = np.logspace(np.log10(0.5), np.log10(30.0), 4000)
    lam_aa = 12.398 / e_kev
    sed = np.asarray(
        xray_agn_corona_lopez24(
            lam_aa, l_nu_12um, alpha_irx=alpha_irx, log_nh=18.0, apply_anisotropy=False
        )
    )
    nu = _C_AA / lam_aa
    band = (e_kev >= 2.0) & (e_kev <= 10.0)
    order = np.argsort(nu[band])
    return float(np.trapezoid(sed[band][order], nu[band][order]))


def test_lopez24_lx_is_below_12um_at_default_alpha():
    """L_X(2-10 keV) ≈ νLν(12µm) / 10**α_IRX (Asmus+15 / CIGALE), not × 10**α_IRX."""
    nu_lnu_12um = 1.0e44  # erg/s
    alpha = 0.3
    lx = _integrated_lx_2to10(nu_lnu_12um, alpha)
    expected = nu_lnu_12um / 10.0**alpha  # CIGALE lopez24.py:200
    np.testing.assert_allclose(lx, expected, rtol=0.02)
    # The old inverted port would give ~10**(2α) = 4× this — well outside 2%.
    assert lx < nu_lnu_12um, "X-ray must sit below the 12 µm luminosity (α_IRX > 0)"


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6])
def test_lopez24_scales_inversely_with_alpha_irx(alpha):
    """Higher α_IRX → fainter X-ray (L_X = νLν / 10**α_IRX)."""
    nu_lnu_12um = 1.0e44
    lx = _integrated_lx_2to10(nu_lnu_12um, alpha)
    np.testing.assert_allclose(lx, nu_lnu_12um / 10.0**alpha, rtol=0.02)
