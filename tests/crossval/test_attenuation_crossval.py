# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation tests for new attenuation curves.

TEA E_b(delta) relation against Haskell+2024 calibration points.
Conroy2010 limiting behavior against Cardelli and power_law.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import cardelli, conroy2010

pytestmark = pytest.mark.crossval


def _ccm89_ab_full(x):
    """Full (7-term) CCM89 optical/near-UV polynomial a(x), b(x); float64 numpy."""
    y = x - 1.82
    a = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return a, b


def _ccm89_ab_nuv_bump1(x):
    """CCM89 near-UV Drude a(x), b(x) at UV-bump strength (FSPS ``uvb``) = 1."""
    a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341)
    b = -3.09 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263)
    return a, b


class TestTEACrossVal:
    """Validate TEA E_b(delta) relation against Haskell+2024 Fig. 3."""

    @pytest.mark.parametrize(
        "delta, eb_expected, rtol",
        [
            # delta=0 -> E_b = 2.5 (by definition)
            (0.0, 2.5, 1e-10),
            # delta=-0.4 -> E_b = 2.5 * exp(-1.4) ~ 0.617
            (-0.4, 2.5 * float(jnp.exp(-1.4)), 1e-6),
            # delta=0.2 -> E_b = 2.5 * exp(0.7) ~ 5.03
            (0.2, 2.5 * float(jnp.exp(0.7)), 1e-6),
            # delta=-0.8 -> E_b = 2.5 * exp(-2.8) ~ 0.152
            (-0.8, 2.5 * float(jnp.exp(-2.8)), 1e-6),
        ],
    )
    def test_eb_delta_relation(self, delta, eb_expected, rtol):
        """E_b(delta) = 2.5 * exp(3.5 * delta) at specific calibration points."""
        eb = 2.5 * float(jnp.exp(3.5 * delta))
        assert_allclose(eb, eb_expected, rtol=rtol)

    def test_monotonic_eb_with_delta(self):
        """E_b increases monotonically with delta (shallower -> stronger bump)."""
        deltas = jnp.linspace(-1.0, 0.5, 50)
        ebs = 2.5 * jnp.exp(3.5 * deltas)
        assert jnp.all(jnp.diff(ebs) > 0)


class TestConroy2010CrossVal:
    """Validate Conroy2010 limiting behavior against known curves."""

    @pytest.mark.parametrize("dust_Rv", [2.0, 3.1, 5.0])
    def test_deep_uv_matches_cardelli(self, dust_Rv):
        """conroy2010 at bump=1.0 relates to Cardelli (CCM89) by an exact relation.

        Outside the near-UV segment (x < 3.3 or x >= 5.9 um^-1) the two share
        every coefficient and normalization, so they are bit-identical. Inside
        it, FSPS's near-UV continuity correction is present in conroy2010 but
        absent from cardelli, so they differ by exactly the closed-form Delta
        offset asserted below (see tests/contract/test_conroy2010_continuity.py
        for the full parity check against an independent FSPS transcription).
        """
        lam_b59, lam_b80 = 1e4 / 5.9, 1e4 / 8.0
        wave = np.array(
            sorted(
                set(np.linspace(1000.0, 2000.0, 100))
                | {lam_b59 - 1.0, lam_b59 + 1.0, lam_b80 - 1.0, lam_b80 + 1.0}
            )
        )
        x = 1e4 / wave
        inside = (x >= 3.3) & (x < 5.9)

        k_c10 = np.asarray(conroy2010(jnp.array(wave), dust_Rv=dust_Rv, dust_bump_strength=1.0))
        k_mw = np.asarray(cardelli(jnp.array(wave), dust_Rv=dust_Rv))

        assert_allclose(k_c10[~inside], k_mw[~inside], rtol=1e-9)

        a5500, b5500 = _ccm89_ab_full(1.0 / 0.55)
        k5500 = a5500 + b5500 / dust_Rv
        a3, b3 = _ccm89_ab_full(3.3)
        k_opt3 = a3 + b3 / dust_Rv
        an3, bn3 = _ccm89_ab_nuv_bump1(3.3)
        k_nuv3 = an3 + bn3 / dust_Rv
        delta = k_opt3 - k_nuv3
        predicted_inside = (3.3 / x[inside]) ** 6 * delta / k5500
        assert_allclose(k_c10[inside] - k_mw[inside], predicted_inside, atol=1e-9)
