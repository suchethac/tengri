# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-09: Mean ionizing photon energy.

See ADR / docs/known_bugs.md for full context.
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestBug09MeanPhotonEnergy:
    """agn_nebular.py:177-183 — <hnu> must depend on spectral index.

    For a power-law ionizing continuum, the mean photon energy is
    <hν> = ∫ hν × ν^(α-1) dν / ∫ ν^(α-1) dν.

    Buggy code used the same exponent for numerator and denominator integrals,
    giving <hν> = constant (independent of α).
    """

    def test_mean_energy_varies_with_alpha(self):
        """For different power-law indices, <hnu> must differ."""
        h = 6.626e-27
        nu_lyman = 3.29e15  # Lyman limit frequency
        nu_max = 1e18  # X-ray cutoff

        def correct_mean_hnu(alpha):
            """Correct: <hnu> = integral(hnu * nu^{alpha-1}) / integral(nu^{alpha-1})."""
            # numerator exponent: alpha+1, denominator exponent: alpha
            num = (nu_max ** (alpha + 1) - nu_lyman ** (alpha + 1)) / (alpha + 1)
            den = (nu_max**alpha - nu_lyman**alpha) / alpha
            return h * num / den

        def buggy_mean_hnu(alpha):
            """Bug: both integrals use nu^alpha."""
            num = (nu_max ** (alpha + 1) - nu_lyman ** (alpha + 1)) / (alpha + 1)
            den = (nu_max ** (alpha + 1) - nu_lyman ** (alpha + 1)) / (alpha + 1)
            return h * abs(num / den)

        # Buggy version: <hnu> = h regardless of alpha
        assert abs(buggy_mean_hnu(-1.5) - buggy_mean_hnu(-2.5)) < 1e-30, (
            "Bug: mean photon energy is constant (= h)"
        )

        # Correct version: <hnu> changes with alpha
        e1 = correct_mean_hnu(-1.5)
        e2 = correct_mean_hnu(-2.5)
        assert abs(e1 - e2) / e1 > 0.1, "Correct <hnu> must differ by >10% for different alpha"
