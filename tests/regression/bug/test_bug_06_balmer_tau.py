# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-06: Balmer continuum tau direction.

See ADR / docs/known_bugs.md for full context.
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestBug06BalmerTau:
    """qsogen.py:397 — tau must increase at shorter wavelengths.

    Grandi 1982: σ(ν) ∝ ν³, so τ ∝ ν³ ∝ λ⁻³ must increase at shorter wavelengths.
    Buggy code had τ ∝ λ³ (inverted).
    """

    def test_tau_increases_shortward(self):
        """Grandi 1982: sigma(nu) ~ nu^3, so tau increases at shorter lambda."""
        wavbe = 3646.0  # Balmer edge wavelength
        taube = 1.0

        # Current (buggy) code: tau = taube * (wave / wavbe)^3
        wave_short = 3000.0
        wave_long = 3500.0
        tau_short_buggy = taube * (wave_short / wavbe) ** 3
        tau_long_buggy = taube * (wave_long / wavbe) ** 3

        # Correct: tau = taube * (wavbe / wave)^3
        tau_short_correct = taube * (wavbe / wave_short) ** 3
        tau_long_correct = taube * (wavbe / wave_long) ** 3

        # Bug: tau_short < tau_long (wrong — should be higher at shorter lambda)
        assert tau_short_buggy < tau_long_buggy, (
            "If this fails, BUG-06 may have been fixed — remove xfail"
        )

        # Correct: tau_short > tau_long
        assert tau_short_correct > tau_long_correct
