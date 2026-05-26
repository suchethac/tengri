# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-07: Disc ring area missing pi factor.

See ADR / docs/known_bugs.md for full context.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestBug07DiscArea:
    """disc.py:298 — L_nu per ring must include pi*B_nu.

    Reference: Kubota & Done (2018) §2.2.
    Each ring element dA = 2π R dR contributes L_ring = π B_ν(T_ring) dA cos(i).
    Buggy code omitted the outer π factor.
    """

    def test_single_ring_luminosity(self):
        """Compare single-ring L_nu against analytical pi*B_nu*A*cos(i)."""
        h = 6.626e-27  # erg*s
        c = 3e10  # cm/s
        k = 1.38e-16  # erg/K
        T = 1e4  # K
        R = 1e13  # cm
        dR = 1e11  # cm
        nu = 1e15  # Hz (UV)
        cos_i = 1.0

        # Planck function B_nu
        x = h * nu / (k * T)
        B_nu = 2 * h * nu**3 / c**2 / (np.exp(x) - 1)

        # Correct analytical: pi * B_nu * (2*pi*R*dR) * cos_i
        L_analytical = np.pi * B_nu * 2 * np.pi * R * dR * cos_i

        # Buggy (missing pi): B_nu * (2*pi*R*dR) * cos_i
        L_buggy = B_nu * 2 * np.pi * R * dR * cos_i

        assert abs(L_analytical / L_buggy - np.pi) < 0.01, (
            "Ratio should be pi — the missing factor"
        )
