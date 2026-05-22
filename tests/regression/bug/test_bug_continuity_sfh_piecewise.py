# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for continuity SFH piecewise-constant bug.

Bug: nonparametric.py:99-101 — jnp.interp on bin centers gives linear interpolation;
Leja+2019 defines the SFH as piecewise-constant (step functions).
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestContinuitySFHPiecewiseConstant:
    """Bug: nonparametric.py:99-101 — continuity returned linear not step functions."""

    def test_sfr_constant_within_bin(self):
        """SFR must be exactly constant within each bin (step function)."""
        from tengri.components.stellar.sfh.nonparametric import continuity

        edges = jnp.array([0.0, 1.0, 5.0, 13.7])  # 3 bins in Gyr
        # All ages within the middle bin [1, 5] Gyr must have identical SFR
        ages_in_bin1 = jnp.linspace(1.01e9, 4.99e9, 50)
        kwargs = {"ratio_0": 1.0, "ratio_1": -0.5}
        sfr = continuity(ages_in_bin1, log_total_mass=10.0, bin_edges_gyr=edges, **kwargs)
        # Maximum deviation from mean should be zero (step function)
        max_dev = float(jnp.max(jnp.abs(sfr - sfr[0])))
        assert max_dev == 0.0, f"SFR varies by {max_dev:.3e} within a bin (should be zero)"
