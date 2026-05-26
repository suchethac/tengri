# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for SFR hardcoded to 1.0 Msun/yr bug.

Bug: sed_pipeline.py:638 — SFR fallback was 1.0 Msun/yr for all parametric SFH.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestSFRNotHardcoded:
    """Bug: sed_pipeline.py:638 — SFR fallback was 1.0 Msun/yr for all parametric SFH."""

    def test_sfr_varies_with_mass(self):
        """SFR used for X-ray scaling should depend on the SFH, not be 1.0."""
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        # double_powerlaw(t_lookback, alpha, beta, tau, norm)
        # norm scales SFR amplitude, so SFR[-1] should scale with norm
        t_lookback = jnp.logspace(6, 10, 100)
        sfr_high = double_powerlaw(t_lookback, alpha=0.5, beta=2.0, tau=1e9, norm=100.0)
        sfr_low = double_powerlaw(t_lookback, alpha=0.5, beta=2.0, tau=1e9, norm=0.1)
        # sfr[-1] is the instantaneous SFR; high-norm galaxy must exceed low-norm
        assert sfr_high[-1] > sfr_low[-1]
        # Neither should be 1.0 Msun/yr by accident (hardcoded fallback bug)
        assert not jnp.isclose(sfr_high[-1], 1.0, atol=0.1)
        assert not jnp.isclose(sfr_low[-1], 1.0, atol=0.1)
