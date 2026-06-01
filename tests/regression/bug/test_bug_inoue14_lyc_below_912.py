# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #494: Inoue14 Lyman-continuum opacity below 912 Å.

The previous implementation of ``_tau_lc_laf`` / ``_tau_lc_dla`` masked out
``wave_obs <= _LAMBDA_LIMIT`` so the rest-frame Lyman-continuum region
returned exactly zero opacity (T = 1 from the LC channel, but the function
returned T = 0 because the Lyman-series opacity blew up). The fix ports the
eazy-py analytic formulas from Inoue et al. (2014, Eqs. 25–29), with the
correct active mask ``wave_obs < lamL*(1+z_source)``, which extends opacity
continuously below the limit.

Reference: BAGPIPES Inoue14 table at z=4 (rest 800–950 Å region).
"""

import jax.numpy as jnp
import pytest

from tengri.components.igm import igm_transmission

pytestmark = pytest.mark.regression_bug


class TestBug494LyCBelowLimit:
    """Inoue+2014 LyC opacity is smooth across the 912 Å limit, not a hard cliff."""

    def test_lyc_nonzero_below_912_at_z4(self):
        """Below the rest-frame Lyman limit, T must be > 0 (continuum, not a cliff)."""
        z = 4.0
        # Rest 800, 850, 900 Å → observed 4000, 4250, 4500 Å.
        wave_obs = jnp.array([800.0, 850.0, 900.0]) * (1.0 + z)
        T = igm_transmission(wave_obs, z)
        for w, t in zip([800, 850, 900], T):
            assert 0.0 < float(t) < 1.0, (
                f"rest {w} Å (z=4): T={float(t):.3f} — must be in (0, 1), not a hard cliff."
            )

    def test_lyc_below_limit_matches_bagpipes_order_of_magnitude(self):
        """At z=4, rest 800–900 Å transmission has the same order of magnitude as BAGPIPES.

        BAGPIPES reads the Inoue+2014 supplementary table; tengri uses the analytic
        formulas (eazy-py port). Both should agree to a factor of ~few in the LyC
        region — the previous T = 0 floor was wrong by 6+ orders of magnitude.
        """
        z = 4.0
        # BAGPIPES reference (#494):
        # rest 800 → 0.048, rest 850 → 0.18, rest 900 → 0.36
        wave_obs = jnp.array([800.0, 850.0, 900.0]) * (1.0 + z)
        T = igm_transmission(wave_obs, z)
        # All three must be > 0.01 (vs prior 0.0 cliff) and < 0.6.
        assert float(T.min()) > 0.01
        assert float(T.max()) < 0.6

    def test_lyc_smooth_across_limit(self):
        """No discontinuity at the rest 912 Å limit — transmission varies smoothly."""
        z = 3.0
        # Sample across 905, 910, 915, 920, 925 Å rest.
        wave_rest = jnp.array([905.0, 910.0, 915.0, 920.0, 925.0])
        wave_obs = wave_rest * (1.0 + z)
        T = igm_transmission(wave_obs, z)
        diffs = jnp.diff(T)
        # Differences should be small (smooth) — no cliff of order unity.
        assert float(jnp.abs(diffs).max()) < 0.3, (
            f"IGM T should be smooth across 912 Å, got diffs={diffs}"
        )
