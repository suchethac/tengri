"""Regression test for Leitherer02 cutoff consistency bug.

Bug: attenuation.py:143 — _calzetti_l02_kprime helper used 0.15 um cutoff,
but L02 polynomial is valid 970-1800 A (Leitherer+2002 ApJS 140 303 Eq. 14).
Standalone leitherer02 used 0.18 um; now both match.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestLeithererCutoff:
    """Bug: attenuation.py:143 — inconsistent Leitherer02 cutoffs."""

    def test_calzetti_l02_kprime_matches_leitherer02_at_1700A(self):
        """At 1700 A (between 1500 and 1800 A), both implementations should agree."""
        from tengri.components.dust.attenuation import _calzetti_l02_kprime, leitherer02

        wave = jnp.array([1700.0])
        k_helper = _calzetti_l02_kprime(wave)  # was using 0.15 cutoff, now 0.18
        k_standalone = leitherer02(wave)

        # The two should now agree (both use L02 polynomial at 1700 A < 1800 A)
        assert jnp.allclose(k_helper / 4.05, k_standalone, rtol=0.01), (
            f"_calzetti_l02_kprime/RV={float(k_helper / 4.05):.4f} != "
            f"leitherer02={float(k_standalone):.4f} at 1700 A"
        )
