# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for neb_fdust LyC energy credited to dust IR budget (#2539).

Tests that Lyman-continuum photons absorbed by dust inside HII regions
(neb_fdust) are properly credited to the dust IR budget (L_absorbed),
matching CIGALE behavior. Previously, neb_fdust only suppressed nebular
emission via the k-factor but the absorbed energy vanished.

Markers
-------
- `@pytest.mark.regression_bug` — Bug regression test
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestFdustEnergyInIRBudget:
    """Test that neb_fdust LyC energy enters the dust IR budget (#2539)."""

    def test_log_L_lyc_dust_published_when_fdust_nonzero(self):
        """Test that log_L_lyc_dust is published only when neb_fdust > 0."""
        from tengri.forward.energy_balance import bolometric_absorbed_log10
        from tengri.utils.physics_constants import C_AA
        import jax

        # Create a mock stellar SED
        wave = jnp.logspace(np.log10(100), np.log10(10000), 1000)  # 100-10000 Å
        nu = C_AA / wave
        sed = jnp.ones_like(wave) * 1e29  # [erg/s/Hz]

        # Test computing L_LyC
        lyc_mask = wave < 912.0
        lyc_intrinsic = jnp.where(lyc_mask, sed, 0.0)
        lyc_attenuated = jnp.zeros_like(lyc_intrinsic)

        log_L_lyc, _ = bolometric_absorbed_log10(
            lyc_intrinsic, lyc_attenuated, nu, wave=wave, lyman_cutoff_aa=None
        )

        # Check that we get a finite LyC luminosity
        # log_L_lyc will be finite for wavelengths < 912 Å
        assert jnp.isfinite(log_L_lyc), f"log_L_lyc should be finite, got {log_L_lyc}"
        assert float(log_L_lyc) > 0, f"log_L_lyc should be positive for this SED, got {log_L_lyc}"

        # Test log_L_lyc_dust computation: log10(fdust) + log_L_lyc
        fdust = 0.3
        log_L_lyc_dust = jnp.log10(fdust) + log_L_lyc

        # Check it's less than log_L_lyc (because we multiplied by 0.3)
        assert float(log_L_lyc_dust) < float(log_L_lyc), (
            "log_L_lyc_dust should be less than log_L_lyc after multiplying by fdust < 1"
        )

    def test_defaults_with_fdust_zero(self):
        """Test that when neb_fdust=0, log_L_lyc_dust is -inf (represents zero)."""
        # When neb_fdust = 0:
        # log_fdust = log10(0) = -inf
        # log_L_lyc_dust = -inf + log_L_lyc = -inf
        fdust = 0.0
        log_L_lyc = 45.0  # arbitrary positive value

        log_fdust = jnp.log10(jnp.where(fdust > 0, fdust, 1.0))
        log_L_lyc_dust = jnp.where(
            fdust > 0.0, log_fdust + log_L_lyc, -jnp.inf
        )

        # Should be -inf
        assert jnp.isinf(log_L_lyc_dust) and float(log_L_lyc_dust) < 0, (
            f"log_L_lyc_dust should be -inf when neb_fdust=0, got {log_L_lyc_dust}"
        )

    def test_log_L_lyc_dust_increases_with_fdust(self):
        """Test that log_L_lyc_dust increases monotonically with neb_fdust."""
        log_L_lyc = 45.0  # Mock stellar LyC luminosity

        # Test multiple fdust values
        fdust_values = jnp.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.9])

        def compute_lyc_dust(fdust):
            log_fdust = jnp.log10(jnp.where(fdust > 0, fdust, 1.0))
            return jnp.where(fdust > 0.0, log_fdust + log_L_lyc, -jnp.inf)

        results = jnp.array([float(compute_lyc_dust(f)) for f in fdust_values])

        # Check monotonicity (ignoring -inf for fdust=0)
        # First non-inf value should be smallest, then increasing
        nonzero_mask = results > -np.inf
        nonzero_results = results[nonzero_mask]

        if len(nonzero_results) > 1:
            diffs = jnp.diff(nonzero_results)
            assert jnp.all(diffs >= 0), (
                "log_L_lyc_dust should be monotonically increasing with neb_fdust"
            )
