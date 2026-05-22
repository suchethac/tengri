# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for shock emission unit consistency bug.

Bug: shock.py:182-206 — Gaussian branch multiplied by _LSUN_ERG giving erg/s/Hz;
delta-function branch returned Lsun/Hz.  Both should return Lsun/Hz.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestShockEmissionUnits:
    """Bug: shock.py:182-206 — unit mismatch between Gaussian and delta branches."""

    def test_gaussian_delta_consistent_total_power(self):
        """Total power (integral of SED over frequency) should match between branches.

        Both branches represent the same physical emission, so the total luminosity
        (integral of SED over nu) must be approximately equal for the same input.

        Uses line_sigma_aa=200 Å to ensure the Gaussian is well-sampled by the
        log-spaced wavelength grid (grid spacing ~38 Å at Halpha → sigma/spacing ≈ 5).
        A 3 Å sigma would be unresolved and the integral would underestimate power.
        """
        from tengri.components.nebular.shock import compute_shock_sed

        # Dense linear grid around optical lines to ensure Gaussian is well-sampled
        wave = jnp.linspace(3500.0, 7500.0, 10000)
        _C_AA = 2.99792458e18
        nu = _C_AA / wave

        # Gaussian branch — wide sigma (200 Å) to be well-sampled by the grid
        sed_gaussian = compute_shock_sed(
            wave, shock_velocity=200.0, l_shock_halpha=1.0, line_sigma_aa=200.0
        )
        # Delta branch
        sed_delta = compute_shock_sed(
            wave, shock_velocity=200.0, l_shock_halpha=1.0, line_sigma_aa=0.0
        )

        # Sort for integration (nu decreases as wave increases)
        sort_idx = jnp.argsort(nu)
        power_gaussian = jnp.abs(jnp.trapezoid(sed_gaussian[sort_idx], nu[sort_idx]))
        power_delta = jnp.abs(jnp.trapezoid(sed_delta[sort_idx], nu[sort_idx]))

        # Both should be in the same units (Lsun/Hz); ratio should be order unity
        if power_gaussian > 0 and power_delta > 0:
            ratio = power_gaussian / power_delta
            assert 0.01 < ratio < 100.0, (
                f"Gaussian/delta power ratio = {ratio:.2e}; suggests unit mismatch (expected ~1)"
            )

    def test_sed_order_of_magnitude(self):
        """SED values should be in Lsun/Hz, not erg/s/Hz.

        For Halpha luminosity = 1 Lsun, the peak SED value in Lsun/Hz should be
        much smaller than 3.828e33 (which would indicate erg/s/Hz units).
        """
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.logspace(2.5, 5.0, 500)
        sed = compute_shock_sed(wave, shock_velocity=200.0, l_shock_halpha=1.0, line_sigma_aa=3.0)
        peak = float(jnp.max(sed))
        # If units were erg/s/Hz, peak would be ~3.828e33 × (1/sigma_nu) >> 1
        # In Lsun/Hz, peak should be << 3.828e33
        assert peak < 1e10, f"SED peak {peak:.2e} Lsun/Hz suggests wrong units (erg/s/Hz?)"
