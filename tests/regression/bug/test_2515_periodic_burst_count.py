# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2515: periodic burst count cap removed.

Issue: periodic had a hard-coded cap of 100 bursts, which cut off
the history prematurely when age * n_bursts exceeded 100.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.mean_sfh import periodic

pytestmark = pytest.mark.regression_bug


def cig_periodic(age, typ, delta, tau):
    """CIGALE periodic: bursts at t = k*delta, k = 0, 1, 2, ... while k*delta < age."""
    tg = np.arange(age)
    s = np.zeros(age)
    # Burst shapes
    exp_shape = np.exp(-tg / tau)
    delayed_shape = np.exp(-tg / tau) * tg / tau**2
    rect_shape = (tg <= int(tau)).astype(float)
    b = [exp_shape, delayed_shape, rect_shape][typ]
    # Roll back by delta, accumulating
    for k in range(age // delta + 1):
        if k * delta < age:
            s += b
            b = np.roll(b, delta)
            b[:delta] = 0.0
    return s


class TestPeriodicBurstCount:
    """Verify periodic produces correct results with no burst cap."""

    @pytest.mark.parametrize("burst_type", [0, 1, 2])
    def test_periodic_no_cap(self, burst_type):
        """With delta=10 Myr, tau=2 Myr, age=5 Gyr, we have 500 bursts (>> 100 old cap)."""
        age_yr = 5e9
        age_myr = age_yr / 1e6
        delta_yr = 10e6
        delta_myr = delta_yr / 1e6
        tau_yr = 2e6
        tau_myr = tau_yr / 1e6

        # Myr grid (1 Myr resolution)
        t_myr = np.arange(1, int(age_myr) + 1)
        t_yr = t_myr * 1e6

        # Lookback times (cell centers)
        lb_center = (t_myr - 0.5) * 1e6
        lb_center = np.clip(lb_center, 0, age_yr - 1)

        sfr = np.asarray(
            periodic(
                jnp.array(lb_center),
                log_total_mass=10.0,
                delta_bursts_yr=delta_yr,
                tau_bursts_yr=tau_yr,
                burst_type=burst_type,
                age_yr=age_yr,
            )
        )

        # Test 1: every Δ-wide window has nonzero formed mass
        # (bursts are regularly spaced, so every window in [0, age] should have some mass)
        dt = jnp.gradient(jnp.array(lb_center))
        n_windows = int(age_myr / delta_myr)
        for k in range(n_windows):
            window_start = k * delta_myr * 1e6
            window_end = (k + 1) * delta_myr * 1e6
            # Lookback window: [age - window_end, age - window_start]
            mask = (lb_center >= age_yr - window_end) & (lb_center <= age_yr - window_start)
            if np.any(mask):
                mass_in_window = np.sum(sfr[mask] * dt[mask])
                assert mass_in_window > 0, (
                    f"Window {k} ([{window_start:.0f}, {window_end:.0f}] yr) has zero mass"
                )

    @pytest.mark.parametrize("burst_type", [0, 1, 2])
    def test_periodic_mass_distribution(self, burst_type):
        """Test 2: mass fraction formed at lookback > 1 Gyr is 0.8 ± 0.01."""
        age_yr = 5e9
        delta_yr = 10e6
        tau_yr = 2e6

        # Myr grid
        t_myr = np.arange(1, 5001)
        t_yr = t_myr * 1e6
        lb_center = (t_myr - 0.5) * 1e6
        lb_center = np.clip(lb_center, 0, age_yr - 1)

        sfr = np.asarray(
            periodic(
                jnp.array(lb_center),
                log_total_mass=10.0,
                delta_bursts_yr=delta_yr,
                tau_bursts_yr=tau_yr,
                burst_type=burst_type,
                age_yr=age_yr,
            )
        )

        dt = jnp.gradient(jnp.array(lb_center))
        total_mass = np.sum(sfr * dt)

        # Mass formed at lookback > 1 Gyr
        mask_old = lb_center > 1e9
        mass_old = np.sum(sfr[mask_old] * dt[mask_old])
        frac_old = mass_old / total_mass

        # Verify it's approximately 0.8
        assert 0.79 < frac_old < 0.81, (
            f"Mass fraction at z > 1 Gyr: {frac_old:.3f} (should be ~0.80)"
        )

    @pytest.mark.parametrize("burst_type", [0, 1, 2])
    def test_periodic_vs_cigale_no_cap(self, burst_type):
        """Compare against CIGALE periodic (reversed to lookback) with many bursts."""
        age_yr = 5e9
        age_myr = age_yr / 1e6
        delta_myr = 10.0
        tau_myr = 2.0

        # CIGALE: forward time
        cig_sfh = cig_periodic(int(age_myr), burst_type, int(delta_myr), int(tau_myr))
        cig_reversed = cig_sfh[::-1]

        # Tengri: evaluate at lookback cell centers
        t_myr = np.arange(1, int(age_myr) + 1)
        t_yr = t_myr * 1e6
        lb_center = (t_myr - 0.5) * 1e6
        lb_center = np.clip(lb_center, 0, age_yr - 1)

        sfr_tengri = np.asarray(
            periodic(
                jnp.array(lb_center),
                log_total_mass=10.0,
                delta_bursts_yr=delta_myr * 1e6,
                tau_bursts_yr=tau_myr * 1e6,
                burst_type=burst_type,
                age_yr=age_yr,
            )
        )

        # Normalize to unit sum
        sfr_tengri_norm = sfr_tengri / (np.sum(sfr_tengri) + 1e-30)
        cig_reversed_norm = cig_reversed / (np.sum(cig_reversed) + 1e-30)

        # L1 distance
        d = 0.5 * np.sum(np.abs(sfr_tengri_norm - cig_reversed_norm))

        # Should be small (close match to CIGALE reversed)
        assert d < 1e-3, f"Distance to CIGALE reversed: {d:.2e} (should be small)"
