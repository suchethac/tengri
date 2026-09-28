# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2514: CIGALE SFH time-reversal correction.

Issues: delayed_bq, periodic, buat08 were reading CIGALE's forward time as
lookback time, giving time-reversed histories.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.mean_sfh import buat08, delayed_bq, periodic

pytestmark = pytest.mark.regression_bug

# Myr grid (1 Myr resolution, 1 Gyr = 1000 Myr max)
_MYR_GRID = np.arange(1, 1001)  # 1 to 1000 Myr
_AGE_MYR = _MYR_GRID[-1]  # 1000 Myr = 1 Gyr
_AGE_YR = _AGE_MYR * 1e6


def cig_delayed(age, tau):
    """CIGALE delayed: t * exp(-t / tau) / tau^2, t in [0, age]."""
    t = np.arange(age)
    return t * np.exp(-t / tau) / tau**2


def cig_delayedbq(age, tau, age_bq, r):
    """CIGALE delayed_bq: delayed with burst/quench at t >= age - age_bq."""
    s = cig_delayed(age, tau)
    t = np.arange(age)
    # At onset: SFR = r * SFR(age - age_bq)
    sfr_onset = s[age - age_bq - 1] if age > age_bq else s[-1]
    s[t >= age - age_bq] = r * sfr_onset
    return s


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


def cig_buat08(age, v):
    """CIGALE buat08: velocity-parameterized SFH from Buat+2008."""
    V = [40, 50, 60, 70, 80, 90, 100, 150, 220, 290, 360]
    a_vals = [4.73, 5.28, 5.77, 6.21, 6.62, 6.99, 7.34, 8.74, 10.01, 10.82, 11.35]
    b_vals = [-0.11, 0.029, 0.16, 0.29, 0.41, 0.51, 0.61, 0.98, 1.25, 1.36, 1.37]
    c_vals = [0.79, 0.68, 0.57, 0.46, 0.36, 0.27, 0.18, -0.20, -0.55, -0.74, -0.85]
    a, b, c = (np.interp(v, V, x) for x in [a_vals, b_vals, c_vals])
    t = (np.arange(age) + 1) / 1000  # time in Gyr
    return 10 ** (a + b * np.log10(t) + c * t**0.5 - 9)


class TestDelayedBqOrientation:
    """Verify delayed_bq uses cosmic time since formation (T = age - t_lb), not lookback."""

    @pytest.mark.parametrize(
        "tau_yr,age_bq_yr,r_sfr",
        [
            (2e9, 0.3e9, 0.1),
            (2e9, 0.3e9, 5.0),
        ],
    )
    def test_delayed_bq_vs_cigale(self, tau_yr, age_bq_yr, r_sfr):
        """Compare tengri delayed_bq against CIGALE (reversed to lookback)."""
        age_yr = 8e9
        age_myr = age_yr / 1e6

        # CIGALE: forward time (Myr since formation)
        cig_sfh = cig_delayedbq(int(age_myr), tau_yr / 1e6, age_bq_yr / 1e6, r_sfr)
        cig_reversed = cig_sfh[::-1]  # reverse to lookback convention

        # Tengri: evaluate at lookback cell centers
        lb_center = (_MYR_GRID - 0.5) * 1e6  # yr
        lb_center = np.clip(lb_center, 0, age_yr - 1)
        sfr_tengri = np.asarray(
            delayed_bq(
                jnp.array(lb_center),
                log_total_mass=10.0,
                tau_main_yr=tau_yr,
                age_main_yr=age_yr,
                age_bq_yr=age_bq_yr,
                r_sfr=r_sfr,
            )
        )

        # Normalize to unit sum for comparison
        sfr_tengri_norm = sfr_tengri / (np.sum(sfr_tengri) + 1e-30)
        cig_reversed_norm = cig_reversed / (np.sum(cig_reversed) + 1e-30)

        # L1 distance
        d_reversed = 0.5 * np.sum(np.abs(sfr_tengri_norm - cig_reversed_norm))

        # Verify distance is small (close to CIGALE reversed)
        assert d_reversed < 1e-3, f"Distance to CIGALE reversed: {d_reversed:.2e} (too large)"

        # Verify distance to unreversed is large (confirms we're not time-reversed)
        d_unreversed = 0.5 * np.sum(
            np.abs(
                sfr_tengri_norm
                - cig_delayedbq(int(age_myr), tau_yr / 1e6, age_bq_yr / 1e6, r_sfr)
                / np.sum(cig_delayedbq(int(age_myr), tau_yr / 1e6, age_bq_yr / 1e6, r_sfr))
            )
        )
        assert d_unreversed > 0.2, (
            f"Distance to CIGALE unreversed: {d_unreversed:.2e} (should be large)"
        )

    def test_delayed_bq_burst_physics(self):
        """Physics check: for r_sfr > 1 (burst), verify burst level and extent."""
        age_yr = 8e9
        age_bq_yr = 0.3e9
        tau_yr = 2e9
        r_sfr = 5.0

        # Evaluate at lookback times < 10 Myr and > 10 Myr
        t_recent = np.array([1e6, 5e6])  # < age_bq_yr
        t_old = np.array([0.5e9])  # > age_bq_yr

        sfr_recent = np.asarray(
            delayed_bq(
                jnp.array(t_recent),
                log_total_mass=10.0,
                tau_main_yr=tau_yr,
                age_main_yr=age_yr,
                age_bq_yr=age_bq_yr,
                r_sfr=r_sfr,
            )
        )
        sfr_old = np.asarray(
            delayed_bq(
                jnp.array(t_old),
                log_total_mass=10.0,
                tau_main_yr=tau_yr,
                age_main_yr=age_yr,
                age_bq_yr=age_bq_yr,
                r_sfr=r_sfr,
            )
        )

        # Recent times should have nonzero SFR (in the burst window)
        assert np.all(sfr_recent > 0), "Recent burst SFR should be > 0"

        # Burst SFR is constant at r_sfr * delayed(age_bq) for t_lb < age_bq
        # So the ratio of recent to old should be approximately constant
        ratio = sfr_recent[0] / sfr_old[0]
        assert ratio > 0, "Burst ratio should be > 0"

        # Verify burst lies only in recent lookback times (t_lb < age_bq_yr)
        t_beyond_burst = np.array([age_bq_yr + 1e7])  # beyond burst window
        sfr_beyond = np.asarray(
            delayed_bq(
                jnp.array(t_beyond_burst),
                log_total_mass=10.0,
                tau_main_yr=tau_yr,
                age_main_yr=age_yr,
                age_bq_yr=age_bq_yr,
                r_sfr=r_sfr,
            )
        )
        # Beyond burst, should be back to delayed shape
        assert sfr_beyond[0] < sfr_recent[0], "SFR beyond burst window should be < burst SFR"


class TestPeriodicOrientation:
    """Verify periodic uses cosmic time since formation (T = age - t_lb), not lookback."""

    @pytest.mark.parametrize("burst_type", [0, 1, 2])
    def test_periodic_vs_cigale(self, burst_type):
        """Compare tengri periodic against CIGALE (reversed to lookback)."""
        age_yr = 8e9
        age_myr = age_yr / 1e6
        delta_myr = 0.7e9 / 1e6  # 0.7 Gyr
        tau_myr = 0.1e9 / 1e6  # 0.1 Gyr

        # CIGALE: forward time (Myr since formation)
        cig_sfh = cig_periodic(int(age_myr), burst_type, int(delta_myr), int(tau_myr))
        cig_reversed = cig_sfh[::-1]

        # Tengri: evaluate at lookback cell centers
        lb_center = (_MYR_GRID - 0.5) * 1e6
        lb_center = np.clip(lb_center, 0, age_yr - 1)
        sfr_tengri = np.asarray(
            periodic(
                jnp.array(lb_center),
                log_total_mass=10.0,
                delta_bursts_yr=0.7e9,
                tau_bursts_yr=0.1e9,
                burst_type=burst_type,
                age_yr=age_yr,
            )
        )

        # Normalize to unit sum
        sfr_tengri_norm = sfr_tengri / (np.sum(sfr_tengri) + 1e-30)
        cig_reversed_norm = cig_reversed / (np.sum(cig_reversed) + 1e-30)

        # L1 distance
        d_reversed = 0.5 * np.sum(np.abs(sfr_tengri_norm - cig_reversed_norm))

        # Verify distance is small
        assert d_reversed < 1e-3, f"Distance to CIGALE reversed: {d_reversed:.2e} (too large)"

        # Verify distance to unreversed is large
        d_unreversed = 0.5 * np.sum(
            np.abs(
                sfr_tengri_norm
                - cig_periodic(int(age_myr), burst_type, int(delta_myr), int(tau_myr))
                / np.sum(cig_periodic(int(age_myr), burst_type, int(delta_myr), int(tau_myr)))
            )
        )
        assert d_unreversed > 0.2, (
            f"Distance to CIGALE unreversed: {d_unreversed:.2e} (should be large)"
        )


class TestBuat08Orientation:
    """Verify buat08 uses cosmic time since formation (T = age - t_lb), not lookback."""

    @pytest.mark.parametrize("velocity", [100, 250])
    def test_buat08_vs_cigale(self, velocity):
        """Compare tengri buat08 against CIGALE (reversed to lookback)."""
        age_yr = 8e9
        age_myr = age_yr / 1e6

        # CIGALE: forward time (Myr since formation)
        cig_sfh = cig_buat08(int(age_myr), velocity)
        cig_reversed = cig_sfh[::-1]

        # Tengri: evaluate at lookback cell centers
        lb_center = (_MYR_GRID - 0.5) * 1e6
        lb_center = np.clip(lb_center, 0, age_yr - 1)
        sfr_tengri = np.asarray(
            buat08(
                jnp.array(lb_center),
                log_total_mass=10.0,
                velocity_km_s=velocity,
                age_yr=age_yr,
            )
        )

        # Normalize to unit sum
        sfr_tengri_norm = sfr_tengri / (np.sum(sfr_tengri) + 1e-30)
        cig_reversed_norm = cig_reversed / (np.sum(cig_reversed) + 1e-30)

        # L1 distance
        d_reversed = 0.5 * np.sum(np.abs(sfr_tengri_norm - cig_reversed_norm))

        # Verify distance is small
        assert d_reversed < 1e-3, f"Distance to CIGALE reversed: {d_reversed:.2e} (too large)"

        # Verify distance to unreversed is large
        d_unreversed = 0.5 * np.sum(
            np.abs(
                sfr_tengri_norm
                - cig_buat08(int(age_myr), velocity) / np.sum(cig_buat08(int(age_myr), velocity))
            )
        )
        assert d_unreversed > 0.2, (
            f"Distance to CIGALE unreversed: {d_unreversed:.2e} (should be large)"
        )
