# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2514: CIGALE SFH time-reversal correction.

Issues: delayed_bq, periodic, buat08 were reading CIGALE's forward time
(time since formation) as lookback time, giving time-reversed histories.
The fix evaluates each model's CIGALE formula in T = age - t_lookback.

Reference NumPy implementations of the same models as CIGALE's ``_init_code``
(1 Myr grid, index = Myr since formation, last element = present), validated
against pcigale, are compared against tengri evaluated at lookback cell centers,
after reversing the CIGALE array to lookback order.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.mean_sfh import buat08, delayed_bq, periodic

pytestmark = pytest.mark.regression_bug

# Myr grid (1 Myr resolution), age = 8 Gyr for every case in this file.
_AGE_YR = 8e9
_AGE_MYR = int(_AGE_YR / 1e6)
_MYR_GRID = np.arange(1, _AGE_MYR + 1)
_LB_CENTER = (_MYR_GRID - 0.5) * 1e6  # lookback cell centers [yr]


def cig_delayed(age, tau):
    """CIGALE delayed: t * exp(-t / tau) / tau^2, t in [0, age)."""
    t = np.arange(age)
    return t * np.exp(-t / tau) / tau**2


def cig_delayedbq(age, tau, age_bq, r):
    """CIGALE delayed_bq: delayed with burst/quench at t >= age - age_bq."""
    age = int(age)
    age_bq = int(age_bq)
    s = cig_delayed(age, tau)
    t = np.arange(age)
    s[t >= age - age_bq] = r * s[age - age_bq - 1]
    return s


def cig_periodic(age, typ, delta, tau):
    """CIGALE periodic: bursts at t = k*delta, k = 0, 1, 2, ... while k*delta < age."""
    tg = np.arange(age)
    s = np.zeros(age)
    b = [np.exp(-tg / tau), np.exp(-tg / tau) * tg / tau**2, (tg <= int(tau)).astype(float)][typ]
    for _ in np.arange(0, age, delta):
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


def _l1_distance(p, q):
    """D = 0.5 * sum|p - q| between two unit-sum-normalized arrays."""
    p = p / (np.sum(p) + 1e-300)
    q = q / (np.sum(q) + 1e-300)
    return float(0.5 * np.sum(np.abs(p - q)))


class TestDelayedBqOrientation:
    """Verify delayed_bq uses cosmic time since formation (T = age - t_lb), not lookback."""

    # Measured on the fixed code: D_reversed ~9.6e-5 for both r_sfr, D_unreversed
    # ~0.32-0.35. Thresholds set a little above/below those measured values.
    @pytest.mark.parametrize(
        "tau_yr,age_bq_yr,r_sfr",
        [
            (2e9, 0.3e9, 0.1),
            (2e9, 0.3e9, 5.0),
        ],
    )
    def test_delayed_bq_vs_cigale(self, tau_yr, age_bq_yr, r_sfr):
        """Compare tengri delayed_bq against CIGALE (reversed to lookback)."""
        cig_sfh = cig_delayedbq(_AGE_MYR, tau_yr / 1e6, age_bq_yr / 1e6, r_sfr)
        cig_reversed = cig_sfh[::-1]

        sfr_tengri = np.asarray(
            delayed_bq(
                jnp.array(_LB_CENTER),
                log_total_mass=10.0,
                tau_main_yr=tau_yr,
                age_main_yr=_AGE_YR,
                age_bq_yr=age_bq_yr,
                r_sfr=r_sfr,
            )
        )

        d_reversed = _l1_distance(sfr_tengri, cig_reversed)
        d_unreversed = _l1_distance(sfr_tengri, cig_sfh)

        assert d_reversed < 1e-3, f"Distance to CIGALE reversed: {d_reversed:.2e} (too large)"
        assert d_unreversed > 0.2, (
            f"Distance to CIGALE unreversed: {d_unreversed:.2e} (should be large)"
        )

    def test_delayed_bq_burst_physics(self):
        """Physics check for r_sfr=5 (bursting): the constant post-episode plateau.

        The mean SFR over t_lb < 10 Myr (entirely inside the burst window,
        since age_bq=300 Myr) must be > 0 and equal r_sfr times the delayed
        SFR at T = age_main - age_bq, in the SAME renormalized units — obtained
        by probing a point 1 yr past age_bq (still in the delayed branch, but
        indistinguishable from the T_bq value to float precision). It must
        also differ from the branch immediately on the other side of age_bq,
        confirming the branch boundary sits exactly at t_lb = age_bq.

        All probes share ONE ``delayed_bq`` call: the mass renormalization is
        a single scalar computed from the whole input array, so values from
        two different calls (with different point sets) are not comparable.
        """
        age_main = 8e9
        age_bq = 0.3e9
        tau = 2e9
        r_sfr = 5.0

        t_recent = jnp.linspace(0.0, 9e6, 20)  # entirely inside [0, 10 Myr)
        t_just_below = age_bq - 1.0  # constant branch, adjacent to the boundary
        t_at_bq = age_bq  # boundary itself: T >= T_bq holds at equality
        t_probe = age_bq + 1.0  # just above age_bq: delayed branch, ~T_bq
        t_all = jnp.concatenate([t_recent, jnp.array([t_just_below, t_at_bq, t_probe])])

        sfr = delayed_bq(
            t_all,
            log_total_mass=10.0,
            tau_main_yr=tau,
            age_main_yr=age_main,
            age_bq_yr=age_bq,
            r_sfr=r_sfr,
        )
        sfr_recent = sfr[:20]
        sfr_just_below = float(sfr[20])
        sfr_at_bq = float(sfr[21])
        sfr_probe = float(sfr[22])

        mean_recent = float(jnp.mean(sfr_recent))
        assert mean_recent > 0, "Recent burst SFR should be > 0"
        assert np.allclose(mean_recent, r_sfr * sfr_probe, rtol=1e-6), (
            f"mean SFR at t_lb<10 Myr ({mean_recent:.6g}) != r_sfr * delayed SFR at "
            f"T=age_main-age_bq ({r_sfr * sfr_probe:.6g})"
        )

        # The r_sfr branch lies entirely at t_lb <= age_bq: the boundary point
        # and a point just below it match the plateau exactly; a point just
        # above it (delayed branch) does not.
        assert np.allclose(sfr_just_below, mean_recent, rtol=1e-10)
        assert np.allclose(sfr_at_bq, mean_recent, rtol=1e-10)
        assert sfr_probe < sfr_just_below, (
            "SFR just past age_bq (delayed branch) should be below the burst plateau"
        )


class TestPeriodicOrientation:
    """Verify periodic uses cosmic time since formation (T = age - t_lb), not lookback."""

    # tau_bursts=0.1 Gyr = 100 Myr is well above the 1 Myr grid resolution, so
    # the CIGALE-vs-tengri discretization mismatch stays small (measured
    # D_reversed: type0 ~6e-17, type1 ~1.8e-3, type2 ~9.9e-3). Thresholds are
    # set a little above the worst measured value per type.
    @pytest.mark.parametrize(
        "burst_type,d_rev_max,d_unrev_min",
        [
            (0, 1e-6, 0.2),
            (1, 5e-3, 0.2),
            (2, 2e-2, 0.2),
        ],
    )
    def test_periodic_vs_cigale(self, burst_type, d_rev_max, d_unrev_min):
        """Compare tengri periodic against CIGALE (reversed to lookback)."""
        delta_myr = 700  # 0.7 Gyr
        tau_myr = 100  # 0.1 Gyr

        cig_sfh = cig_periodic(_AGE_MYR, burst_type, delta_myr, tau_myr)
        cig_reversed = cig_sfh[::-1]

        sfr_tengri = np.asarray(
            periodic(
                jnp.array(_LB_CENTER),
                log_total_mass=10.0,
                delta_bursts_yr=delta_myr * 1e6,
                tau_bursts_yr=tau_myr * 1e6,
                burst_type=burst_type,
                age_yr=_AGE_YR,
            )
        )

        d_reversed = _l1_distance(sfr_tengri, cig_reversed)
        d_unreversed = _l1_distance(sfr_tengri, cig_sfh)

        assert d_reversed < d_rev_max, f"Distance to CIGALE reversed: {d_reversed:.2e} (too large)"
        assert d_unreversed > d_unrev_min, (
            f"Distance to CIGALE unreversed: {d_unreversed:.2e} (should be large)"
        )


class TestBuat08Orientation:
    """Verify buat08 uses cosmic time since formation (T = age - t_lb), not lookback."""

    # Measured on the fixed code: D_reversed ~3-6e-5 for both velocities.
    # D_unreversed is 0.50 at v=100 but only 0.077 at v=250 -- buat08's SFR(T)
    # curve at high velocity has low dynamic range over an 8 Gyr window, so
    # reversing it moves less mass around in aggregate even though the
    # pointwise match to the correctly-reversed CIGALE curve is excellent
    # (D_reversed is ~1300x smaller than D_unreversed at v=250). The v=250
    # threshold is set a little below its measured D_unreversed rather than
    # forcing the generic 0.2 floor.
    @pytest.mark.parametrize(
        "velocity,d_unrev_min",
        [
            (100, 0.2),
            (250, 0.05),
        ],
    )
    def test_buat08_vs_cigale(self, velocity, d_unrev_min):
        """Compare tengri buat08 against CIGALE (reversed to lookback)."""
        cig_sfh = cig_buat08(_AGE_MYR, velocity)
        cig_reversed = cig_sfh[::-1]

        sfr_tengri = np.asarray(
            buat08(
                jnp.array(_LB_CENTER),
                10.0,
                float(velocity),
                age_yr=_AGE_YR,
            )
        )

        d_reversed = _l1_distance(sfr_tengri, cig_reversed)
        d_unreversed = _l1_distance(sfr_tengri, cig_sfh)

        assert d_reversed < 1e-3, f"Distance to CIGALE reversed: {d_reversed:.2e} (too large)"
        assert d_unreversed > d_unrev_min, (
            f"Distance to CIGALE unreversed: {d_unreversed:.2e} (should be large)"
        )
