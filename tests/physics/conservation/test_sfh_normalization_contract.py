# SPDX-License-Identifier: BSD-3-Clause
"""SFH normalization contract test — verify all SFH callables conserve mass.

Contract: Each parametric SFH callable with log_total_mass parameter must satisfy:
    ∫ SFR(t; log_total_mass) dt = 10^log_total_mass [Msun]

This test enforces the NEW normalization (2026-05-25) where every SFH callable
rescales its shape to achieve the target total stellar mass, not peak SFR.

References
----------
Bagpipes convention: massformed (Carnall et al.)
Prospector convention: mass_formed (Johnson et al.)
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.conservation

from scipy.integrate import trapezoid

from tengri.components.stellar.sfh.mean_sfh import (
    buat08,
    constant,
    constant_then_exponential,
    delayed_bq,
    delayed_exponential,
    dpl,
    exponential,
    gaussian,
    gaussian_burst,
    lognormal,
    periodic,
    psb_wild2020,
    skewnormal,
    snorm_burst,
    snorm_trunc_burst,
    top_hat,
    truncated_skewnormal,
)

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9

# Backwards-compatible aliases used in the parametrize tables below.
skewnormal_burst = snorm_burst
skewnormal_trunc_burst = snorm_trunc_burst

# Standard lookback time grid for integration: 10 kyr to 14 Gyr
T_LOOKBACK = np.linspace(1e5, 14e9, 2000)


class TestSFHNormalizationContract:
    """Every parametric SFH must conserve total stellar mass: integral = 10^log_total_mass."""

    @staticmethod
    def _check_normalization(sfh_fn, t_grid, log_total_mass, **kwargs):
        """Verify integral of SFH equals target mass within 1%."""
        sfr = np.array(sfh_fn(jnp.array(t_grid), log_total_mass=log_total_mass, **kwargs))
        integral = float(trapezoid(sfr, t_grid))
        expected = 10.0**log_total_mass
        relative_error = abs(integral - expected) / expected
        assert relative_error < 0.01, (
            f"Integral {integral:.2e} != expected {expected:.2e} (error={relative_error:.1%})"
        )

    def test_exponential_normalization(self):
        """exponential(log_total_mass) must conserve mass."""
        self._check_normalization(exponential, T_LOOKBACK, log_total_mass=1.0, tau=2e9)
        self._check_normalization(exponential, T_LOOKBACK, log_total_mass=0.5, tau=5e9)
        self._check_normalization(exponential, T_LOOKBACK, log_total_mass=2.5, tau=1e9)

    def test_delayed_exponential_normalization(self):
        """delayed_exponential(log_total_mass) must conserve mass."""
        self._check_normalization(delayed_exponential, T_LOOKBACK, log_total_mass=1.0, tau=3e9)
        self._check_normalization(delayed_exponential, T_LOOKBACK, log_total_mass=0.5, tau=5e9)
        self._check_normalization(
            delayed_exponential, T_LOOKBACK, log_total_mass=2.0, tau=2e9, start=1e9
        )

    def test_dpl_normalization(self):
        """dpl(log_total_mass) must conserve mass."""
        self._check_normalization(
            dpl, T_LOOKBACK, log_total_mass=1.0, alpha=1.5, beta=2.0, tau=3e9, age=_AGE_UNIV_YR
        )
        self._check_normalization(
            dpl, T_LOOKBACK, log_total_mass=0.5, alpha=2.0, beta=1.5, tau=2e9, age=_AGE_UNIV_YR
        )
        self._check_normalization(
            dpl, T_LOOKBACK, log_total_mass=2.5, alpha=1.0, beta=1.0, tau=5e9, age=_AGE_UNIV_YR
        )

    def test_gaussian_normalization(self):
        """gaussian(log_total_mass) must conserve mass."""
        self._check_normalization(
            gaussian, T_LOOKBACK, log_total_mass=1.0, peak_lbt=5e9, width=2e9
        )
        self._check_normalization(
            gaussian, T_LOOKBACK, log_total_mass=0.5, peak_lbt=3e9, width=1e9
        )
        self._check_normalization(
            gaussian, T_LOOKBACK, log_total_mass=2.0, peak_lbt=8e9, width=3e9
        )

    def test_lognormal_normalization(self):
        """lognormal(log_total_mass) must conserve mass.

        ``width`` is the log-space (dex) standard deviation, not a time —
        see the registry prior ``Uniform(0.1, 2.0)`` and the function
        docstring (sigma = width x ln 10). ``peak``/``age`` are in years.
        """
        self._check_normalization(
            lognormal, T_LOOKBACK, log_total_mass=1.0, peak=5e9, width=0.3, age=_AGE_UNIV_YR
        )
        self._check_normalization(
            lognormal, T_LOOKBACK, log_total_mass=0.5, peak=6e9, width=0.5, age=_AGE_UNIV_YR
        )
        self._check_normalization(
            lognormal, T_LOOKBACK, log_total_mass=2.0, peak=4e9, width=0.2, age=_AGE_UNIV_YR
        )

    def test_truncated_skewnormal_normalization(self):
        """truncated_skewnormal(log_total_mass) must conserve mass."""
        self._check_normalization(
            truncated_skewnormal,
            T_LOOKBACK,
            log_total_mass=1.0,
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
            trunc=3.0,
        )
        self._check_normalization(
            truncated_skewnormal,
            T_LOOKBACK,
            log_total_mass=0.5,
            peak_lbt=6e9,
            width=1.5e9,
            skew=0.5,
            trunc=2.0,
        )

    def test_skewnormal_normalization(self):
        """skewnormal(log_total_mass) must conserve mass."""
        self._check_normalization(
            skewnormal,
            T_LOOKBACK,
            log_total_mass=1.0,
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
        )
        self._check_normalization(
            skewnormal,
            T_LOOKBACK,
            log_total_mass=0.5,
            peak_lbt=6e9,
            width=1.5e9,
            skew=0.5,
        )
        self._check_normalization(
            skewnormal,
            T_LOOKBACK,
            log_total_mass=2.0,
            peak_lbt=4e9,
            width=2e9,
            skew=-0.3,
        )

    def test_skewnormal_burst_normalization(self):
        """skewnormal_burst(log_total_mass) must conserve mass."""
        self._check_normalization(
            skewnormal_burst,
            T_LOOKBACK,
            log_total_mass=1.0,
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
            burst_sfr=0.5,
            burst_age=500e6,
        )
        self._check_normalization(
            skewnormal_burst,
            T_LOOKBACK,
            log_total_mass=0.5,
            peak_lbt=6e9,
            width=1.5e9,
            skew=0.3,
            burst_sfr=1.0,
            burst_age=100e6,
        )

    def test_truncated_skewnormal_burst_normalization(self):
        """truncated_skewnormal_burst(log_total_mass) must conserve mass."""
        self._check_normalization(
            skewnormal_trunc_burst,
            T_LOOKBACK,
            log_total_mass=1.0,
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
            trunc=3.0,
            burst_sfr=0.5,
            burst_age=500e6,
        )
        self._check_normalization(
            skewnormal_trunc_burst,
            T_LOOKBACK,
            log_total_mass=0.5,
            peak_lbt=6e9,
            width=1.5e9,
            skew=0.2,
            trunc=2.5,
            burst_sfr=0.8,
            burst_age=200e6,
        )

    def test_psb_wild2020_normalization(self):
        """psb_wild2020(log_total_mass) must conserve mass."""
        self._check_normalization(
            psb_wild2020,
            T_LOOKBACK,
            log_total_mass=1.0,
            age=10e9,
            tau=3e9,
            burstage=500e6,
            alpha=2.0,
            beta=1.0,
            fburst=0.5,
        )
        self._check_normalization(
            psb_wild2020,
            T_LOOKBACK,
            log_total_mass=0.5,
            age=12e9,
            tau=2e9,
            burstage=100e6,
            alpha=2.0,
            beta=1.0,
            fburst=0.3,
        )
        self._check_normalization(
            psb_wild2020,
            T_LOOKBACK,
            log_total_mass=2.0,
            age=9e9,
            tau=5e9,
            burstage=800e6,
            alpha=2.0,
            beta=1.0,
            fburst=0.1,
        )

    def test_constant_normalization(self):
        """constant(log_total_mass) must conserve mass."""
        self._check_normalization(constant, T_LOOKBACK, log_total_mass=1.0, start=1e9, end=10e9)
        self._check_normalization(constant, T_LOOKBACK, log_total_mass=0.5, start=2e9, end=9e9)
        self._check_normalization(constant, T_LOOKBACK, log_total_mass=2.5, start=0.5e9, end=13e9)

    def test_constant_then_exponential_normalization(self):
        """constant_then_exponential(log_total_mass) must conserve mass."""
        self._check_normalization(
            constant_then_exponential,
            T_LOOKBACK,
            log_total_mass=1.0,
            tau=2e9,
            quench_age=5e9,
            age=10e9,
        )
        self._check_normalization(
            constant_then_exponential,
            T_LOOKBACK,
            log_total_mass=0.5,
            tau=3e9,
            quench_age=6e9,
            age=11e9,
        )
        self._check_normalization(
            constant_then_exponential,
            T_LOOKBACK,
            log_total_mass=2.0,
            tau=1e9,
            quench_age=4e9,
            age=9e9,
        )

    def test_top_hat_normalization(self):
        """top_hat(log_total_mass) must conserve mass."""
        self._check_normalization(
            top_hat, T_LOOKBACK, log_total_mass=1.0, t_start=5e9, t_end=3e9, smooth_width=5e8
        )
        self._check_normalization(
            top_hat, T_LOOKBACK, log_total_mass=0.5, t_start=6e9, t_end=2e9, smooth_width=1e8
        )
        self._check_normalization(
            top_hat, T_LOOKBACK, log_total_mass=2.0, t_start=10e9, t_end=5e9, smooth_width=2e8
        )

    def test_gaussian_burst_normalization(self):
        """gaussian_burst(log_total_mass) must conserve mass."""
        self._check_normalization(
            gaussian_burst, T_LOOKBACK, log_total_mass=1.0, t_peak=5e9, sigma=1e8
        )
        self._check_normalization(
            gaussian_burst, T_LOOKBACK, log_total_mass=0.5, t_peak=3e9, sigma=2e8
        )
        self._check_normalization(
            gaussian_burst, T_LOOKBACK, log_total_mass=2.0, t_peak=8e9, sigma=5e8
        )

    def test_delayed_bq_normalization(self):
        """delayed_bq(log_total_mass) must conserve mass."""
        self._check_normalization(
            delayed_bq,
            T_LOOKBACK,
            log_total_mass=1.0,
            tau_main_yr=2e9,
            age_main_yr=5e9,
            age_bq_yr=500e6,
            r_sfr=0.5,
        )
        self._check_normalization(
            delayed_bq,
            T_LOOKBACK,
            log_total_mass=0.5,
            tau_main_yr=3e9,
            age_main_yr=6e9,
            age_bq_yr=1e9,
            r_sfr=1.5,
        )
        self._check_normalization(
            delayed_bq,
            T_LOOKBACK,
            log_total_mass=2.0,
            tau_main_yr=1e9,
            age_main_yr=8e9,
            age_bq_yr=200e6,
            r_sfr=0.1,
        )

    def test_periodic_normalization(self):
        """periodic(log_total_mass) must conserve mass."""
        self._check_normalization(
            periodic,
            T_LOOKBACK,
            log_total_mass=1.0,
            delta_bursts_yr=0.5e9,
            tau_bursts_yr=0.05e9,
            burst_type=2,
            age_yr=10e9,
        )
        self._check_normalization(
            periodic,
            T_LOOKBACK,
            log_total_mass=0.5,
            delta_bursts_yr=0.2e9,
            tau_bursts_yr=0.01e9,
            burst_type=2,
            age_yr=8e9,
        )
        self._check_normalization(
            periodic,
            T_LOOKBACK,
            log_total_mass=2.0,
            delta_bursts_yr=1.0e9,
            tau_bursts_yr=0.1e9,
            burst_type=2,
            age_yr=12e9,
        )

    def test_buat08_normalization(self):
        """buat08(log_total_mass) must conserve mass."""
        self._check_normalization(buat08, T_LOOKBACK, log_total_mass=1.0, velocity_km_s=200.0)
        self._check_normalization(buat08, T_LOOKBACK, log_total_mass=0.5, velocity_km_s=100.0)
        self._check_normalization(buat08, T_LOOKBACK, log_total_mass=2.0, velocity_km_s=300.0)

    def test_normalization_scales_linearly(self):
        """Doubling log_total_mass should double the integral (linear scaling)."""
        sfr1 = np.array(
            dpl(
                jnp.array(T_LOOKBACK),
                log_total_mass=1.0,
                alpha=1.5,
                beta=2.0,
                tau=3e9,
                age=_AGE_UNIV_YR,
            )
        )
        sfr2 = np.array(
            dpl(
                jnp.array(T_LOOKBACK),
                log_total_mass=2.0,
                alpha=1.5,
                beta=2.0,
                tau=3e9,
                age=_AGE_UNIV_YR,
            )
        )
        integral1 = float(trapezoid(sfr1, T_LOOKBACK))
        integral2 = float(trapezoid(sfr2, T_LOOKBACK))
        ratio = integral2 / integral1
        np.testing.assert_allclose(ratio, 10.0, rtol=0.01)

    def test_shape_invariance_across_normalizations(self):
        """SFR shapes should be identical, only amplitude scales with log_total_mass."""
        # Compute SFR for two different normalizations
        sfr1 = np.array(
            gaussian(jnp.array(T_LOOKBACK), log_total_mass=1.0, peak_lbt=5e9, width=2e9)
        )
        sfr2 = np.array(
            gaussian(jnp.array(T_LOOKBACK), log_total_mass=2.0, peak_lbt=5e9, width=2e9)
        )
        # Ratio should be constant (10x) everywhere
        ratio = sfr2 / np.where(sfr1 > 1e-10, sfr1, 1.0)
        ratio_nonzero = ratio[sfr1 > 1e-10]
        np.testing.assert_allclose(ratio_nonzero, 10.0, rtol=1e-10)
