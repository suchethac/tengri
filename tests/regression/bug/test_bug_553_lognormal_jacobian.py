"""Regression test for issue #553: lognormal SFH now has correct 1/T Jacobian.

This test verifies that the lognormal SFH implementation matches Carnall+2018
Eq. 2 exactly, including the 1/T factor that arises from the change of
variables in the lognormal probability distribution.

See: https://github.com/langmore/tengri/issues/553
     arXiv:1712.04452 (Carnall et al. 2018)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh import lognormal


@pytest.mark.regression_bug
class TestLognormalJacobianFix:
    """Verify that lognormal now has the correct 1/T Jacobian."""

    def test_lognormal_has_1_over_t_factor(self):
        """Lognormal should have 1/T asymmetry, not be symmetric in ln(T).

        A pure log-Gaussian (without 1/T factor) is symmetric in ln(T),
        peaking at ln(T_peak). The true lognormal has a 1/T factor that
        shifts the peak earlier in cosmic time.

        We verify this by checking that the SFR is NOT symmetric in ln(T)
        about the peak parameter.
        """
        age = 13.6e9  # yr
        peak = 3.0e9  # yr
        width = 0.3  # dex
        log_total_mass = 10.0

        # Create a symmetric grid in ln(T) about the peak.
        # ln(peak) is the center.
        ln_peak = np.log(peak)
        delta_ln = 1.0  # ~2.718× offset from peak in linear T

        ln_grid = np.array([ln_peak - delta_ln, ln_peak, ln_peak + delta_ln])
        T_grid = np.exp(ln_grid)

        # Convert to lookback time.
        t_lookback = age - T_grid

        # Evaluate SFR.
        sfr = lognormal(t_lookback, log_total_mass, peak, width, age)

        # For a pure log-Gaussian (without 1/T), sfr[0] would equal sfr[2]
        # (symmetry). With the 1/T factor, sfr[0] > sfr[2] (earlier times
        # are enhanced relative to later times).
        assert sfr[0] > sfr[2], (
            f"SFR should be higher at earlier cosmic time (T={T_grid[0]:.2e}) "
            f"than later (T={T_grid[2]:.2e}) due to 1/T factor. "
            f"Got sfr[0]={sfr[0]:.4f}, sfr[2]={sfr[2]:.4f}"
        )

    def test_lognormal_matches_theoretical_shape(self):
        """Verify SFR shape matches Carnall+2018 lognormal formula at sample points.

        Carnall+2018 Eq. 2: SFR(T) ∝ (1/T) * exp(-(ln(T) - μ)^2 / (2σ^2))
        where mode = exp(μ - σ^2) = peak.
        """
        age = 13.6e9  # yr
        peak = 2.5e9  # yr
        width = 0.4  # dex
        log_total_mass = 10.0

        # Create an unnormalized reference from the formula.
        T_sample = np.logspace(7, 10, 100)
        t_lookback_sample = age - T_sample

        # Parameters for the Gaussian in ln space.
        sigma_ln = width * np.log(10.0)
        ln_peak = np.log(peak)
        mu = ln_peak + sigma_ln**2

        # Reference: unnormalized (1/T) * exp(-(ln(T) - μ)^2 / (2σ^2))
        # for T > 0, else zero.
        mask = T_sample > 0
        reference = np.zeros_like(T_sample, dtype=float)
        reference[mask] = (1.0 / T_sample[mask]) * np.exp(
            -0.5 * ((np.log(T_sample[mask]) - mu) / sigma_ln) ** 2
        )

        # Evaluate the function (which includes mass normalization).
        sfr = lognormal(jnp.array(t_lookback_sample), log_total_mass, peak, width, age)

        # The function outputs are renormalized to match log_total_mass,
        # but the SHAPE should match the reference (up to normalization).
        # Normalize both to have the same integral for comparison.
        from jax.numpy import trapezoid

        sfr_integral = float(trapezoid(sfr, t_lookback_sample))
        ref_integral = np.trapezoid(reference, t_lookback_sample)

        # Relative difference in shape (after normalizing by integral).
        sfr_norm = np.array(sfr) / (sfr_integral + 1e-30)
        ref_norm = reference / (ref_integral + 1e-30)

        relative_diff = np.abs(sfr_norm - ref_norm) / (np.abs(ref_norm) + 1e-30)

        # We expect very close agreement (relative diff < 1% except in tails).
        # Use median to avoid tail regions where both are tiny.
        median_rel_diff = np.median(relative_diff[mask])
        assert median_rel_diff < 0.01, (
            f"Median relative difference in shape: {median_rel_diff:.4f}. "
            f"Expected < 1% (median); sfr and reference shapes don't match."
        )

    def test_lognormal_normalization_conserved(self):
        """Verify that log_total_mass parameter is still conserved exactly."""
        age = 13.6e9
        peak = 3.5e9
        width = 0.35
        log_total_mass = 9.5

        t_lookback = jnp.logspace(7, 10.14, 256)
        sfr = lognormal(t_lookback, log_total_mass, peak, width, age)

        # Integrate and check.
        from jax.numpy import trapezoid

        mass_int = trapezoid(sfr, t_lookback)
        expected_mass = 10.0**log_total_mass

        # Should match to machine precision.
        rel_error = abs(float(mass_int) - expected_mass) / expected_mass
        assert rel_error < 1e-6, (
            f"Mass integral mismatch: got {mass_int:.4e}, "
            f"expected {expected_mass:.4e}, relative error {rel_error:.2e}"
        )

    def test_lognormal_peak_location_sensible(self):
        """Verify the peak of the SFR is near the requested peak parameter.

        For a lognormal with 1/T factor, the mode is at exp(μ - σ^2),
        which we set to equal the peak parameter. The SFR peak should
        therefore be close to the peak parameter.
        """
        age = 13.6e9
        peak = 2.0e9
        width = 0.5
        log_total_mass = 10.0

        # Fine grid to find the peak.
        t_lookback = jnp.logspace(6, 10, 1000)
        sfr = lognormal(t_lookback, log_total_mass, peak, width, age)

        # Find the index of maximum SFR.
        max_idx = int(jnp.argmax(sfr))
        T_at_max = age - t_lookback[max_idx]

        # Peak should be within ~50% of the requested peak in cosmic time.
        # (width=0.5 dex is quite wide, so peak may be off by a bit in log space.)
        ratio = T_at_max / peak
        assert 0.5 < ratio < 2.0, (
            f"SFR peaks at T={T_at_max:.2e}, expected peak={peak:.2e}. "
            f"Ratio {ratio:.2f} is outside [0.5, 2.0]."
        )

    def test_lognormal_zero_before_formation(self):
        """Verify SFR is exactly zero at t_lookback >= age (before formation)."""
        age = 13.6e9
        peak = 3.0e9
        width = 0.3
        log_total_mass = 10.0

        # Create t_lookback values, some beyond age.
        t_lookback = jnp.array([5e9, 10e9, 13.6e9, 14.0e9, 15.0e9])
        sfr = lognormal(t_lookback, log_total_mass, peak, width, age)

        # SFR should be exactly zero where t_lookback >= age.
        mask_before_formation = t_lookback >= age
        assert jnp.all(sfr[mask_before_formation] == 0.0), (
            f"SFR should be zero before formation (t_lookback >= age). "
            f"Got {sfr[mask_before_formation]}"
        )
