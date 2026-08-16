# SPDX-License-Identifier: BSD-3-Clause
"""Comprehensive tests for DSPS-backed cosmology utilities.

Tests cover all cosmology functions including flexible API, backward compat,
consistency checks, and numerical accuracy.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds

from dsps.cosmology import CosmoParams
from numpy.testing import assert_allclose

from tengri.utils.cosmology import (
    DEFAULT_H0,
    DEFAULT_OM0,
    PLANCK18,
    age_at_z,
    age_at_z0,
    angular_diameter_distance,
    angular_diameter_distance_mpc,
    arcsec_per_kpc,
    comoving_distance,
    comoving_distance_mpc,
    comoving_volume_element,
    distance_modulus,
    kpc_per_arcsec,
    lookback_time,
    luminosity_distance,
    luminosity_distance_mpc,
)
from tengri.utils.physics_constants import MPC_CM, PC_CM


class TestResolvesCosmo:
    """Test _resolve_cosmo behavior via public functions."""

    def test_default_returns_planck18(self):
        """No args should use PLANCK18."""
        result = luminosity_distance(0.5)
        expected = luminosity_distance(0.5, cosmo=PLANCK18)
        assert jnp.allclose(result, expected)

    def test_h0_om0_kwargs_create_correct_cosmo(self):
        """h0 and om0 kwargs should create correct CosmoParams (h = H0/100)."""
        # h0=70 km/s/Mpc -> h=0.7
        result = luminosity_distance(0.5, h0=70.0, om0=0.3)
        # Should differ from default (67.66, 0.30966)
        expected = luminosity_distance(0.5, h0=DEFAULT_H0, om0=DEFAULT_OM0)
        assert not jnp.allclose(result, expected)

    def test_cosmo_object_passes_through(self):
        """CosmoParams object should be used directly."""
        custom_cosmo = CosmoParams(Om0=0.3, w0=-1.0, wa=0.0, h=0.7)
        result = luminosity_distance(0.5, cosmo=custom_cosmo)
        expected = luminosity_distance(0.5, h0=70.0, om0=0.3)
        assert jnp.allclose(result, expected)

    def test_positional_args_take_priority(self):
        """Positional h0/om0 should override cosmo kwarg."""
        # Create a custom cosmo, but pass positional args that differ
        custom_cosmo = CosmoParams(Om0=0.3, w0=-1.0, wa=0.0, h=0.7)
        result = luminosity_distance(0.5, 70.0, 0.3, cosmo=custom_cosmo)
        # Should use positional args (70.0, 0.3), not custom_cosmo
        expected = luminosity_distance(0.5, h0=70.0, om0=0.3)
        assert jnp.allclose(result, expected)


# z: (DC [Mpc], DL [Mpc], age(z) [Gyr], lookback [Gyr]) frozen from
# astropy 7.2.0 Planck18 (Planck Collaboration 2020, A&A 641, A6,
# TT,TE,EE+lowE+lensing — the parameter set PLANCK18 pins, see #401).
_PLANCK18_ASTROPY_REFERENCE = {
    0.1: (432.5657, 475.8223, 12.441687, 1.345198),
    0.5: (1946.4166, 2919.6250, 8.590646, 5.196240),
    1.0: (3395.6345, 6791.2689, 5.851343, 7.935542),
    2.0: (5308.1889, 15924.5667, 3.276830, 10.510055),
    5.0: (7946.2934, 47677.7604, 1.170706, 12.616179),
    10.0: (9636.2851, 105999.1359, 0.471415, 13.315471),
}


class TestPlanck18FrozenReference:
    """Distances and times match astropy.cosmology.Planck18 reference values.

    DSPS integrates the Friedmann equation on a fixed quadrature grid, which
    agrees with astropy to ≤0.7% for z ≤ 10, hence rtol=1e-2 (the
    scalar-physics tolerance). Catches any unit, h-convention, or
    (1+z)-factor error outright.
    """

    @pytest.mark.parametrize("z", sorted(_PLANCK18_ASTROPY_REFERENCE))
    def test_distances_and_times_match_astropy_planck18(self, z):
        dc_ref, dl_ref, age_ref, tlb_ref = _PLANCK18_ASTROPY_REFERENCE[z]
        assert_allclose(float(comoving_distance_mpc(z)), dc_ref, rtol=1e-2)
        assert_allclose(float(luminosity_distance_mpc(z)), dl_ref, rtol=1e-2)
        assert_allclose(float(age_at_z(z)), age_ref, rtol=1e-2)
        assert_allclose(float(lookback_time(z)), tlb_ref, rtol=1e-2)


class TestDistances:
    """Test distance functions and consistency."""

    def test_luminosity_distance_mpc_matches(self):
        """luminosity_distance_mpc should match luminosity_distance / MPC_CM."""
        z = 0.5
        dl_cm = luminosity_distance(z)
        dl_mpc = luminosity_distance_mpc(z)
        assert jnp.allclose(dl_cm, dl_mpc * MPC_CM)

    def test_luminosity_distance_relation(self):
        """luminosity_distance = comoving_distance × (1+z)."""
        z = 0.5
        dl_cm = luminosity_distance(z)
        dc_cm = comoving_distance(z)
        expected = dc_cm * (1.0 + z)
        assert jnp.allclose(dl_cm, expected, rtol=1e-10)

    def test_angular_diameter_distance_relation(self):
        """angular_diameter_distance = comoving_distance / (1+z)."""
        z = 0.5
        da_cm = angular_diameter_distance(z)
        dc_cm = comoving_distance(z)
        expected = dc_cm / (1.0 + z)
        assert jnp.allclose(da_cm, expected, rtol=1e-10)

    def test_comoving_distance_mpc_matches(self):
        """comoving_distance_mpc should match comoving_distance / MPC_CM."""
        z = 0.5
        dc_cm = comoving_distance(z)
        dc_mpc = comoving_distance_mpc(z)
        assert jnp.allclose(dc_cm, dc_mpc * MPC_CM)

    def test_angular_diameter_distance_mpc_matches(self):
        """angular_diameter_distance_mpc should match ad / MPC_CM."""
        z = 0.5
        da_cm = angular_diameter_distance(z)
        da_mpc = angular_diameter_distance_mpc(z)
        assert jnp.allclose(da_cm, da_mpc * MPC_CM)

    def test_distance_decreases_with_h0(self):
        """Comoving distance should decrease with increasing H0."""
        z = 0.5
        dc_low_h0 = comoving_distance(z, h0=65.0, om0=0.315)
        dc_high_h0 = comoving_distance(z, h0=75.0, om0=0.315)
        assert dc_low_h0 > dc_high_h0

    def test_distance_decreases_with_om0(self):
        """Comoving distance decreases with Ω_m at fixed H0.

        In flat ΛCDM, raising Ω_m trades dark energy for matter, so H(z)
        is larger at every z > 0 and D_C = ∫ c dz / H(z) shrinks.
        """
        z = 0.5
        dc_low_om = comoving_distance(z, h0=67.4, om0=0.2)
        dc_high_om = comoving_distance(z, h0=67.4, om0=0.4)
        assert dc_low_om > dc_high_om


class TestDistanceModulus:
    """Test distance modulus calculations."""

    def test_distance_modulus_increases_with_z(self):
        """Distance modulus should increase with redshift."""
        z1, z2, z3 = 0.1, 0.5, 1.0
        mu1 = distance_modulus(z1)
        mu2 = distance_modulus(z2)
        mu3 = distance_modulus(z3)
        assert mu1 < mu2 < mu3

    def test_distance_modulus_consistency(self):
        """μ = 5 log10(d_L_pc) - 5."""
        z = 0.5
        mu = distance_modulus(z)
        dl_cm = luminosity_distance(z)
        dl_pc = dl_cm / PC_CM
        expected = 5.0 * jnp.log10(dl_pc) - 5.0
        assert jnp.allclose(mu, expected, rtol=1e-10)


class TestTimes:
    """Test age and lookback time functions."""

    def test_lookback_time_at_z0_is_zero(self):
        """Lookback time at z=0 should be ~0."""
        t_lb = lookback_time(0.0)
        assert jnp.abs(t_lb) < 1e-6

    def test_lookback_time_increases_with_z(self):
        """Lookback time should increase with z."""
        z_vals = jnp.array([0.1, 0.5, 1.0, 2.0])
        ts = jnp.array([lookback_time(z) for z in z_vals])
        assert jnp.all(jnp.diff(ts) > 0.0)

    def test_lookback_time_less_than_universe_age(self):
        """Lookback time should be less than current universe age."""
        age0 = age_at_z0()
        z_vals = jnp.array([0.1, 0.5, 1.0, 2.0])
        ts = jnp.array([lookback_time(z) for z in z_vals])
        assert jnp.all(ts < age0)

    def test_age_at_z0_planck18(self):
        """Age of universe with PLANCK18 is 13.787 ± 0.020 Gyr (Planck 2020).

        Scalar output shape is part of the contract (DSPS returns a 0-d
        array; the wrapper unwraps it).
        """
        age0 = age_at_z0()
        assert jnp.ndim(age0) == 0
        assert_allclose(float(age0), 13.787, rtol=1e-2)

    def test_age_at_z_array_matches_scalar_calls(self):
        """age_at_z on an array equals the elementwise scalar results exactly.

        Subsumes the old scalar-shape / array-shape-only tests.
        """
        z_vals = jnp.array([0.0, 0.5, 1.0])
        ages = age_at_z(z_vals)
        chex.assert_equal_shape([ages, z_vals])
        assert jnp.ndim(age_at_z(0.5)) == 0
        expected = jnp.stack([age_at_z(float(z)) for z in z_vals])
        chex.assert_trees_all_close(ages, expected, rtol=1e-12)

    def test_age_at_z_at_z0(self):
        """age_at_z(0.0) should equal age_at_z0()."""
        age_at_0 = age_at_z(0.0)
        age0 = age_at_z0()
        assert jnp.allclose(age_at_0, age0, rtol=1e-10)

    def test_age_universe_relation(self):
        """age_at_z(z) + lookback_time(z) ≈ age_at_z0()."""
        z = 0.5
        age_z = age_at_z(z)
        lookback = lookback_time(z)
        age0 = age_at_z0()
        assert jnp.allclose(age_z + lookback, age0, rtol=1e-10)

    def test_age_decreases_with_z(self):
        """Age of universe decreases with increasing redshift."""
        z_vals = jnp.array([0.0, 0.5, 1.0, 2.0])
        ages = jnp.array([age_at_z(z) for z in z_vals])
        assert jnp.all(jnp.diff(ages) < 0.0)


class TestAngularScales:
    """Test angular scale functions."""

    def test_arcsec_and_kpc_are_inverses(self):
        """kpc_per_arcsec ≈ 1 / arcsec_per_kpc."""
        z = 0.8
        a2k = arcsec_per_kpc(z)
        k2a = kpc_per_arcsec(z)
        assert jnp.allclose(a2k * k2a, 1.0, rtol=1e-10)

    def test_arcsec_per_kpc_at_z_is_plausible(self):
        """arcsec_per_kpc at z~1 should be ~0.12 arcsec/kpc with Planck18."""
        z = 1.0
        scale = arcsec_per_kpc(z)
        # At z~1, ~0.12 arcsec/kpc (inverse of ~8.2 kpc/arcsec)
        assert 0.10 < scale < 0.15

    def test_kpc_per_arcsec_at_z_is_plausible(self):
        """kpc_per_arcsec at z~1 should be ~8.2 kpc/arcsec with Planck18."""
        z = 1.0
        scale = kpc_per_arcsec(z)
        # Inverse of arcsec_per_kpc; at z~1, ~8.2 kpc/arcsec
        assert 8.0 < scale < 8.5

    def test_angular_scale_changes_with_redshift(self):
        """Angular scale should change with redshift."""
        z1, z2 = 0.5, 1.5
        k2a_1 = kpc_per_arcsec(z1)
        k2a_2 = kpc_per_arcsec(z2)
        assert not jnp.allclose(k2a_1, k2a_2)


class TestFlexibleAPI:
    """Test flexible API with different input methods."""

    def test_default_same_as_planck18(self):
        """luminosity_distance(z) should equal luminosity_distance(z, cosmo=PLANCK18)."""
        z = 0.5
        result1 = luminosity_distance(z)
        result2 = luminosity_distance(z, cosmo=PLANCK18)
        assert jnp.allclose(result1, result2)

    def test_h0_om0_kwargs_match_cosmo(self):
        """luminosity_distance(z, h0=h, om0=o) should match with cosmo object."""
        z = 0.5
        h0, om0 = 67.66, 0.30966
        result1 = luminosity_distance(z, h0=h0, om0=om0)
        cosmo = CosmoParams(Om0=om0, w0=-1.0, wa=0.0, h=h0 / 100.0)
        result2 = luminosity_distance(z, cosmo=cosmo)
        assert jnp.allclose(result1, result2)

    def test_different_h0_om0_differs(self):
        """Different h0/om0 should produce different results."""
        z = 0.5
        result1 = luminosity_distance(z, h0=67.66, om0=0.30966)
        result2 = luminosity_distance(z, h0=70.0, om0=0.3)
        assert not jnp.allclose(result1, result2)

    def test_all_functions_support_flexible_api(self):
        """All distance functions should support flexible API."""
        z = 0.5
        funcs = [
            luminosity_distance,
            comoving_distance,
            angular_diameter_distance,
            lookback_time,
            age_at_z,
            comoving_volume_element,
            arcsec_per_kpc,
            kpc_per_arcsec,
        ]
        for func in funcs:
            result1 = func(z)
            result2 = func(z, h0=67.66, om0=0.30966)
            assert jnp.allclose(result1, result2), f"{func.__name__} failed"

    def test_distance_modulus_supports_flexible_api(self):
        """distance_modulus should support flexible API."""
        z = 0.5
        result1 = distance_modulus(z)
        result2 = distance_modulus(z, h0=67.66, om0=0.30966)
        assert jnp.allclose(result1, result2)


class TestBackwardCompat:
    """Test backward compatibility with positional arguments."""

    def test_luminosity_distance_positional_args(self):
        """luminosity_distance(z, h0, om0) should work (positional args)."""
        result = luminosity_distance(0.5, 67.66, 0.30966)
        expected = luminosity_distance(0.5, h0=67.66, om0=0.30966)
        assert jnp.allclose(result, expected)

    def test_age_at_z_positional_args(self):
        """age_at_z(z, h0, om0) should work (positional args)."""
        result = age_at_z(0.5, 67.66, 0.30966)
        expected = age_at_z(0.5, h0=67.66, om0=0.30966)
        assert jnp.allclose(result, expected)

    def test_comoving_distance_positional_args(self):
        """comoving_distance(z, h0, om0) should work (positional args)."""
        result = comoving_distance(0.5, 67.66, 0.30966)
        expected = comoving_distance(0.5, h0=67.66, om0=0.30966)
        assert jnp.allclose(result, expected)

    def test_distance_modulus_positional_args(self):
        """distance_modulus(z, h0, om0) should work (positional args)."""
        result = distance_modulus(0.5, 67.66, 0.30966)
        expected = distance_modulus(0.5, h0=67.66, om0=0.30966)
        assert jnp.allclose(result, expected)

    def test_lookback_time_positional_args(self):
        """lookback_time(z, h0, om0) should work (positional args)."""
        result = lookback_time(0.5, 67.66, 0.30966)
        expected = lookback_time(0.5, h0=67.66, om0=0.30966)
        assert jnp.allclose(result, expected)


class TestComovingVolumeElement:
    """Test differential comoving volume element."""

    def test_comoving_volume_element_increases_with_z(self):
        """Comoving volume element is positive and increases with z (LCDM, z ≲ 2)."""
        z_vals = jnp.array([0.1, 0.5, 1.0, 2.0])
        dvs = jnp.array([comoving_volume_element(z) for z in z_vals])
        assert jnp.all(dvs > 0.0)
        assert jnp.all(jnp.diff(dvs) > 0.0)


class TestConsistencyWithDSPS:
    """Test that values match expected DSPS outputs."""

    def test_age_at_z0_planck18_vs_dsps(self):
        """age_at_z0() should match DSPS directly."""
        from dsps.cosmology.flat_wcdm import age_at_z0 as dsps_age_at_z0

        age0 = age_at_z0(cosmo=PLANCK18)
        expected = dsps_age_at_z0(PLANCK18.Om0, PLANCK18.w0, PLANCK18.wa, PLANCK18.h)
        assert jnp.allclose(age0, expected, rtol=1e-12)

    def test_luminosity_distance_vs_dsps(self):
        """luminosity_distance should match DSPS directly."""
        from dsps.cosmology.flat_wcdm import luminosity_distance_to_z as dsps_dl

        z = 0.5
        dl = luminosity_distance_mpc(z, cosmo=PLANCK18)
        expected = dsps_dl(z, PLANCK18.Om0, PLANCK18.w0, PLANCK18.wa, PLANCK18.h)
        assert jnp.allclose(dl, expected, rtol=1e-12)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_z_equals_zero(self):
        """Functions should handle z=0 gracefully."""
        from tengri.utils.physics_constants import TEN_PC_CM

        # Luminosity distance at z=0 should be 10 pc (optical absolute mag convention)
        dl = luminosity_distance(0.0)
        assert jnp.allclose(dl, TEN_PC_CM, rtol=1e-6)

        # Comoving distance at z=0 should be 0
        dc = comoving_distance(0.0)
        assert jnp.allclose(dc, 0.0, atol=1e-10)

        # Angular diameter distance at z=0 should be 0
        da = angular_diameter_distance(0.0)
        assert jnp.allclose(da, 0.0, atol=1e-10)

    def test_very_small_redshift_recovers_hubble_law(self):
        """As z → 0⁺ the luminosity distance recovers the Hubble law.

        Limit test: d_L → c z / H0 to leading order in z, so at z = 1e-6 the
        relative deviation must be below 1e-4 (corrections are O(z)).
        """
        z = 1e-6
        dl_mpc = float(luminosity_distance_mpc(z))
        hubble_mpc = z * 299792.458 / DEFAULT_H0  # c [km/s] / H0 [km/s/Mpc]
        assert_allclose(dl_mpc, hubble_mpc, rtol=1e-4)
        assert jnp.isfinite(distance_modulus(z))


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_luminosity_distance_monotonic(self):
        """Luminosity distance should be monotonically increasing."""
        z_vals = jnp.linspace(0.0, 3.0, 50)
        dls = jnp.array([luminosity_distance(z) for z in z_vals])
        # Check monotonicity
        assert jnp.all(jnp.diff(dls) >= -1e-10)  # Allow for numerical error

    def test_age_at_z_monotonic_decreasing(self):
        """Age at z should be monotonically decreasing."""
        z_vals = jnp.linspace(0.0, 3.0, 50)
        ages = jnp.array([age_at_z(z) for z in z_vals])
        # Check monotonicity (decreasing)
        assert jnp.all(jnp.diff(ages) <= 1e-10)  # Allow for numerical error

    def test_distance_modulus_smooth(self):
        """Distance modulus should vary smoothly and monotonically."""
        z_vals = jnp.linspace(0.1, 3.0, 50)
        mus = jnp.array([distance_modulus(z) for z in z_vals])
        # Check that differences between adjacent points are monotonic
        diffs = jnp.diff(mus)
        assert jnp.all(diffs > 0.0)  # Monotonically increasing
        # At low z, steps can be ~1 mag per 0.06 dz, so allow larger step sizes
        assert jnp.all(diffs < 2.0)  # Reasonable step size


# ─────────────────────────────────────────────────────────────────────
# Redshift inversion: z_at_cosmic_time and z_at_lookback_time
# ─────────────────────────────────────────────────────────────────────


class TestZAtCosmicTime:
    """Tests for z_at_cosmic_time (inverse of age_at_z)."""

    def test_round_trip_scalar(self):
        """z_at_cosmic_time(age_at_z(z)) ≈ z for scalar z."""
        from tengri.utils.cosmology import age_at_z, z_at_cosmic_time

        for z_true in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            t = float(age_at_z(z_true))
            z_rec = float(z_at_cosmic_time(t))
            assert_allclose(z_rec, z_true, atol=0.01, rtol=0.01)

    def test_round_trip_array(self):
        """z_at_cosmic_time works with array input."""
        from tengri.utils.cosmology import age_at_z, z_at_cosmic_time

        z_true = jnp.array([0.5, 1.0, 2.0, 5.0])
        t_arr = jnp.array([float(age_at_z(z)) for z in z_true])
        z_rec = z_at_cosmic_time(t_arr)
        assert_allclose(z_rec, z_true, atol=0.01, rtol=0.01)

    def test_z_at_t0_is_zero(self):
        """Cosmic time = age of universe → z = 0."""
        from tengri.utils.cosmology import age_at_z0, z_at_cosmic_time

        t0 = age_at_z0()
        z = float(z_at_cosmic_time(t0))
        assert_allclose(z, 0.0, atol=0.01)

    def test_early_times_give_high_z(self):
        """Very early cosmic time → high redshift."""
        from tengri.utils.cosmology import z_at_cosmic_time

        z = float(z_at_cosmic_time(0.5))  # 0.5 Gyr after Big Bang
        assert z > 5.0, f"z at 0.5 Gyr should be > 5, got {z}"

    def test_monotonic(self):
        """z decreases as cosmic time increases."""
        from tengri.utils.cosmology import z_at_cosmic_time

        t_grid = jnp.linspace(0.5, 13.0, 20)
        z_grid = z_at_cosmic_time(t_grid)
        diffs = jnp.diff(z_grid)
        assert jnp.all(diffs < 0), "z should decrease with cosmic time"

    def test_flexible_api(self):
        """Accepts h0/om0 kwargs and cosmo object."""
        from tengri.utils.cosmology import PLANCK18, z_at_cosmic_time

        z1 = float(z_at_cosmic_time(5.0))
        z2 = float(z_at_cosmic_time(5.0, cosmo=PLANCK18))
        z3 = float(z_at_cosmic_time(5.0, h0=67.66, om0=0.30966))
        assert_allclose(z1, z2, rtol=1e-10)
        assert_allclose(z1, z3, rtol=0.01)


class TestZAtLookbackTime:
    """Tests for z_at_lookback_time (inverse of lookback_time)."""

    def test_round_trip_scalar(self):
        """z_at_lookback_time(lookback_time(z)) ≈ z."""
        from tengri.utils.cosmology import lookback_time, z_at_lookback_time

        for z_true in [0.0, 0.5, 1.0, 2.0, 5.0]:
            t_lb = float(lookback_time(z_true))
            z_rec = float(z_at_lookback_time(t_lb))
            assert_allclose(z_rec, z_true, atol=0.01, rtol=0.01)

    def test_zero_lookback_is_z0(self):
        """Lookback time = 0 → z = 0 (now)."""
        from tengri.utils.cosmology import z_at_lookback_time

        z = float(z_at_lookback_time(0.0))
        assert_allclose(z, 0.0, atol=0.01)

    def test_large_lookback_is_high_z(self):
        """Large lookback time → high redshift."""
        from tengri.utils.cosmology import z_at_lookback_time

        z = float(z_at_lookback_time(12.5))  # 12.5 Gyr ago
        assert z > 3.0, f"z at 12.5 Gyr lookback should be > 3, got {z}"

    def test_monotonic(self):
        """z increases with lookback time."""
        from tengri.utils.cosmology import z_at_lookback_time

        t_lb = jnp.linspace(0.0, 12.0, 20)
        z_grid = z_at_lookback_time(t_lb)
        diffs = jnp.diff(z_grid)
        assert jnp.all(diffs > 0), "z should increase with lookback time"

    def test_array_input(self):
        """Works with array input."""
        from tengri.utils.cosmology import z_at_lookback_time

        t_lb = jnp.array([0.0, 5.0, 8.0, 12.0])
        z_arr = z_at_lookback_time(t_lb)
        chex.assert_shape(z_arr, (4,))
        chex.assert_tree_all_finite(z_arr)
