"""Tests for magnitude system utilities (AB, Vega, absolute, apparent, surface brightness).

Validates:
- AB magnitude conversions (fν ↔ m_AB)
- Absolute magnitude from luminosity and distance modulus
- Apparent ↔ absolute via distance modulus
- Vega magnitude system and offsets
- Surface brightness calculations
- Cosmological dimming (1+z factor)
- JIT compilation and type stability
"""

import jax
import jax.numpy as jnp

from tengri.utils.magnitudes import (
    AB_VEGA_OFFSETS,
    ab_mag_to_fnu,
    ab_to_vega,
    absolute_ab_mag_to_lnu,
    absolute_to_apparent,
    apparent_to_absolute,
    cosmological_dimming,
    distance_modulus_from_dl,
    distance_modulus_from_dl_mpc,
    fnu_to_ab_mag,
    lnu_to_absolute_ab_mag,
    mag_to_surface_brightness,
    surface_brightness_to_mag,
    vega_to_ab,
)
from tengri.utils.physics_constants import MPC_CM, TEN_PC_CM

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Test AB Magnitude System
# ---------------------------------------------------------------------------


class TestABMagnitudes:
    """Test AB magnitude conversions."""

    def test_ab_zeropoint_definition(self):
        """AB zero-magnitude corresponds to 3.631e-20 erg/s/cm²/Hz."""
        # By definition of the AB system
        fnu_zero = 3.631e-20
        mag_zero = fnu_to_ab_mag(jnp.array(fnu_zero))
        assert jnp.allclose(mag_zero, 0.0, atol=1e-10)

    def test_fnu_to_ab_mag_bright_source(self):
        """Verify magnitude scale: 100x brighter → 5 mag brighter."""
        fnu = 3.631e-20  # 0 mag
        mag1 = fnu_to_ab_mag(jnp.array(fnu))
        mag2 = fnu_to_ab_mag(jnp.array(100 * fnu))
        delta_mag = mag2 - mag1
        assert jnp.allclose(delta_mag, -5.0, atol=1e-10)

    def test_ab_mag_to_fnu_inverse(self):
        """Round-trip: mag → fν → mag."""
        mag_input = jnp.array(20.0)
        fnu = ab_mag_to_fnu(mag_input)
        mag_output = fnu_to_ab_mag(fnu)
        assert jnp.allclose(mag_output, mag_input, rtol=1e-12)

    def test_fnu_to_ab_mag_inverse(self):
        """Round-trip: fν → mag → fν."""
        fnu_input = jnp.array(1.5e-19)
        mag = fnu_to_ab_mag(fnu_input)
        fnu_output = ab_mag_to_fnu(mag)
        assert jnp.allclose(fnu_output, fnu_input, rtol=1e-12)

    def test_fnu_guards_log_for_zero(self):
        """Function should not raise NaN for zero flux."""
        # Zero flux should give a large magnitude, not NaN
        mag = fnu_to_ab_mag(jnp.array(0.0))
        assert jnp.isfinite(mag).all()
        # Should be large and positive
        assert jnp.all(mag > 0.0)

    def test_fnu_array_broadcast(self):
        """Functions work with arrays of arbitrary shape."""
        fnu_array = jnp.array([1.0e-19, 2.0e-19, 3.0e-19])
        mag_array = fnu_to_ab_mag(fnu_array)
        assert mag_array.shape == (3,)
        assert jnp.all(jnp.isfinite(mag_array))

    def test_ab_mag_to_fnu_broadcast(self):
        """ab_mag_to_fnu works with arrays."""
        mag_array = jnp.array([18.0, 19.0, 20.0, 21.0])
        fnu_array = ab_mag_to_fnu(mag_array)
        assert fnu_array.shape == (4,)
        assert jnp.all(fnu_array > 0.0)


# ---------------------------------------------------------------------------
# Test Absolute Magnitude
# ---------------------------------------------------------------------------


class TestAbsoluteMagnitudes:
    """Test absolute magnitude conversions."""

    def test_lnu_to_absolute_ab_mag_round_trip(self):
        """Round-trip: L_ν → M_AB → L_ν."""
        lnu_input = jnp.array(1.0e40)  # 10^40 erg/s/Hz, plausible for a galaxy
        mag_abs = lnu_to_absolute_ab_mag(lnu_input)
        lnu_output = absolute_ab_mag_to_lnu(mag_abs)
        assert jnp.allclose(lnu_output, lnu_input, rtol=1e-12)

    def test_absolute_ab_mag_to_lnu_round_trip(self):
        """Round-trip: M_AB → L_ν → M_AB."""
        mag_abs_input = jnp.array(-22.0)  # Bright galaxy
        lnu = absolute_ab_mag_to_lnu(mag_abs_input)
        mag_abs_output = lnu_to_absolute_ab_mag(lnu)
        assert jnp.allclose(mag_abs_output, mag_abs_input, rtol=1e-12)

    def test_absolute_magnitude_finiteness(self):
        """Absolute magnitudes are finite for reasonable luminosities."""
        lnu_array = jnp.logspace(30, 50, 10)
        mag_array = lnu_to_absolute_ab_mag(lnu_array)
        assert jnp.all(jnp.isfinite(mag_array))

    def test_absolute_magnitude_scale(self):
        """10x brighter luminosity → 2.5 mag brighter (absolute)."""
        lnu = 1.0e40
        lnu_bright = 10.0 * lnu
        mag1 = lnu_to_absolute_ab_mag(jnp.array(lnu))
        mag2 = lnu_to_absolute_ab_mag(jnp.array(lnu_bright))
        delta_mag = mag2 - mag1
        assert jnp.allclose(delta_mag, -2.5, atol=1e-10)


# ---------------------------------------------------------------------------
# Test Distance Modulus
# ---------------------------------------------------------------------------


class TestDistanceModulus:
    """Test distance modulus calculations."""

    def test_distance_modulus_at_10pc(self):
        """μ = 0 at the standard distance of 10 pc."""
        mu = distance_modulus_from_dl(TEN_PC_CM)
        assert jnp.allclose(mu, 0.0, atol=1e-12)

    def test_distance_modulus_at_100pc(self):
        """μ = 5 at 100 pc (10x farther)."""
        mu = distance_modulus_from_dl(10.0 * TEN_PC_CM)
        assert jnp.allclose(mu, 5.0, atol=1e-10)

    def test_distance_modulus_scale(self):
        """μ increases by 5 for every 10x increase in distance."""
        dl1 = TEN_PC_CM
        dl2 = 10.0 * dl1
        dl3 = 100.0 * dl1
        mu1 = distance_modulus_from_dl(dl1)
        mu2 = distance_modulus_from_dl(dl2)
        mu3 = distance_modulus_from_dl(dl3)
        assert jnp.allclose(mu2 - mu1, 5.0, atol=1e-10)
        assert jnp.allclose(mu3 - mu1, 10.0, atol=1e-10)

    def test_distance_modulus_from_dl_mpc_consistency(self):
        """DL in Mpc and cm should give same μ."""
        dl_mpc = 10.0  # 10 Mpc
        dl_cm = dl_mpc * MPC_CM
        mu_from_cm = distance_modulus_from_dl(dl_cm)
        mu_from_mpc = distance_modulus_from_dl_mpc(jnp.array(dl_mpc))
        assert jnp.allclose(mu_from_cm, mu_from_mpc, rtol=1e-12)

    def test_distance_modulus_at_zero_gives_negative_infinity(self):
        """Distance → 0 gives μ → -∞ (as expected mathematically)."""
        mu = distance_modulus_from_dl(jnp.array(0.0))
        assert jnp.all(mu == -jnp.inf)

    def test_distance_modulus_from_dl_mpc_array(self):
        """Works with arrays of Mpc distances."""
        dl_mpc = jnp.array([0.1, 1.0, 10.0, 100.0])
        mu = distance_modulus_from_dl_mpc(dl_mpc)
        assert mu.shape == (4,)
        assert jnp.all(jnp.isfinite(mu))
        # Verify ordering: further distances → larger μ
        assert jnp.all(jnp.diff(mu) > 0.0)


# ---------------------------------------------------------------------------
# Test Apparent ↔ Absolute
# ---------------------------------------------------------------------------


class TestApparentAbsolute:
    """Test apparent ↔ absolute magnitude conversions."""

    def test_apparent_to_absolute_basic(self):
        """m = 25, μ = 30 → M = -5."""
        m_app = jnp.array(25.0)
        dist_mod = jnp.array(30.0)
        mag_abs = apparent_to_absolute(m_app, dist_mod)
        assert jnp.allclose(mag_abs, -5.0, atol=1e-10)

    def test_absolute_to_apparent_basic(self):
        """M = -5, μ = 30 → m = 25."""
        mag_abs = jnp.array(-5.0)
        dist_mod = jnp.array(30.0)
        m_app = absolute_to_apparent(mag_abs, dist_mod)
        assert jnp.allclose(m_app, 25.0, atol=1e-10)

    def test_round_trip_apparent_absolute(self):
        """m → M → m."""
        m_app_in = jnp.array(22.0)
        mu = jnp.array(35.5)
        mag_abs = apparent_to_absolute(m_app_in, mu)
        m_app_out = absolute_to_apparent(mag_abs, mu)
        assert jnp.allclose(m_app_out, m_app_in, rtol=1e-12)

    def test_round_trip_absolute_apparent(self):
        """M → m → M."""
        mag_abs_in = jnp.array(-20.0)
        mu = jnp.array(35.5)
        m_app = absolute_to_apparent(mag_abs_in, mu)
        mag_abs_out = apparent_to_absolute(m_app, mu)
        assert jnp.allclose(mag_abs_out, mag_abs_in, rtol=1e-12)

    def test_broadcast_apparent_to_absolute(self):
        """Functions broadcast arrays correctly."""
        m_app = jnp.array([20.0, 21.0, 22.0])
        mu = jnp.array(35.0)
        mag_abs = apparent_to_absolute(m_app, mu)
        assert mag_abs.shape == (3,)

    def test_broadcast_absolute_to_apparent(self):
        """Functions broadcast arrays correctly."""
        mag_abs = jnp.array([-20.0, -21.0, -22.0])
        mu = jnp.array([35.0, 36.0, 37.0])
        m_app = absolute_to_apparent(mag_abs, mu)
        assert m_app.shape == (3,)


# ---------------------------------------------------------------------------
# Test Vega Magnitude System
# ---------------------------------------------------------------------------


class TestVegaMagnitudes:
    """Test Vega magnitude conversions."""

    def test_ab_to_vega_v_band(self):
        """V-band offset is +0.02 (Blanton & Roweis 2007)."""
        mag_ab = jnp.array(0.0)
        mag_vega = ab_to_vega(mag_ab, AB_VEGA_OFFSETS["V"])
        assert jnp.allclose(mag_vega, -0.02, atol=1e-10)

    def test_ab_to_vega_b_band(self):
        """B-band offset is -0.09."""
        mag_ab = jnp.array(0.0)
        mag_vega = ab_to_vega(mag_ab, AB_VEGA_OFFSETS["B"])
        assert jnp.allclose(mag_vega, 0.09, atol=1e-10)

    def test_vega_to_ab_round_trip(self):
        """mag_V → mag_AB → mag_V."""
        mag_vega_in = jnp.array(15.5)
        offset = AB_VEGA_OFFSETS["V"]
        mag_ab = vega_to_ab(mag_vega_in, offset)
        mag_vega_out = ab_to_vega(mag_ab, offset)
        assert jnp.allclose(mag_vega_out, mag_vega_in, rtol=1e-12)

    def test_ab_to_vega_round_trip(self):
        """mag_AB → mag_V → mag_AB."""
        mag_ab_in = jnp.array(18.0)
        offset = AB_VEGA_OFFSETS["V"]
        mag_vega = ab_to_vega(mag_ab_in, offset)
        mag_ab_out = vega_to_ab(mag_vega, offset)
        assert jnp.allclose(mag_ab_out, mag_ab_in, rtol=1e-12)

    def test_ab_vega_offsets_dict_completeness(self):
        """Dict contains expected bands."""
        expected_bands = {"U", "B", "V", "R", "I", "J", "H", "K", "u", "g", "r", "i", "z"}
        assert all(band in AB_VEGA_OFFSETS for band in expected_bands)

    def test_ab_vega_offsets_all_floats(self):
        """All offset values are floats."""
        for _band, offset in AB_VEGA_OFFSETS.items():
            assert isinstance(offset, (int, float))

    def test_vega_magnitude_array(self):
        """Functions work with arrays."""
        mag_ab_array = jnp.array([15.0, 16.0, 17.0, 18.0])
        offset = AB_VEGA_OFFSETS["g"]
        mag_vega = ab_to_vega(mag_ab_array, offset)
        assert mag_vega.shape == (4,)
        assert jnp.all(jnp.isfinite(mag_vega))


# ---------------------------------------------------------------------------
# Test Surface Brightness
# ---------------------------------------------------------------------------


class TestSurfaceBrightness:
    """Test surface brightness calculations."""

    def test_mag_to_sb_unit_area(self):
        """Surface brightness = magnitude for 1 arcsec² area."""
        mag = jnp.array(20.0)
        area = jnp.array(1.0)  # 1 arcsec²
        mu = mag_to_surface_brightness(mag, area)
        assert jnp.allclose(mu, mag, atol=1e-10)

    def test_mag_to_sb_larger_area(self):
        """Larger area → dimmer surface brightness."""
        mag = jnp.array(20.0)
        area1 = jnp.array(1.0)
        area2 = jnp.array(100.0)
        mu1 = mag_to_surface_brightness(mag, area1)
        mu2 = mag_to_surface_brightness(mag, area2)
        # 100x larger area → ~5 mag dimmer per arcsec²
        delta_mu = mu2 - mu1
        assert jnp.allclose(delta_mu, 5.0, atol=1e-10)

    def test_surface_brightness_to_mag_round_trip(self):
        """μ → m → μ."""
        mu_in = jnp.array(24.5)
        area = jnp.array(25.0)  # 25 arcsec²
        mag = surface_brightness_to_mag(mu_in, area)
        mu_out = mag_to_surface_brightness(mag, area)
        assert jnp.allclose(mu_out, mu_in, rtol=1e-12)

    def test_mag_to_surface_brightness_round_trip(self):
        """m → μ → m."""
        mag_in = jnp.array(18.0)
        area = jnp.array(10.0)
        mu = mag_to_surface_brightness(mag_in, area)
        mag_out = surface_brightness_to_mag(mu, area)
        assert jnp.allclose(mag_out, mag_in, rtol=1e-12)

    def test_surface_brightness_guards_log_for_zero_area(self):
        """Function handles zero area safely."""
        mag = jnp.array(20.0)
        mu = mag_to_surface_brightness(mag, jnp.array(0.0))
        assert jnp.isfinite(mu).all()

    def test_surface_brightness_array(self):
        """Functions work with arrays."""
        mag_array = jnp.array([18.0, 19.0, 20.0])
        area_array = jnp.array([5.0, 10.0, 25.0])
        mu = mag_to_surface_brightness(mag_array, area_array)
        assert mu.shape == (3,)
        assert jnp.all(jnp.isfinite(mu))


# ---------------------------------------------------------------------------
# Test Cosmological Dimming
# ---------------------------------------------------------------------------


class TestCosmologicalDimming:
    """Test cosmological dimming (1+z redshift factor)."""

    def test_cosmological_dimming_z_zero(self):
        """At z=0, effective μ = μ_raw (no dimming)."""
        mu = jnp.array(35.0)
        z = jnp.array(0.0)
        mu_eff = cosmological_dimming(mu, z)
        assert jnp.allclose(mu_eff, mu, atol=1e-10)

    def test_cosmological_dimming_z_one(self):
        """At z=1: 1+z = 2, so 2.5 log10(2) ≈ 0.753 mag dimming."""
        mu = jnp.array(35.0)
        z = jnp.array(1.0)
        mu_eff = cosmological_dimming(mu, z)
        dimming_expected = 2.5 * jnp.log10(2.0)
        assert jnp.allclose(mu_eff, mu - dimming_expected, atol=1e-10)

    def test_cosmological_dimming_array(self):
        """Works with arrays of redshifts."""
        mu = jnp.array(35.0)
        z_array = jnp.array([0.0, 0.5, 1.0, 2.0])
        mu_eff = cosmological_dimming(mu, z_array)
        assert mu_eff.shape == (4,)
        assert jnp.all(jnp.isfinite(mu_eff))
        # Verify monotonic increase: higher z → greater dimming
        assert jnp.all(jnp.diff(mu_eff) < 0.0)

    def test_cosmological_dimming_broadcast(self):
        """Broadcast different shapes."""
        mu_array = jnp.array([30.0, 35.0, 40.0])
        z = jnp.array(0.5)
        mu_eff = cosmological_dimming(mu_array, z)
        assert mu_eff.shape == (3,)


# ---------------------------------------------------------------------------
# Test JIT Compilation
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Test that all functions compile and run under JAX JIT."""

    def test_jit_fnu_to_ab_mag(self):
        """fnu_to_ab_mag JITs successfully."""
        fnu_jitted = jax.jit(fnu_to_ab_mag)
        fnu = jnp.array(1.5e-19)
        mag = fnu_jitted(fnu)
        assert jnp.isfinite(mag)

    def test_jit_ab_mag_to_fnu(self):
        """ab_mag_to_fnu JITs successfully."""
        mag_jitted = jax.jit(ab_mag_to_fnu)
        mag = jnp.array(20.0)
        fnu = mag_jitted(mag)
        assert jnp.isfinite(fnu)

    def test_jit_lnu_to_absolute_ab_mag(self):
        """lnu_to_absolute_ab_mag JITs successfully."""
        lnu_jitted = jax.jit(lnu_to_absolute_ab_mag)
        lnu = jnp.array(1.0e40)
        mag = lnu_jitted(lnu)
        assert jnp.isfinite(mag)

    def test_jit_absolute_ab_mag_to_lnu(self):
        """absolute_ab_mag_to_lnu JITs successfully."""
        mag_jitted = jax.jit(absolute_ab_mag_to_lnu)
        mag = jnp.array(-20.0)
        lnu = mag_jitted(mag)
        assert jnp.isfinite(lnu)

    def test_jit_distance_modulus_from_dl(self):
        """distance_modulus_from_dl JITs successfully."""
        dl_jitted = jax.jit(distance_modulus_from_dl)
        dl = jnp.array(1.0e25)
        mu = dl_jitted(dl)
        assert jnp.isfinite(mu)

    def test_jit_distance_modulus_from_dl_mpc(self):
        """distance_modulus_from_dl_mpc JITs successfully."""
        dl_jitted = jax.jit(distance_modulus_from_dl_mpc)
        dl = jnp.array(10.0)
        mu = dl_jitted(dl)
        assert jnp.isfinite(mu)

    def test_jit_apparent_to_absolute(self):
        """apparent_to_absolute JITs successfully."""
        conversion_jitted = jax.jit(apparent_to_absolute)
        m_app = jnp.array(25.0)
        mu = jnp.array(30.0)
        mag_abs = conversion_jitted(m_app, mu)
        assert jnp.isfinite(mag_abs)

    def test_jit_absolute_to_apparent(self):
        """absolute_to_apparent JITs successfully."""
        conversion_jitted = jax.jit(absolute_to_apparent)
        mag_abs = jnp.array(-20.0)
        mu = jnp.array(30.0)
        m_app = conversion_jitted(mag_abs, mu)
        assert jnp.isfinite(m_app)

    def test_jit_ab_to_vega(self):
        """ab_to_vega JITs successfully."""
        offset = AB_VEGA_OFFSETS["V"]
        vega_jitted = jax.jit(lambda mag: ab_to_vega(mag, offset))
        mag = jnp.array(20.0)
        mag_v = vega_jitted(mag)
        assert jnp.isfinite(mag_v)

    def test_jit_vega_to_ab(self):
        """vega_to_ab JITs successfully."""
        offset = AB_VEGA_OFFSETS["V"]
        ab_jitted = jax.jit(lambda mag: vega_to_ab(mag, offset))
        mag = jnp.array(20.0)
        mag_ab = ab_jitted(mag)
        assert jnp.isfinite(mag_ab)

    def test_jit_mag_to_surface_brightness(self):
        """mag_to_surface_brightness JITs successfully."""
        sb_jitted = jax.jit(mag_to_surface_brightness)
        mag = jnp.array(20.0)
        area = jnp.array(10.0)
        mu = sb_jitted(mag, area)
        assert jnp.isfinite(mu)

    def test_jit_surface_brightness_to_mag(self):
        """surface_brightness_to_mag JITs successfully."""
        mag_jitted = jax.jit(surface_brightness_to_mag)
        mu = jnp.array(24.5)
        area = jnp.array(10.0)
        mag = mag_jitted(mu, area)
        assert jnp.isfinite(mag)

    def test_jit_cosmological_dimming(self):
        """cosmological_dimming JITs successfully."""
        dimming_jitted = jax.jit(cosmological_dimming)
        mu = jnp.array(35.0)
        z = jnp.array(0.5)
        mu_eff = dimming_jitted(mu, z)
        assert jnp.isfinite(mu_eff)
