# SPDX-License-Identifier: BSD-3-Clause
"""Tests for magnitude system utilities (AB, Vega, absolute, apparent, surface brightness).

Validates:
- AB magnitude conversions (fν ↔ m_AB) against the AB zeropoint definition
- Absolute magnitude from luminosity and distance modulus limits (10 pc, 100 pc)
- Apparent ↔ absolute via distance modulus
- Vega magnitude offsets (Blanton & Roweis 2007 values)
- Surface brightness dilution
- Cosmological dimming (1+z factor)
- One parametrized vectorization-consistency test and one parametrized
  JIT-parity test cover every public function (replacing the old per-function
  shape-only and finite-only tests).
"""

import chex
import jax
import jax.numpy as jnp
import pytest

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

pytestmark = pytest.mark.bounds
from tengri.utils.physics_constants import MPC_CM, TEN_PC_CM

# ── Test AB Magnitude System ──────────────────────────────────────


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

    def test_fnu_zero_flux_hits_documented_clamp(self):
        """Zero flux hits the documented 1e-300 log guard, not NaN.

        The guard is `jnp.maximum(fnu, 1e-300)`, so the limit is exactly
        m_AB = -2.5 log10(1e-300 / 3.631e-20).
        """
        mag = fnu_to_ab_mag(jnp.array(0.0))
        expected = -2.5 * jnp.log10(1e-300 / 3.631e-20)
        assert jnp.allclose(mag, expected, rtol=1e-12)


# ── Test Absolute Magnitude ───────────────────────────────────────


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

    def test_absolute_magnitude_scale(self):
        """10x brighter luminosity → 2.5 mag brighter (absolute)."""
        lnu = 1.0e40
        lnu_bright = 10.0 * lnu
        mag1 = lnu_to_absolute_ab_mag(jnp.array(lnu))
        mag2 = lnu_to_absolute_ab_mag(jnp.array(lnu_bright))
        delta_mag = mag2 - mag1
        assert jnp.allclose(delta_mag, -2.5, atol=1e-10)


# ── Test Distance Modulus ─────────────────────────────────────────


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

    def test_distance_modulus_monotonic_in_distance(self):
        """Farther distances give strictly larger μ."""
        dl_mpc = jnp.array([0.1, 1.0, 10.0, 100.0])
        mu = distance_modulus_from_dl_mpc(dl_mpc)
        assert jnp.all(jnp.diff(mu) > 0.0)


# ── Test Apparent ↔ Absolute ──────────────────────────────────────


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


# ── Test Vega Magnitude System ────────────────────────────────────


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


# ── Test Surface Brightness ───────────────────────────────────────


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

    def test_zero_area_hits_documented_clamp(self):
        """Zero area hits the documented 1e-300 log guard: μ = m - 750 exactly."""
        mag = jnp.array(20.0)
        mu = mag_to_surface_brightness(mag, jnp.array(0.0))
        assert jnp.allclose(mu, mag + 2.5 * jnp.log10(1e-300), rtol=1e-12)


# ── Test Cosmological Dimming ─────────────────────────────────────


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

    def test_cosmological_dimming_monotonic_in_z(self):
        """Higher z → greater dimming, monotonically."""
        mu = jnp.array(35.0)
        z_array = jnp.array([0.0, 0.5, 1.0, 2.0])
        mu_eff = cosmological_dimming(mu, z_array)
        assert jnp.all(jnp.diff(mu_eff) < 0.0)


# ── Vectorization + JIT parity (all public functions) ────────────

# (test id, function, example args). Each function is exercised twice below:
# once for vectorized-call ≡ elementwise-scalar-calls, once for jit ≡ eager.
_FUNCTION_CASES = [
    ("fnu_to_ab_mag", fnu_to_ab_mag, (jnp.array([1.0e-19, 2.0e-19, 3.0e-19]),)),
    ("ab_mag_to_fnu", ab_mag_to_fnu, (jnp.array([18.0, 19.0, 20.0, 21.0]),)),
    ("lnu_to_absolute_ab_mag", lnu_to_absolute_ab_mag, (jnp.logspace(35.0, 45.0, 4),)),
    ("absolute_ab_mag_to_lnu", absolute_ab_mag_to_lnu, (jnp.array([-22.0, -20.0, -18.0]),)),
    ("distance_modulus_from_dl", distance_modulus_from_dl, (jnp.array([1e24, 1e25, 1e26]),)),
    (
        "distance_modulus_from_dl_mpc",
        distance_modulus_from_dl_mpc,
        (jnp.array([0.1, 1.0, 10.0, 100.0]),),
    ),
    (
        "apparent_to_absolute",
        apparent_to_absolute,
        (jnp.array([20.0, 21.0, 22.0]), jnp.array(35.0)),
    ),
    (
        "absolute_to_apparent",
        absolute_to_apparent,
        (jnp.array([-20.0, -21.0, -22.0]), jnp.array([35.0, 36.0, 37.0])),
    ),
    ("ab_to_vega", ab_to_vega, (jnp.array([15.0, 16.0, 17.0]), jnp.array(-0.08))),
    ("vega_to_ab", vega_to_ab, (jnp.array([15.5, 16.5]), jnp.array(0.02))),
    (
        "mag_to_surface_brightness",
        mag_to_surface_brightness,
        (jnp.array([18.0, 19.0, 20.0]), jnp.array([5.0, 10.0, 25.0])),
    ),
    (
        "surface_brightness_to_mag",
        surface_brightness_to_mag,
        (jnp.array([24.5, 25.5]), jnp.array(10.0)),
    ),
    ("cosmological_dimming", cosmological_dimming, (jnp.array(35.0), jnp.array([0.0, 0.5, 2.0]))),
]

_CASE_IDS = [c[0] for c in _FUNCTION_CASES]


@pytest.mark.parametrize("fn,args", [c[1:] for c in _FUNCTION_CASES], ids=_CASE_IDS)
def test_vectorized_call_matches_scalar_calls(fn, args):
    """Array inputs give exactly the elementwise scalar results.

    Subsumes the old per-function shape-only broadcast tests: shape is implied
    and every element is value-checked against the scalar code path.
    """
    broadcast = jnp.broadcast_arrays(*args)
    out_vec = jnp.ravel(fn(*args))
    out_scalar = jnp.stack(
        [fn(*(jnp.ravel(a)[i] for a in broadcast)) for i in range(out_vec.shape[0])]
    )
    chex.assert_trees_all_close(out_vec, out_scalar, rtol=1e-12)


@pytest.mark.parametrize("fn,args", [c[1:] for c in _FUNCTION_CASES], ids=_CASE_IDS)
def test_jit_matches_eager(fn, args):
    """jax.jit output matches the eager output (rtol=1e-6 JIT-parity convention).

    Subsumes the old per-function finite-only JIT tests.
    """
    chex.assert_trees_all_close(jax.jit(fn)(*args), fn(*args), rtol=1e-6)
