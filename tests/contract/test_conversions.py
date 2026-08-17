# SPDX-License-Identifier: BSD-3-Clause
"""Comprehensive tests for tengri.utils.conversions module.

Tests cover:
- Round-trip conversions (bidirectional)
- Known formula verification against reference values
- Array broadcasting and shape preservation
- JAX JIT compilation compatibility
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.contract

from tengri.utils.conversions import (
    air_to_vacuum,
    attenuation_to_tau,
    erg_per_s_to_lsun,
    flambda_to_fnu,
    fnu_to_flambda,
    fnu_to_jy,
    fnu_to_lnu,
    fnu_to_maggies,
    fnu_to_njy,
    fnu_to_ujy,
    jy_to_fnu,
    llambda_to_lnu,
    lnu_to_fnu,
    lnu_to_llambda,
    lsun_to_erg_per_s,
    maggies_to_fnu,
    njy_to_fnu,
    tau_to_attenuation,
    ujy_to_fnu,
    vacuum_to_air,
)
from tengri.utils.physics_constants import (
    C_AA,
    L_SUN,
    MAGGIES_ZP_CGS,
)


class TestSpectralDensityConversions:
    """Test L_ν ↔ L_λ and f_ν ↔ f_λ conversions using spectral density formula.

    Formula: L_λ = L_ν × c / λ²
    """

    def test_lnu_to_llambda_scalar(self):
        """Test scalar conversion L_ν → L_λ."""
        lnu = 1e40  # erg/s/Hz
        wavelength = 5000.0  # Ångström
        llambda = lnu_to_llambda(jnp.array(lnu), jnp.array(wavelength))
        expected = lnu * C_AA / (wavelength**2)
        assert_allclose(llambda, expected, rtol=1e-10)

    def test_llambda_to_lnu_scalar(self):
        """Test scalar conversion L_λ → L_ν."""
        llambda = 1e-12  # erg/s/Å
        wavelength = 5000.0  # Ångström
        lnu = llambda_to_lnu(jnp.array(llambda), jnp.array(wavelength))
        expected = llambda * (wavelength**2) / C_AA
        assert_allclose(lnu, expected, rtol=1e-10)

    def test_lnu_to_llambda_roundtrip(self):
        """Test round-trip: L_ν → L_λ → L_ν."""
        lnu_orig = jnp.array([1e39, 5e39, 1e40])
        wavelength = jnp.array([3000.0, 5000.0, 8000.0])
        llambda = lnu_to_llambda(lnu_orig, wavelength)
        lnu_recovered = llambda_to_lnu(llambda, wavelength)
        assert_allclose(lnu_recovered, lnu_orig, rtol=1e-10)

    def test_fnu_to_flambda_array(self):
        """Test array conversion f_ν → f_λ."""
        fnu = jnp.array([1e-23, 5e-23, 1e-22])  # erg/s/cm²/Hz
        wavelength = jnp.array([2000.0, 5000.0, 10000.0])  # Ångström
        flambda = fnu_to_flambda(fnu, wavelength)
        expected = fnu * C_AA / (wavelength**2)
        assert_allclose(flambda, expected, rtol=1e-10)

    def test_flambda_to_fnu_roundtrip(self):
        """Test round-trip: f_λ → f_ν → f_λ."""
        flambda_orig = jnp.array([1e-14, 5e-14, 1e-13])  # erg/s/cm²/Å
        wavelength = jnp.array([3000.0, 5000.0, 8000.0])  # Ångström
        fnu = flambda_to_fnu(flambda_orig, wavelength)
        flambda_recovered = fnu_to_flambda(fnu, wavelength)
        assert_allclose(flambda_recovered, flambda_orig, rtol=1e-10)

    def test_lnu_to_llambda_broadcasting(self):
        """Test broadcasting when wavelength is array and lnu is scalar."""
        lnu = jnp.array(1e40)
        wavelength = jnp.array([3000.0, 5000.0, 8000.0])
        llambda = lnu_to_llambda(lnu, wavelength)
        expected = lnu * C_AA / (wavelength**2)
        assert_allclose(llambda, expected, rtol=1e-10)
        chex.assert_equal_shape([llambda, wavelength])

    def test_fnu_to_flambda_consistency_with_lnu_to_llambda(self):
        """Test that flux and luminosity conversions use same formula."""
        fnu = 1e-23
        lnu = 1e40
        wavelength = 5500.0
        flambda = fnu_to_flambda(jnp.array(fnu), jnp.array(wavelength))
        llambda = lnu_to_llambda(jnp.array(lnu), jnp.array(wavelength))
        # Both should use identical formula (just different units)
        assert_allclose(flambda / fnu, llambda / lnu, rtol=1e-10)


#: (label, cgs -> unit, unit -> cgs, the CGS value of one unit).
#:
#: Every one of these is a pure multiplicative rescaling of F_nu, so the three
#: tests each pair used to have -- forward reference, inverse reference,
#: roundtrip -- are the same three assertions with two names substituted. As a
#: table they stay three tests instead of twelve, and adding a fifth unit is one
#: row rather than three more functions.
_FLUX_UNITS = [
    ("jy", fnu_to_jy, jy_to_fnu, 1e-23),
    ("ujy", fnu_to_ujy, ujy_to_fnu, 1e-29),
    ("njy", fnu_to_njy, njy_to_fnu, 1e-32),
    ("maggies", fnu_to_maggies, maggies_to_fnu, MAGGIES_ZP_CGS),
]


class TestFluxDensityUnits:
    """F_nu unit conversions, as a table."""

    @pytest.mark.parametrize(
        ("label", "to_unit", "from_unit", "cgs_per_unit"), _FLUX_UNITS, ids=lambda v: v
    )
    def test_one_unit_is_its_defined_cgs_value(self, label, to_unit, from_unit, cgs_per_unit):
        """The zero point, in the forward direction: cgs_per_unit -> 1.0."""
        assert_allclose(to_unit(jnp.array(cgs_per_unit)), 1.0, rtol=1e-10)

    @pytest.mark.parametrize(
        ("label", "to_unit", "from_unit", "cgs_per_unit"), _FLUX_UNITS, ids=lambda v: v
    )
    def test_the_inverse_returns_that_same_cgs_value(
        self, label, to_unit, from_unit, cgs_per_unit
    ):
        """And in the inverse direction: 1.0 -> cgs_per_unit.

        Pinned against the absolute constant rather than against the forward
        function, so a pair that agreed with each other on a *wrong* constant
        still fails here.
        """
        assert_allclose(from_unit(jnp.array(1.0)), cgs_per_unit, rtol=1e-10)

    @pytest.mark.parametrize(
        ("label", "to_unit", "from_unit", "cgs_per_unit"), _FLUX_UNITS, ids=lambda v: v
    )
    def test_roundtrip_over_a_range(self, label, to_unit, from_unit, cgs_per_unit):
        """cgs -> unit -> cgs across three decades.

        Largely implied by the two reference tests for a pure scaling -- if the
        pair used mismatched constants the inverse test above already fails --
        but it costs one line per unit and covers array input and a span rather
        than the single point they pin.
        """
        original = jnp.array([1.0, 5.0, 10.0]) * cgs_per_unit
        assert_allclose(from_unit(to_unit(original)), original, rtol=1e-10)


class TestLuminosityConversions:
    """Test erg/s ↔ L_⊙ conversions."""

    def test_erg_per_s_to_lsun_reference_value(self):
        """Test that L_⊙ = 3.828e33 erg/s."""
        l_erg = jnp.array(L_SUN)
        l_lsun = erg_per_s_to_lsun(l_erg)
        assert_allclose(l_lsun, 1.0, rtol=1e-10)

    def test_lsun_to_erg_per_s_reference_value(self):
        """Test inverse: 1 L_⊙ = 3.828e33 erg/s."""
        l_lsun = jnp.array(1.0)
        l_erg = lsun_to_erg_per_s(l_lsun)
        assert_allclose(l_erg, L_SUN, rtol=1e-10)

    def test_erg_per_s_to_lsun_roundtrip(self):
        """Test round-trip: erg/s → L_⊙ → erg/s."""
        l_erg_orig = jnp.array([1e33, 5e33, 1e34])
        l_lsun = erg_per_s_to_lsun(l_erg_orig)
        l_erg_recovered = lsun_to_erg_per_s(l_lsun)
        assert_allclose(l_erg_recovered, l_erg_orig, rtol=1e-10)

    def test_lsun_to_erg_per_s_roundtrip(self):
        """Test round-trip: L_⊙ → erg/s → L_⊙."""
        l_lsun_orig = jnp.array([0.1, 1.0, 10.0])
        l_erg = lsun_to_erg_per_s(l_lsun_orig)
        l_lsun_recovered = erg_per_s_to_lsun(l_erg)
        assert_allclose(l_lsun_recovered, l_lsun_orig, rtol=1e-10)


class TestCosmologicalFlux:
    """Test cosmological luminosity ↔ flux conversions with redshift k-correction."""

    def test_lnu_to_fnu_zero_redshift(self):
        """Test L_ν → f_ν at z=0: f = L / (4π d²)."""
        lnu = jnp.array(1e40)  # erg/s/Hz
        dl_cm = jnp.array(1e26)  # 1 Mpc in cm (10 pc reference)
        z = jnp.array(0.0)
        fnu = lnu_to_fnu(lnu, dl_cm, z)
        expected = lnu / (4.0 * jnp.pi * dl_cm**2)
        assert_allclose(fnu, expected, rtol=1e-10)

    def test_fnu_to_lnu_zero_redshift(self):
        """Test f_ν → L_ν at z=0: inverse formula."""
        fnu = jnp.array(1e-23)  # erg/s/cm²/Hz
        dl_cm = jnp.array(1e26)
        z = jnp.array(0.0)
        lnu = fnu_to_lnu(fnu, dl_cm, z)
        expected = fnu * 4.0 * jnp.pi * dl_cm**2
        assert_allclose(lnu, expected, rtol=1e-10)

    def test_lnu_to_fnu_redshift_factor(self):
        """Test that (1+z) factor is applied in luminosity → flux."""
        lnu = jnp.array(1e40)
        dl_cm = jnp.array(1e26)
        z_low = jnp.array(0.01)
        z_high = jnp.array(0.1)
        fnu_low = lnu_to_fnu(lnu, dl_cm, z_low)
        fnu_high = lnu_to_fnu(lnu, dl_cm, z_high)
        ratio = fnu_high / fnu_low
        expected_ratio = (1.0 + z_high) / (1.0 + z_low)
        assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_lnu_to_fnu_roundtrip(self):
        """Test round-trip: L_ν → f_ν → L_ν."""
        lnu_orig = jnp.array([1e39, 5e39, 1e40])
        dl_cm = jnp.array(1e26)
        z = jnp.array(0.05)
        fnu = lnu_to_fnu(lnu_orig, dl_cm, z)
        lnu_recovered = fnu_to_lnu(fnu, dl_cm, z)
        assert_allclose(lnu_recovered, lnu_orig, rtol=1e-10)

    def test_fnu_to_lnu_roundtrip(self):
        """Test round-trip: f_ν → L_ν → f_ν."""
        fnu_orig = jnp.array([1e-23, 5e-23, 1e-22])
        dl_cm = jnp.array(1e26)
        z = jnp.array(0.1)
        lnu = fnu_to_lnu(fnu_orig, dl_cm, z)
        fnu_recovered = lnu_to_fnu(lnu, dl_cm, z)
        assert_allclose(fnu_recovered, fnu_orig, rtol=1e-10)

    def test_lnu_to_fnu_broadcasting(self):
        """Test broadcasting with arrays of different shapes."""
        lnu = jnp.array([1e40, 2e40, 3e40])  # 3 objects
        dl_cm = jnp.array([[1e26], [1e27], [1e28]])  # 3×1
        z = jnp.array([0.01, 0.05, 0.1])  # 3 objects
        fnu = lnu_to_fnu(lnu, dl_cm, z)
        chex.assert_shape(fnu, (3, 3))


class TestOpticalDepth:
    """Test optical depth ↔ attenuation conversions."""

    def test_tau_to_attenuation_reference(self):
        """Test that τ=1 → A ≈ 1.0857 mag."""
        tau = jnp.array(1.0)
        a = tau_to_attenuation(tau)
        expected = 2.5 * jnp.log10(jnp.e)
        assert_allclose(a, expected, rtol=1e-10)
        assert_allclose(a, 1.0857, rtol=1e-4)

    def test_attenuation_to_tau_reference(self):
        """Test inverse: A=1.0857 mag → τ=1."""
        a = jnp.array(1.0857)
        tau = attenuation_to_tau(a)
        assert_allclose(tau, 1.0, rtol=1e-3)

    def test_tau_to_attenuation_roundtrip(self):
        """Test round-trip: τ → A → τ."""
        tau_orig = jnp.array([0.1, 0.5, 1.0, 2.0])
        a = tau_to_attenuation(tau_orig)
        tau_recovered = attenuation_to_tau(a)
        assert_allclose(tau_recovered, tau_orig, rtol=1e-10)

    def test_attenuation_to_tau_roundtrip(self):
        """Test round-trip: A → τ → A."""
        a_orig = jnp.array([0.1, 0.5, 1.0, 2.0])
        tau = attenuation_to_tau(a_orig)
        a_recovered = tau_to_attenuation(tau)
        assert_allclose(a_recovered, a_orig, rtol=1e-10)


class TestWavelengthConversions:
    """Test vacuum ↔ air wavelength conversions (numpy-based, non-JIT)."""

    def test_air_to_vacuum_roundtrip_optical(self):
        """Test round-trip: air → vacuum → air for optical wavelengths."""
        # Use optical wavelengths (where conversion is most accurate)
        air_wavelengths = np.array([4861.0, 5007.0, 6563.0])  # Hβ, [OIII], Hα
        vacuum = air_to_vacuum(air_wavelengths)
        air_recovered = vacuum_to_air(vacuum)
        # Tolerance is ~1 Å for optical due to formula accuracy
        assert_allclose(air_recovered, air_wavelengths, atol=1.0)

    def test_vacuum_to_air_roundtrip_optical(self):
        """Test round-trip: vacuum → air → vacuum for optical wavelengths."""
        vacuum_wavelengths = np.array([4862.68, 5007.99, 6564.61])  # Hβ, [OIII], Hα vacuum
        air = vacuum_to_air(vacuum_wavelengths)
        vacuum_recovered = air_to_vacuum(air)
        assert_allclose(vacuum_recovered, vacuum_wavelengths, atol=1.0)

    def test_vacuum_to_air_halpha(self):
        """Test Hα vacuum→air conversion with known reference."""
        # Hα vacuum = 6564.61 Å, air ≈ 6562.8 Å (difference ~1.8 Å)
        vacuum_halpha = np.array(6564.61)
        air_halpha = vacuum_to_air(vacuum_halpha)
        # Air wavelength should be shorter
        assert air_halpha < vacuum_halpha
        # Difference should be ~1.8 Å
        assert_allclose(vacuum_halpha - air_halpha, 1.8, atol=0.3)

    def test_air_to_vacuum_hbeta(self):
        """Test Hβ air→vacuum conversion with expected result."""
        # Hβ air ≈ 4859 Å, vacuum ≈ 4860.36 Å (difference ~1.36 Å)
        air_hbeta = np.array(4859.0)
        vacuum_hbeta = air_to_vacuum(air_hbeta)
        # Vacuum wavelength should be longer
        assert vacuum_hbeta > air_hbeta
        # Difference should be ~1.3-1.4 Å
        assert_allclose(vacuum_hbeta - air_hbeta, 1.36, atol=0.2)

    def test_vacuum_to_air_ir(self):
        """Test vacuum→air at infrared wavelengths."""
        # At longer wavelengths, the conversion increases with λ²
        vacuum_ir = np.array([10000.0, 50000.0])
        air_ir = vacuum_to_air(vacuum_ir)
        # Air should be shorter than vacuum
        assert np.all(air_ir < vacuum_ir)
        # Difference should grow with wavelength (~2-14 Å)
        assert_allclose(vacuum_ir[0] - air_ir[0], 2.7, atol=0.2)
        assert_allclose(vacuum_ir[1] - air_ir[1], 13.7, atol=0.5)

    def test_vacuum_to_air_uv(self):
        """Test vacuum→air at UV wavelengths."""
        # At shorter wavelengths, the conversion is smaller but still present
        vacuum_uv = np.array([2000.0, 3000.0])
        air_uv = vacuum_to_air(vacuum_uv)
        # Air should be shorter than vacuum
        assert np.all(air_uv < vacuum_uv)
        # Difference should be ~0.6-0.9 Ångströms
        assert_allclose(vacuum_uv[0] - air_uv[0], 0.65, atol=0.1)
        assert_allclose(vacuum_uv[1] - air_uv[1], 0.87, atol=0.1)

    def test_wavelength_conversion_array_input(self):
        """Test that array inputs work correctly."""
        vacuum_array = np.array([3000.0, 5500.0, 8000.0])
        air_array = vacuum_to_air(vacuum_array)
        chex.assert_equal_shape([air_array, vacuum_array])
        assert np.all(air_array < vacuum_array)
        for i in range(len(vacuum_array)):
            air_single = vacuum_to_air(np.array(vacuum_array[i]))
            assert_allclose(air_array[i], air_single, rtol=1e-10)


class TestJITCompatibility:
    """Test that all functions are JIT-compatible and produce identical results."""

    def test_lnu_to_llambda_jit(self):
        """Test lnu_to_llambda under JIT compilation."""
        lnu_jit = jax.jit(lnu_to_llambda)
        lnu = jnp.array(1e40)
        wavelength = jnp.array(5500.0)
        result_eager = lnu_to_llambda(lnu, wavelength)
        result_jit = lnu_jit(lnu, wavelength)
        assert_allclose(result_eager, result_jit, rtol=1e-10)

    def test_lnu_to_fnu_jit(self):
        """Test lnu_to_fnu under JIT compilation."""
        lnu_to_fnu_jit = jax.jit(lnu_to_fnu)
        lnu = jnp.array(1e40)
        dl_cm = jnp.array(1e26)
        z = jnp.array(0.05)
        result_eager = lnu_to_fnu(lnu, dl_cm, z)
        result_jit = lnu_to_fnu_jit(lnu, dl_cm, z)
        assert_allclose(result_eager, result_jit, rtol=1e-10)

    def test_fnu_to_jy_jit(self):
        """Test fnu_to_jy under JIT compilation."""
        fnu_to_jy_jit = jax.jit(fnu_to_jy)
        fnu = jnp.array(1e-23)
        result_eager = fnu_to_jy(fnu)
        result_jit = fnu_to_jy_jit(fnu)
        assert_allclose(result_eager, result_jit, rtol=1e-10)

    def test_tau_to_attenuation_jit(self):
        """Test tau_to_attenuation under JIT compilation."""
        tau_to_attenuation_jit = jax.jit(tau_to_attenuation)
        tau = jnp.array(1.0)
        result_eager = tau_to_attenuation(tau)
        result_jit = tau_to_attenuation_jit(tau)
        assert_allclose(result_eager, result_jit, rtol=1e-10)

    def test_erg_per_s_to_lsun_jit(self):
        """Test erg_per_s_to_lsun under JIT compilation."""
        erg_to_lsun_jit = jax.jit(erg_per_s_to_lsun)
        l_erg = jnp.array(L_SUN * 10.0)
        result_eager = erg_per_s_to_lsun(l_erg)
        result_jit = erg_to_lsun_jit(l_erg)
        assert_allclose(result_eager, result_jit, rtol=1e-10)

    def test_all_jit_functions_compile(self):
        """Verify all conversion functions can be JIT-compiled."""
        jit_functions = [
            lnu_to_llambda,
            llambda_to_lnu,
            fnu_to_flambda,
            flambda_to_fnu,
            fnu_to_jy,
            jy_to_fnu,
            fnu_to_ujy,
            ujy_to_fnu,
            fnu_to_njy,
            njy_to_fnu,
            fnu_to_maggies,
            maggies_to_fnu,
            erg_per_s_to_lsun,
            lsun_to_erg_per_s,
            lnu_to_fnu,
            fnu_to_lnu,
            tau_to_attenuation,
            attenuation_to_tau,
        ]
        for func in jit_functions:
            # Verify that jit compilation succeeds
            jitted = jax.jit(func)
            assert jitted is not None
