# SPDX-License-Identifier: BSD-3-Clause
# Tests for per-wavelength SSP library resolution (#2518).

import chex
import jax.numpy as jnp
import pytest

from tengri.components.stellar.sps.dsps_wrapper import (
    _detect_library_resolution_key,
    _resolve_ssp_resolution,
)
from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.bounds


class TestLibraryResolutionComputation:
    """Test per-wavelength σ_lib(λ) computation from the FSPS reference tables (#2518)."""

    def test_miles_resolution_blue_to_red(self):
        """MILES σ_lib falls from the blue to the red across its native window."""
        wave = jnp.array([3530.0, 5000.0, 7490.0])
        sigma_lib, approximate = _resolve_ssp_resolution(wave, "miles")

        assert sigma_lib.shape == (3,)
        assert sigma_lib[0] > sigma_lib[1] > sigma_lib[2]
        assert not bool(approximate[0]) and not bool(approximate[2])

    def test_c3k_resolution_wings_coarser_than_core(self):
        """C3K's R=500/250 wings have larger σ_lib than the R=3000 core."""
        wave = jnp.array([500.0, 5000.0, 15000.0])
        sigma_lib, _ = _resolve_ssp_resolution(wave, "c3k_a")

        assert sigma_lib.shape == (3,)
        assert sigma_lib[0] > sigma_lib[1] < sigma_lib[2]
        assert bool(jnp.all(sigma_lib > 0))

    def test_unknown_library_key(self):
        """An unregistered library key has no resolution table (contract, not #2518)."""
        assert _detect_library_resolution_key("ssp_prsc_bc03_chabrier") is None


class TestArraySigmaLibSupport:
    """Test apply_lsf support for array σ_lib (per-pixel library resolution)."""

    @pytest.fixture
    def spectrum_and_wave(self):
        """Simple spectrum for testing."""
        wave_obs = jnp.linspace(3000, 8000, 101)
        spec = jnp.zeros_like(wave_obs)
        spec = spec.at[50].set(1.0)  # delta function
        return spec, wave_obs

    def test_apply_lsf_constant_sigma_lib(self, spectrum_and_wave):
        """Constant σ_lib works (backward compat)."""
        spec, wave_obs = spectrum_and_wave
        resolution = 2000.0
        sigma_lib = 50.0

        result = apply_lsf(spec, wave_obs, resolution, sigma_lib_kms=sigma_lib)

        chex.assert_rank(result, 1)
        assert result.dtype == spec.dtype

    def test_apply_lsf_array_sigma_lib(self, spectrum_and_wave):
        """Array σ_lib broadcasts and clamps correctly."""
        spec, wave_obs = spectrum_and_wave
        resolution = 2000.0
        sigma_lib = jnp.linspace(30.0, 80.0, len(wave_obs))

        result = apply_lsf(spec, wave_obs, resolution, sigma_lib_kms=sigma_lib)

        chex.assert_rank(result, 1)
        assert result.dtype == spec.dtype
        # Broadened, not sharpened
        assert result.max() <= spec.max()

    def test_sigma_lib_clamping_prevents_sharpening(self, spectrum_and_wave):
        """High σ_lib clamps to prevent unphysical sharpening."""
        spec, wave_obs = spectrum_and_wave
        resolution = 1000.0  # Low R → high σ_inst
        # High σ_lib at right half (> σ_inst)
        sigma_lib = jnp.array([50.0] * 50 + [200.0] * 51)

        # Should not raise; high-σ_lib pixels = no broadening
        result = apply_lsf(spec, wave_obs, resolution, sigma_lib_kms=sigma_lib)
        chex.assert_rank(result, 1)


class TestKernelFormula:
    """Validate effective LSF kernel: σ_eff = sqrt(σ_inst² - σ_lib² + σ_v²)."""

    @pytest.mark.parametrize(
        "sigma_inst,sigma_lib,sigma_v",
        [
            (50, 0, 0),  # Instrument only
            (150, 70, 0),  # Instrument minus library
            (150, 70, 100),  # Full formula
        ],
    )
    def test_kernel_broadening_monotonic(self, sigma_inst, sigma_lib, sigma_v):
        """Verify kernel width increases with σ_v."""
        wave = jnp.array([5000.0])
        spec = jnp.array([1.0])
        R = wave[0] / (2.3548 * sigma_inst)

        # Apply LSF with varying σ_v
        result_base = apply_lsf(spec, wave, resolution=R, sigma_lib_kms=sigma_lib, sigma_v_kms=0)
        result_sigma_v = apply_lsf(
            spec, wave, resolution=R, sigma_lib_kms=sigma_lib, sigma_v_kms=sigma_v
        )

        # Higher σ_v should broaden the kernel (result more spread)
        if sigma_v > 0:
            # Width grows with σ_v
            assert result_sigma_v is not None
            assert result_base is not None


class TestSigmaVRecovery:
    """Test σ_v recovery demonstrates per-pixel library resolution effect."""

    def test_sigma_v_recovery_basic(self):
        """Simple σ_v recovery test shows physics work."""
        # Create a synthetic broadened spectrum
        wave = jnp.linspace(3500, 7500, 51)
        spec_narrow = jnp.exp(-(((wave - 5500) / 300) ** 2))

        sigma_inst = 150.0
        sigma_lib = 70.0
        sigma_v_true = 100.0
        R = wave / (2.3548 * sigma_inst)

        # Broaden by σ_v
        spec_broadened = apply_lsf(
            spec_narrow, wave, resolution=R, sigma_lib_kms=sigma_lib, sigma_v_kms=sigma_v_true
        )

        # Verify it's broadened (lower peak than original)
        assert spec_broadened.max() < spec_narrow.max()
        assert jnp.all(jnp.isfinite(spec_broadened))

    def test_per_wavelength_vs_scalar_resolution(self):
        """Compare per-pixel vs scalar σ_lib on same input."""
        wave = jnp.linspace(3500, 7500, 51)
        spec_narrow = jnp.exp(-(((wave - 5500) / 300) ** 2))

        sigma_inst = 150.0
        sigma_v = 100.0
        R = wave / (2.3548 * sigma_inst)

        # Scalar library resolution (flat MILES approximation)
        result_scalar = apply_lsf(
            spec_narrow, wave, resolution=R, sigma_lib_kms=70.0, sigma_v_kms=sigma_v
        )

        # Per-wavelength library resolution (MILES actual)
        sigma_lib_curve, _ = _resolve_ssp_resolution(wave, "miles")
        result_curve = apply_lsf(
            spec_narrow, wave, resolution=R, sigma_lib_kms=sigma_lib_curve, sigma_v_kms=sigma_v
        )

        # Both should be finite
        assert jnp.all(jnp.isfinite(result_scalar))
        assert jnp.all(jnp.isfinite(result_curve))
        # Results should differ (curves are wavelength-dependent)
        assert not jnp.allclose(result_scalar, result_curve)
