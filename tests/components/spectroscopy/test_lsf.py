"""Tests for wavelength-dependent Line Spread Function (LSF) convolution.

Validates that:
1. Constant-R LSF reduces spectral resolution (smooths features)
2. Variable-R LSF applies more smoothing where R is lower
3. Flux is conserved (integral preserved)
4. Library resolution subtraction works correctly
5. Instrument resolution profiles return expected values
6. All paths are differentiable and JIT-compatible
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.observation.spectrum import (
    _C_KM_S,
    _FWHM_TO_SIGMA,
    SSP_LIBRARY_RESOLUTIONS,
    _resolution_to_sigma_kms,
    apply_lsf,
    nirspec_g140m_resolution,
    nirspec_prism_resolution,
)

pytestmark = pytest.mark.bounds

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def wave():
    """Uniformly spaced wavelength grid (1000 pixels, 5000-25000 A)."""
    return jnp.linspace(5000.0, 25000.0, 1000)


@pytest.fixture
def delta_spectrum(wave):
    """Flat spectrum with a delta-function spike at pixel 500."""
    spec = jnp.ones(1000)
    return spec.at[500].set(10.0)


@pytest.fixture
def flat_spectrum():
    """Flat spectrum (should be unchanged by LSF)."""
    return jnp.ones(1000) * 1e-17


# ── Instrument resolution profiles ────────────────────────────────


class TestInstrumentProfiles:
    """Tests for instrument R(lambda) utility functions."""

    def test_nirspec_prism_range(self):
        """PRISM resolution ranges from ~30 to ~330."""
        wave_um = jnp.linspace(0.6, 5.3, 100)
        R = nirspec_prism_resolution(wave_um)
        assert float(R[0]) == pytest.approx(30.0, abs=1.0)
        assert float(R[-1]) == pytest.approx(288.5, abs=30.0)
        # Monotonically increasing
        assert jnp.all(jnp.diff(R) >= 0)

    def test_nirspec_prism_clipping(self):
        """PRISM resolution is clipped at boundaries."""
        wave_um = jnp.array([0.1, 0.3, 10.0])
        R = nirspec_prism_resolution(wave_um)
        assert float(R[0]) == 30.0  # clipped at lower bound
        assert float(R[-1]) == 330.0  # clipped at upper bound

    def test_nirspec_g140m_constant(self):
        """G140M returns constant R=1000."""
        wave_um = jnp.linspace(1.0, 1.8, 50)
        R = nirspec_g140m_resolution(wave_um)
        assert_allclose(R, 1000.0 * jnp.ones(50))

    def test_profiles_return_correct_shape(self):
        """Resolution profiles return same shape as input."""
        wave_um = jnp.linspace(0.6, 5.3, 200)
        assert nirspec_prism_resolution(wave_um).shape == (200,)
        assert nirspec_g140m_resolution(wave_um).shape == (200,)


# ── Library resolution constants ──────────────────────────────────


class TestLibraryResolutions:
    """Tests for SSP library resolution dictionary."""

    def test_known_libraries_present(self):
        """Expected library keys exist."""
        assert "miles" in SSP_LIBRARY_RESOLUTIONS
        assert "c3k" in SSP_LIBRARY_RESOLUTIONS
        assert "fsps_default" in SSP_LIBRARY_RESOLUTIONS

    def test_miles_value(self):
        """MILES library sigma is ~70 km/s."""
        assert SSP_LIBRARY_RESOLUTIONS["miles"] == pytest.approx(70.0)

    def test_c3k_value(self):
        """C3K library sigma is ~15 km/s."""
        assert SSP_LIBRARY_RESOLUTIONS["c3k"] == pytest.approx(15.0)


# ── Resolution to sigma conversion ────────────────────────────────


class TestResolutionConversion:
    """Tests for R -> sigma_kms conversion."""

    def test_known_value(self):
        """R=1000 should give sigma ~ 127 km/s."""
        sigma = _resolution_to_sigma_kms(jnp.array(1000.0))
        expected = _C_KM_S / (_FWHM_TO_SIGMA * 1000.0)
        assert float(sigma) == pytest.approx(float(expected), rel=1e-10)

    def test_higher_r_smaller_sigma(self):
        """Higher R means smaller velocity width."""
        sigma_100 = _resolution_to_sigma_kms(jnp.array(100.0))
        sigma_1000 = _resolution_to_sigma_kms(jnp.array(1000.0))
        assert float(sigma_100) > float(sigma_1000)


# ── Constant-R LSF ────────────────────────────────────────────────


class TestApplyLSFConstantR:
    """Tests for apply_lsf with scalar (constant) resolution."""

    def test_output_shape(self, wave, delta_spectrum):
        """Output has same shape as input."""
        result = apply_lsf(delta_spectrum, wave, resolution=100.0)
        assert result.shape == delta_spectrum.shape

    def test_smooths_delta(self, wave, delta_spectrum):
        """LSF reduces the peak of a delta function."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0)
        assert float(jnp.max(smoothed)) < float(jnp.max(delta_spectrum))

    def test_lower_r_more_smoothing(self, wave, delta_spectrum):
        """Lower R (broader LSF) gives more smoothing."""
        peak_r50 = float(jnp.max(apply_lsf(delta_spectrum, wave, resolution=50.0)))
        peak_r100 = float(jnp.max(apply_lsf(delta_spectrum, wave, resolution=100.0)))
        peak_r500 = float(jnp.max(apply_lsf(delta_spectrum, wave, resolution=500.0)))
        assert peak_r50 < peak_r100 < peak_r500

    def test_conserves_flux(self, wave, delta_spectrum):
        """Total flux is conserved under convolution."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0)
        assert_allclose(
            float(jnp.sum(smoothed)),
            float(jnp.sum(delta_spectrum)),
            rtol=1e-6,
        )

    def test_flat_spectrum_unchanged(self, wave, flat_spectrum):
        """Flat spectrum is unchanged by LSF (interior pixels)."""
        smoothed = apply_lsf(flat_spectrum, wave, resolution=100.0)
        interior = slice(50, -50)
        assert_allclose(
            smoothed[interior],
            flat_spectrum[interior],
            rtol=1e-5,
        )

    def test_high_r_near_identity(self, wave, delta_spectrum):
        """Very high R (narrow LSF) barely changes the spectrum."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=50000.0)
        assert_allclose(smoothed, delta_spectrum, atol=0.01)

    def test_finite_output(self, wave, delta_spectrum):
        """No NaN or Inf in output."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0)
        assert jnp.all(jnp.isfinite(smoothed))


# ── Library resolution subtraction ────────────────────────────────


class TestLibrarySubtraction:
    """Tests for sigma_lib quadrature subtraction."""

    def test_no_lib_vs_with_lib(self, wave, delta_spectrum):
        """Subtracting library resolution reduces the effective smoothing."""
        # With sigma_lib=0, full instrument smoothing
        smoothed_full = apply_lsf(delta_spectrum, wave, resolution=100.0, sigma_lib_kms=0.0)
        # With sigma_lib > 0, less additional smoothing needed
        smoothed_sub = apply_lsf(delta_spectrum, wave, resolution=100.0, sigma_lib_kms=50.0)
        # The subtracted version should have a higher peak (less smoothed)
        assert float(jnp.max(smoothed_sub)) > float(jnp.max(smoothed_full))

    def test_lib_exceeds_instrument_no_smoothing(self, wave, delta_spectrum):
        """When sigma_lib > sigma_inst, no smoothing is applied."""
        # R=100 gives sigma_inst ~ 1275 km/s; sigma_lib=2000 exceeds it
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0, sigma_lib_kms=2000.0)
        # Should be essentially unchanged (sigma_eff = 0)
        assert_allclose(smoothed, delta_spectrum, atol=1e-10)


# ── Variable-R LSF ────────────────────────────────────────────────


class TestApplyLSFVariableR:
    """Tests for apply_lsf with per-pixel resolution array."""

    def test_output_shape(self, wave, delta_spectrum):
        """Output has same shape as input."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        result = apply_lsf(delta_spectrum, wave, resolution=R_var)
        assert result.shape == delta_spectrum.shape

    def test_smooths_delta(self, wave, delta_spectrum):
        """Variable-R LSF reduces the peak of a delta function."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        smoothed = apply_lsf(delta_spectrum, wave, resolution=R_var)
        assert float(jnp.max(smoothed)) < float(jnp.max(delta_spectrum))

    def test_conserves_flux(self, wave, delta_spectrum):
        """Total flux is conserved for variable R."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        smoothed = apply_lsf(delta_spectrum, wave, resolution=R_var)
        assert_allclose(
            float(jnp.sum(smoothed)),
            float(jnp.sum(delta_spectrum)),
            rtol=0.02,  # Slightly looser for piecewise approximation
        )

    def test_constant_array_matches_scalar(self, wave, delta_spectrum):
        """Uniform R array should match scalar R result closely."""
        R_scalar = 200.0
        R_array = R_scalar * jnp.ones_like(wave)
        smoothed_scalar = apply_lsf(delta_spectrum, wave, resolution=R_scalar)
        smoothed_array = apply_lsf(delta_spectrum, wave, resolution=R_array)
        # Piecewise approximation should be close but not identical
        assert_allclose(smoothed_array, smoothed_scalar, rtol=0.05, atol=0.05)

    def test_prism_like_variable_r(self, wave, delta_spectrum):
        """PRISM-like variable R produces finite output."""
        R_var = nirspec_prism_resolution(wave / 1e4)
        smoothed = apply_lsf(delta_spectrum, wave, resolution=R_var)
        assert jnp.all(jnp.isfinite(smoothed))

    def test_more_bins_better_accuracy(self, wave, delta_spectrum):
        """More bins should give result closer to constant-R case for uniform R."""
        R_const = 150.0 * jnp.ones_like(wave)
        ref = apply_lsf(delta_spectrum, wave, resolution=150.0)
        err_8 = jnp.max(jnp.abs(apply_lsf(delta_spectrum, wave, R_const, n_bins=8) - ref))
        err_32 = jnp.max(jnp.abs(apply_lsf(delta_spectrum, wave, R_const, n_bins=32) - ref))
        assert float(err_32) <= float(err_8) + 1e-10


# ── Differentiability and JIT ─────────────────────────────────────


class TestLSFGradients:
    """Gradients through LSF convolution are finite."""

    def test_gradient_wrt_spectrum_constant_r(self, wave, delta_spectrum):
        """Gradient w.r.t. input flux is finite (constant R)."""

        def loss(spec):
            return jnp.sum(apply_lsf(spec, wave, resolution=100.0) ** 2)

        g = jax.grad(loss)(delta_spectrum)
        assert jnp.all(jnp.isfinite(g))

    def test_gradient_wrt_spectrum_variable_r(self, wave, delta_spectrum):
        """Gradient w.r.t. input flux is finite (variable R)."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)

        def loss(spec):
            return jnp.sum(apply_lsf(spec, wave, resolution=R_var) ** 2)

        g = jax.grad(loss)(delta_spectrum)
        assert jnp.all(jnp.isfinite(g))

    def test_gradient_wrt_resolution_scalar(self, wave, delta_spectrum):
        """Gradient w.r.t. scalar resolution is finite."""

        def loss(R):
            return jnp.sum(apply_lsf(delta_spectrum, wave, resolution=R) ** 2)

        grad_jax = float(jax.grad(loss)(100.0))
        grad_fd = fd_grad(loss, 100.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_jit_constant_r(self, wave, delta_spectrum):
        """JIT compilation works for constant R."""
        fn = jax.jit(lambda s: apply_lsf(s, wave, resolution=100.0))
        result = fn(delta_spectrum)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_variable_r(self, wave, delta_spectrum):
        """JIT compilation works for variable R."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        fn = jax.jit(lambda s: apply_lsf(s, wave, resolution=R_var))
        result = fn(delta_spectrum)
        assert jnp.all(jnp.isfinite(result))
