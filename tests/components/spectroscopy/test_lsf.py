# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for wavelength-dependent Line Spread Function (LSF) convolution.

Strong anchors: flux conservation (∫F dλ before == after), delta-function
broadening (peak reduction shows convolution), limits (high-R near identity,
flat spectrum unchanged), library resolution quadrature subtraction, and
gradient finitude. All paths are JAX-differentiable.
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
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds


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
    """Frozen: PRISM and G140M resolution values and monotonicity."""

    def test_nirspec_prism_monotonic_increasing(self):
        """PRISM resolution increases monotonically with wavelength."""
        wave_um = jnp.linspace(0.6, 5.3, 100)
        R = nirspec_prism_resolution(wave_um)
        # Should increase monotonically
        assert_non_negative(jnp.diff(R), name="output")

    def test_nirspec_prism_boundary_clipping(self):
        """PRISM resolution is clipped at boundaries (30 and 330)."""
        wave_um = jnp.array([0.1, 0.3, 10.0])
        R = nirspec_prism_resolution(wave_um)
        assert float(R[0]) == 30.0  # clipped at lower bound
        assert float(R[-1]) == 330.0  # clipped at upper bound

    def test_nirspec_g140m_constant_r(self):
        """G140M returns constant R=1000 across wavelength range."""
        wave_um = jnp.linspace(1.0, 1.8, 50)
        R = nirspec_g140m_resolution(wave_um)
        assert_allclose(R, 1000.0 * jnp.ones(50))


# ── Library resolution constants ──────────────────────────────────


class TestLibraryResolutions:
    """Frozen: exact SSP library sigma values (km/s)."""

    def test_miles_resolution(self):
        """MILES library sigma is frozen at 70 km/s."""
        assert SSP_LIBRARY_RESOLUTIONS["miles"] == pytest.approx(70.0)

    def test_c3k_resolution(self):
        """C3K library sigma is frozen at 15 km/s."""
        assert SSP_LIBRARY_RESOLUTIONS["c3k"] == pytest.approx(15.0)


# ── Resolution to sigma conversion ────────────────────────────────


class TestResolutionConversion:
    """Frozen: R -> sigma_kms formula."""

    def test_resolution_to_sigma_formula(self):
        """R=1000 yields sigma = c_kms / (FWHM_TO_SIGMA * R)."""
        sigma = _resolution_to_sigma_kms(jnp.array(1000.0))
        expected = _C_KM_S / (_FWHM_TO_SIGMA * 1000.0)
        assert float(sigma) == pytest.approx(float(expected), rel=1e-10)

    def test_higher_resolution_narrower_velocity_width(self):
        """Higher R (better resolution) gives smaller velocity width sigma."""
        sigma_100 = _resolution_to_sigma_kms(jnp.array(100.0))
        sigma_1000 = _resolution_to_sigma_kms(jnp.array(1000.0))
        assert float(sigma_100) > float(sigma_1000)


# ── Constant-R LSF ────────────────────────────────────────────────


class TestApplyLSFConstantR:
    """Constant-R LSF: conservation, smoothing, and limits."""

    def test_smooths_delta_function(self, wave, delta_spectrum):
        """LSF reduces the peak of a delta function (shows broadening)."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0)
        assert float(jnp.max(smoothed)) < float(jnp.max(delta_spectrum))

    def test_lower_resolution_more_smoothing(self, wave, delta_spectrum):
        """Lower R (broader LSF kernel) produces more smoothing."""
        peak_r50 = float(jnp.max(apply_lsf(delta_spectrum, wave, resolution=50.0)))
        peak_r100 = float(jnp.max(apply_lsf(delta_spectrum, wave, resolution=100.0)))
        peak_r500 = float(jnp.max(apply_lsf(delta_spectrum, wave, resolution=500.0)))
        assert peak_r50 < peak_r100 < peak_r500

    def test_flux_conservation(self, wave, delta_spectrum):
        """Total flux is conserved under LSF convolution (∫F dλ before = after)."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0)
        assert_allclose(
            float(jnp.sum(smoothed)),
            float(jnp.sum(delta_spectrum)),
            rtol=1e-6,
        )

    def test_flat_spectrum_limit(self, wave, flat_spectrum):
        """Flat spectrum is unchanged by LSF (interior pixels, no edge effects)."""
        smoothed = apply_lsf(flat_spectrum, wave, resolution=100.0)
        interior = slice(50, -50)
        assert_allclose(
            smoothed[interior],
            flat_spectrum[interior],
            rtol=1e-5,
        )

    def test_high_resolution_near_identity(self, wave, delta_spectrum):
        """Very high R (very narrow LSF) barely changes the spectrum (limit)."""
        smoothed = apply_lsf(delta_spectrum, wave, resolution=50000.0)
        assert_allclose(smoothed, delta_spectrum, atol=0.01)


# ── Library resolution subtraction ────────────────────────────────


class TestLibrarySubtraction:
    """Sigma quadrature subtraction: sigma_eff² = sigma_inst² - sigma_lib²."""

    def test_library_subtraction_reduces_smoothing(self, wave, delta_spectrum):
        """Subtracting library resolution reduces effective smoothing."""
        # With sigma_lib=0, full instrument smoothing
        smoothed_full = apply_lsf(delta_spectrum, wave, resolution=100.0, sigma_lib_kms=0.0)
        # With sigma_lib > 0, less additional smoothing
        smoothed_sub = apply_lsf(delta_spectrum, wave, resolution=100.0, sigma_lib_kms=50.0)
        # Subtracted version has higher peak (less smoothed)
        assert float(jnp.max(smoothed_sub)) > float(jnp.max(smoothed_full))

    def test_library_exceeds_instrument_no_smoothing(self, wave, delta_spectrum):
        """When sigma_lib > sigma_inst, sigma_eff becomes zero."""
        # R=100 gives sigma_inst ~ 1275 km/s; sigma_lib=2000 exceeds it
        smoothed = apply_lsf(delta_spectrum, wave, resolution=100.0, sigma_lib_kms=2000.0)
        # Should be essentially unchanged
        assert_allclose(smoothed, delta_spectrum, atol=1e-10)


# ── Variable-R LSF ────────────────────────────────────────────────


class TestApplyLSFVariableR:
    """Variable-R LSF: conservation and convergence with bin resolution."""

    def test_smooths_delta_with_variable_r(self, wave, delta_spectrum):
        """Variable-R LSF reduces delta peak (shows convolution works)."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        smoothed = apply_lsf(delta_spectrum, wave, resolution=R_var)
        assert float(jnp.max(smoothed)) < float(jnp.max(delta_spectrum))

    def test_flux_conservation_variable_r(self, wave, delta_spectrum):
        """Flux is conserved for variable R (quadrature on piecewise bins)."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        smoothed = apply_lsf(delta_spectrum, wave, resolution=R_var)
        assert_allclose(
            float(jnp.sum(smoothed)),
            float(jnp.sum(delta_spectrum)),
            rtol=0.02,  # Slightly looser for piecewise approximation
        )

    def test_constant_array_matches_scalar(self, wave, delta_spectrum):
        """Uniform R array should match scalar R result (piecewise → continuous limit)."""
        R_scalar = 200.0
        R_array = R_scalar * jnp.ones_like(wave)
        smoothed_scalar = apply_lsf(delta_spectrum, wave, resolution=R_scalar)
        smoothed_array = apply_lsf(delta_spectrum, wave, resolution=R_array)
        # Piecewise approximation should be close
        assert_allclose(smoothed_array, smoothed_scalar, rtol=0.05, atol=0.05)

    def test_convergence_with_bin_count(self, wave, delta_spectrum):
        """More bins give result closer to scalar-R case (convergence)."""
        R_const = 150.0 * jnp.ones_like(wave)
        ref = apply_lsf(delta_spectrum, wave, resolution=150.0)
        err_8 = jnp.max(jnp.abs(apply_lsf(delta_spectrum, wave, R_const, n_bins=8) - ref))
        err_32 = jnp.max(jnp.abs(apply_lsf(delta_spectrum, wave, R_const, n_bins=32) - ref))
        assert float(err_32) <= float(err_8) + 1e-10


# ── Differentiability and JIT ─────────────────────────────────────


class TestLSFGradients:
    """Gradients through LSF convolution are finite and correct."""

    def test_gradient_finite_constant_r(self, wave, delta_spectrum):
        """Gradient w.r.t. input flux is finite (constant R)."""

        def loss(spec):
            return jnp.sum(apply_lsf(spec, wave, resolution=100.0) ** 2)

        g = assert_grad_matches_fd(loss, delta_spectrum)
        assert jnp.all(jnp.isfinite(g))

    def test_gradient_finite_variable_r(self, wave, delta_spectrum):
        """Gradient w.r.t. input flux is finite (variable R)."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)

        def loss(spec):
            return jnp.sum(apply_lsf(spec, wave, resolution=R_var) ** 2)

        g = assert_grad_matches_fd(loss, delta_spectrum)
        assert jnp.all(jnp.isfinite(g))

    def test_gradient_wrt_resolution_matches_finite_difference(self, wave, delta_spectrum):
        """Gradient w.r.t. scalar resolution matches finite differences."""

        def loss(R):
            return jnp.sum(apply_lsf(delta_spectrum, wave, resolution=R) ** 2)

        grad_jax = float(jax.grad(loss)(100.0))
        grad_fd = fd_grad(loss, 100.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_jit_constant_r(self, wave, delta_spectrum):
        """JIT compilation works for constant R (structural compatibility)."""
        result = assert_jit_matches_eager(
            lambda s: apply_lsf(s, wave, resolution=100.0), delta_spectrum
        )
        assert jnp.all(jnp.isfinite(result))

    def test_jit_variable_r(self, wave, delta_spectrum):
        """JIT compilation works for variable R (structural compatibility)."""
        R_var = 30.0 + 55.0 * (wave / 1e4 - 0.6)
        result = assert_jit_matches_eager(
            lambda s: apply_lsf(s, wave, resolution=R_var), delta_spectrum
        )
        assert jnp.all(jnp.isfinite(result))
