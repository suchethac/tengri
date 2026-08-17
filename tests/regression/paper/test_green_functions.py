# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.analysis.diagnostics.green_functions.

Tests cover:
- compute_green_function: filter mode and monochromatic mode
- compute_window_function: product with SFH
- compute_window_function_fourier: shape, frequency axis, non-negative power
- compute_time_sensitivity_matrix: shape and consistency with per-wavelength Green's functions
"""

import chex
import jax.numpy as jnp
import pytest

from tengri.analysis.diagnostics.green_functions import (
    compute_green_function,
    compute_time_sensitivity_matrix,
    compute_window_function,
    compute_window_function_fourier,
)
from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_paper


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_ssp():
    """Minimal SSP: 20 ages × 100 wavelengths.

    SSP flux is linearly decreasing with age to make the expected values
    analytically predictable.
    """
    n_age = 20
    n_wave = 100
    ssp_wave = jnp.linspace(1000.0, 10000.0, n_wave)
    # Flux = (n_age - i) / n_age at every wavelength, so older SSPs are fainter
    ages_factor = jnp.linspace(1.0, 0.1, n_age)  # shape (n_age,)
    ssp_flux = jnp.outer(ages_factor, jnp.ones(n_wave))  # (n_age, n_wave)
    ages_yr = jnp.logspace(6.0, 10.0, n_age)  # 1 Myr → 10 Gyr
    return ssp_flux, ssp_wave, ages_yr


@pytest.fixture()
def uniform_ssp():
    """SSP with flat spectra — every age has the same flat flux = 1.0."""
    n_age = 15
    n_wave = 80
    ssp_wave = jnp.linspace(2000.0, 9000.0, n_wave)
    ssp_flux = jnp.ones((n_age, n_wave))
    ages_yr = jnp.linspace(1e6, 1e10, n_age)
    return ssp_flux, ssp_wave, ages_yr


# ── compute_green_function — monochromatic mode ───────────────────


class TestComputeGreenFunctionMonochromatic:
    def test_values_positive_for_positive_flux(self, synthetic_ssp):
        """Green function must be positive and finite for positive flux (positivity bound)."""
        ssp_flux, ssp_wave, _ = synthetic_ssp
        g = compute_green_function(ssp_flux, ssp_wave, wave_target=5000.0)
        chex.assert_shape(g, (ssp_flux.shape[0],))
        chex.assert_tree_all_finite(g)
        assert jnp.all(g > 0)

    def test_decreasing_with_age_for_decreasing_ssp(self, synthetic_ssp):
        """synthetic_ssp has flux decreasing with age — so should G."""
        ssp_flux, ssp_wave, _ = synthetic_ssp
        g = compute_green_function(ssp_flux, ssp_wave, wave_target=5000.0)
        # All values should decrease monotonically
        assert jnp.all(jnp.diff(g) < 0)

    def test_uniform_ssp_constant_green_function(self, uniform_ssp):
        """Flat SSP → G(t) is the same at every age."""
        ssp_flux, ssp_wave, _ = uniform_ssp
        g = compute_green_function(ssp_flux, ssp_wave, wave_target=5000.0)
        assert jnp.allclose(g, g[0], rtol=1e-5)

    def test_wavelength_outside_grid_clamps_to_boundary(self, uniform_ssp):
        """jnp.interp (no left= kwarg) clamps to boundary value for out-of-range wavelengths.

        The monochromatic path uses plain jnp.interp without left/right=0, so
        extrapolation returns the edge SSP value rather than zero.
        """
        ssp_flux, ssp_wave, _ = uniform_ssp
        # Below the grid — should return ssp_flux[:, 0] = 1.0 (the left boundary)
        g = compute_green_function(ssp_flux, ssp_wave, wave_target=100.0)
        assert jnp.allclose(g, ssp_flux[:, 0], rtol=1e-5)

    def test_interpolation_at_exact_grid_point(self, uniform_ssp):
        """At an exact grid wavelength, interpolation should be exact."""
        ssp_flux, ssp_wave, _ = uniform_ssp
        wave_exact = float(ssp_wave[40])
        g = compute_green_function(ssp_flux, ssp_wave, wave_target=wave_exact)
        assert jnp.allclose(g, 1.0, rtol=1e-5)


# ── compute_green_function — filter mode ──────────────────────────


class TestComputeGreenFunctionFilter:
    def test_flat_filter_uniform_ssp(self, uniform_ssp):
        """Uniform SSP + flat filter → G(t) constant at all ages (limit test)."""
        ssp_flux, ssp_wave, _ = uniform_ssp
        filter_wave = jnp.linspace(3000.0, 7000.0, 50)
        filter_trans = jnp.ones(50)
        g = compute_green_function(
            ssp_flux, ssp_wave, filter_wave=filter_wave, filter_trans=filter_trans
        )
        chex.assert_shape(g, (ssp_flux.shape[0],))
        chex.assert_tree_all_finite(g)
        assert jnp.allclose(g, g[0], rtol=1e-4)

    def test_filter_outside_ssp_grid_near_zero(self, uniform_ssp):
        """Filter entirely outside SSP wavelength range → G ≈ 0."""
        ssp_flux, ssp_wave, _ = uniform_ssp
        filter_wave = jnp.linspace(100.0, 200.0, 20)  # far UV, outside 2000–9000 Å grid
        filter_trans = jnp.ones(20)
        g = compute_green_function(
            ssp_flux, ssp_wave, filter_wave=filter_wave, filter_trans=filter_trans
        )
        assert jnp.all(jnp.abs(g) < 1e-10)

    def test_raises_without_target_or_filter(self, synthetic_ssp):
        """Calling without either wave_target or filter args raises ValueError."""
        ssp_flux, ssp_wave, _ = synthetic_ssp
        with pytest.raises(ValueError, match="Provide either"):
            compute_green_function(ssp_flux, ssp_wave)


# ── compute_window_function ───────────────────────────────────────


class TestComputeWindowFunction:
    def test_definition_product(self):
        """W(t) = G(t) * SFR(t) — exact product (regression_paper: definition)."""
        g = jnp.array([1.0, 2.0, 3.0, 4.0])
        sfr = jnp.array([0.5, 1.0, 0.0, 2.0])
        w = compute_window_function(g, sfr)
        chex.assert_shape(w, (4,))
        expected = jnp.array([0.5, 2.0, 0.0, 8.0])
        assert jnp.allclose(w, expected, rtol=1e-6)

    def test_zero_sfr_gives_zero_window(self):
        g = jnp.ones(10)
        sfr = jnp.zeros(10)
        w = compute_window_function(g, sfr)
        assert jnp.all(w == 0.0)

    def test_zero_green_gives_zero_window(self):
        g = jnp.zeros(10)
        sfr = jnp.ones(10) * 5.0
        w = compute_window_function(g, sfr)
        assert jnp.all(w == 0.0)

    def test_constant_g_proportional_to_sfr(self):
        """When G is constant, W ∝ SFR."""
        g = jnp.ones(8) * 2.0
        sfr = jnp.arange(1.0, 9.0)
        w = compute_window_function(g, sfr)
        assert jnp.allclose(w, 2.0 * sfr, rtol=1e-6)


# ── compute_window_function_fourier ───────────────────────────────


class TestComputeWindowFunctionFourier:
    def test_power_non_negative(self, synthetic_ssp):
        """Power spectrum must be non-negative (positivity bound)."""
        _, _, ages_yr = synthetic_ssp
        n_age = ages_yr.shape[0]
        window_fn = jnp.ones(n_age)
        power, omega = compute_window_function_fourier(window_fn, ages_yr)
        # rfft output length for n points is n//2 + 1
        expected_len = n_age // 2 + 1
        chex.assert_shape(power, (expected_len,))
        chex.assert_shape(omega, (expected_len,))
        chex.assert_tree_all_finite(power)
        chex.assert_tree_all_finite(omega)
        assert_non_negative(power, name="power")

    def test_zero_window_gives_zero_power(self, synthetic_ssp):
        _, _, ages_yr = synthetic_ssp
        window_fn = jnp.zeros(ages_yr.shape[0])
        power, _ = compute_window_function_fourier(window_fn, ages_yr)
        assert jnp.all(power == 0.0)

    def test_dc_component_positive_for_positive_window(self, synthetic_ssp):
        """DC component (omega=0, index 0) is always ≥ 0 for a positive window."""
        _, _, ages_yr = synthetic_ssp
        window_fn = jnp.abs(jnp.sin(jnp.linspace(0, jnp.pi, ages_yr.shape[0]))) + 0.1
        power, _ = compute_window_function_fourier(window_fn, ages_yr)
        assert power[0] > 0.0

    def test_omega_starts_at_zero(self, synthetic_ssp):
        """First frequency should be 0 (DC component)."""
        _, _, ages_yr = synthetic_ssp
        window_fn = jnp.ones(ages_yr.shape[0])
        _, omega = compute_window_function_fourier(window_fn, ages_yr)
        assert float(omega[0]) == pytest.approx(0.0, abs=1e-20)

    def test_omega_monotonically_increasing(self, synthetic_ssp):
        _, _, ages_yr = synthetic_ssp
        window_fn = jnp.ones(ages_yr.shape[0])
        _, omega = compute_window_function_fourier(window_fn, ages_yr)
        assert_non_negative(jnp.diff(omega), name="output")


# ── compute_time_sensitivity_matrix ───────────────────────────────


class TestComputeTimeSensitivityMatrix:
    def test_rows_match_individual_green_functions(self, synthetic_ssp):
        """Each row of the matrix equals compute_green_function at that wavelength
        (regression_paper).
        """
        ssp_flux, ssp_wave, _ = synthetic_ssp
        wavelengths = jnp.array([3500.0, 5500.0, 7000.0])
        s = compute_time_sensitivity_matrix(ssp_flux, ssp_wave, wavelengths)
        n_age = ssp_flux.shape[0]
        chex.assert_shape(s, (3, n_age))
        chex.assert_tree_all_finite(s)
        for i, wave in enumerate(wavelengths):
            g = compute_green_function(ssp_flux, ssp_wave, wave_target=float(wave))
            assert jnp.allclose(s[i], g, rtol=1e-6), f"Row {i} mismatch for λ={wave} Å"

    def test_single_wavelength(self, synthetic_ssp):
        ssp_flux, ssp_wave, _ = synthetic_ssp
        wavelengths = jnp.array([5000.0])
        s = compute_time_sensitivity_matrix(ssp_flux, ssp_wave, wavelengths)
        chex.assert_shape(s, (1, ssp_flux.shape[0]))

    def test_uniform_ssp_all_rows_equal(self, uniform_ssp):
        """Uniform SSP flux → G(t) is the same for every wavelength within the grid."""
        ssp_flux, ssp_wave, _ = uniform_ssp
        wavelengths = jnp.array([3000.0, 5000.0, 7000.0, 8000.0])
        s = compute_time_sensitivity_matrix(ssp_flux, ssp_wave, wavelengths)
        # All rows should be identical (G = 1 at every age, every wavelength)
        for i in range(1, s.shape[0]):
            assert jnp.allclose(s[0], s[i], rtol=1e-5)
