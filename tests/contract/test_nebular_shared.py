# SPDX-License-Identifier: BSD-3-Clause
"""Tests for models/nebular/_shared.py — shared nebular utilities.

Covers:
  - place_line_profiles  (Gaussian and delta-function paths)
  - compute_qh            (ionizing photon rate)
  - compute_analytic_nebular_continuum  (free-free + two-photon)
  - NebularContinuumFallback (four-tier cascade wrapper)
"""

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri.components.nebular._constants import (
    _LOG10_ZSUN,
    _LOG_OH_OFFSET,
)
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def obs_wave():
    """Optical/UV wavelength grid (Angstrom), 1000–10 000 Å."""
    return jnp.linspace(1000.0, 10000.0, 1000)


@pytest.fixture
def broad_wave():
    """Broad wavelength grid from UV to NIR (Angstrom)."""
    return jnp.logspace(2.5, 4.5, 500)


@pytest.fixture
def single_line_wave():
    """A single Hα-like line wavelength."""
    return jnp.array([6564.61])  # vacuum Hα


@pytest.fixture
def single_line_lum():
    """Luminosity for a single line."""
    return jnp.array([1.0e40])  # erg/s


# ── place_line_profiles ───────────────────────────────────────────


class TestPlaceLineProfiles:
    """Tests for the line-profile placement routine."""

    def test_gaussian_output_shape(self, obs_wave, single_line_wave, single_line_lum):
        """Output has same shape as obs_wavelengths."""
        from tengri.components.nebular._shared import place_line_profiles

        sed = place_line_profiles(single_line_wave, single_line_lum, obs_wave, line_sigma_aa=10.0)
        chex.assert_equal_shape([sed, obs_wave])

    def test_gaussian_finite(self, obs_wave, single_line_wave, single_line_lum):
        from tengri.components.nebular._shared import place_line_profiles

        sed = place_line_profiles(single_line_wave, single_line_lum, obs_wave, line_sigma_aa=10.0)
        chex.assert_tree_all_finite(sed)

    def test_gaussian_non_negative(self, obs_wave, single_line_wave, single_line_lum):
        from tengri.components.nebular._shared import place_line_profiles

        sed = place_line_profiles(single_line_wave, single_line_lum, obs_wave, line_sigma_aa=10.0)
        assert jnp.all(sed >= 0.0)

    def test_gaussian_peaks_near_line_center(self, obs_wave, single_line_wave, single_line_lum):
        """Gaussian profile peaks close to the specified line wavelength."""
        from tengri.components.nebular._shared import place_line_profiles

        sed = place_line_profiles(single_line_wave, single_line_lum, obs_wave, line_sigma_aa=10.0)
        peak_wave = float(obs_wave[jnp.argmax(sed)])
        # Should be within 50 Å of Hα
        assert abs(peak_wave - 6564.61) < 50.0

    def test_zero_luminosity_gives_zero(self, obs_wave, single_line_wave):
        """Zero line luminosity → zero SED everywhere."""
        from tengri.components.nebular._shared import place_line_profiles

        lum_zero = jnp.array([0.0])
        sed = place_line_profiles(single_line_wave, lum_zero, obs_wave, line_sigma_aa=10.0)
        assert jnp.allclose(sed, 0.0)

    def test_delta_function_output_shape(self, obs_wave, single_line_wave, single_line_lum):
        """Delta-function path (sigma <= 0) has correct shape."""
        from tengri.components.nebular._shared import place_line_profiles

        sed = place_line_profiles(single_line_wave, single_line_lum, obs_wave, line_sigma_aa=0.0)
        chex.assert_equal_shape([sed, obs_wave])

    def test_delta_function_finite_non_negative(self, obs_wave, single_line_wave, single_line_lum):
        from tengri.components.nebular._shared import place_line_profiles

        sed = place_line_profiles(single_line_wave, single_line_lum, obs_wave, line_sigma_aa=-1.0)
        chex.assert_tree_all_finite(sed)
        assert jnp.all(sed >= 0.0)

    def test_multiple_lines(self, obs_wave):
        """Multiple lines with different luminosities are placed independently."""
        from tengri.components.nebular._shared import place_line_profiles

        # Hβ + Hα vacuum
        line_waves = jnp.array([4862.68, 6564.61])
        line_lums = jnp.array([1.0e39, 2.8e39])
        sed = place_line_profiles(line_waves, line_lums, obs_wave, line_sigma_aa=5.0)
        chex.assert_equal_shape([sed, obs_wave])
        chex.assert_tree_all_finite(sed)
        assert jnp.all(sed >= 0.0)
        # Should have non-trivial flux near both lines
        hb_mask = (obs_wave > 4800.0) & (obs_wave < 4920.0)
        ha_mask = (obs_wave > 6500.0) & (obs_wave < 6630.0)
        assert jnp.sum(sed[hb_mask]) > 0.0
        assert jnp.sum(sed[ha_mask]) > 0.0

    def test_broader_sigma_spreads_flux(self, obs_wave, single_line_wave, single_line_lum):
        """Wider Gaussian spreads flux over more wavelengths (lower peak, broader wings)."""
        from tengri.components.nebular._shared import place_line_profiles

        sed_narrow = place_line_profiles(
            single_line_wave, single_line_lum, obs_wave, line_sigma_aa=5.0
        )
        sed_wide = place_line_profiles(
            single_line_wave, single_line_lum, obs_wave, line_sigma_aa=50.0
        )
        # Narrower has higher peak
        assert float(jnp.max(sed_narrow)) > float(jnp.max(sed_wide))

    def test_luminosity_linearity(self, obs_wave, single_line_wave):
        """Doubling luminosity doubles the SED everywhere."""
        from tengri.components.nebular._shared import place_line_profiles

        lum1 = jnp.array([1.0e39])
        lum2 = jnp.array([2.0e39])
        sed1 = place_line_profiles(single_line_wave, lum1, obs_wave, line_sigma_aa=10.0)
        sed2 = place_line_profiles(single_line_wave, lum2, obs_wave, line_sigma_aa=10.0)
        ratio = sed2 / jnp.maximum(sed1, 1e-100)
        # Mask pixels far from the line (where both are ~0)
        near_line = (obs_wave > 6500.0) & (obs_wave < 6630.0)
        np.testing.assert_allclose(float(jnp.mean(ratio[near_line])), 2.0, rtol=1e-5)


# ── compute_qh ────────────────────────────────────────────────────


class TestComputeQh:
    """Tests for the ionizing photon rate computation."""

    def test_non_negative(self):
        """compute_qh always returns a non-negative value."""
        from tengri.components.nebular._shared import compute_qh

        wave = jnp.linspace(100.0, 2000.0, 500)
        flux = jnp.ones_like(wave)
        qh = compute_qh(wave, flux)
        assert float(qh) >= 0.0

    def test_zero_flux_gives_zero(self):
        """Zero flux → Q_H ≈ 0."""
        from tengri.components.nebular._shared import compute_qh

        wave = jnp.linspace(100.0, 2000.0, 500)
        flux = jnp.zeros_like(wave)
        qh = compute_qh(wave, flux)
        assert float(qh) == pytest.approx(0.0, abs=1e-30)

    def test_flux_only_longward_of_lyman_gives_zero(self):
        """Flux only at λ > 912 Å contributes nothing to Q_H."""
        from tengri.components.nebular._shared import compute_qh

        wave = jnp.linspace(1000.0, 5000.0, 500)  # all > 912 Å
        flux = jnp.ones_like(wave)
        qh = compute_qh(wave, flux)
        assert float(qh) == pytest.approx(0.0, abs=1e-30)

    def test_ionizing_flux_below_lyman_limit(self):
        """Flux only at λ < 912 Å should give positive Q_H."""
        from tengri.components.nebular._shared import compute_qh

        # Grid spanning 100–900 Å only
        wave = jnp.linspace(100.0, 900.0, 500)
        flux = jnp.ones_like(wave) * 1e-10  # small but non-zero
        qh = compute_qh(wave, flux)
        assert float(qh) > 0.0

    def test_more_flux_more_photons(self):
        """Higher ionizing flux → larger Q_H."""
        from tengri.components.nebular._shared import compute_qh

        wave = jnp.linspace(100.0, 2000.0, 500)
        flux_low = jnp.where(wave < 912.0, 1e-10, 0.0)
        flux_high = jnp.where(wave < 912.0, 1e-9, 0.0)
        qh_low = compute_qh(wave, flux_low)
        qh_high = compute_qh(wave, flux_high)
        assert float(qh_high) > float(qh_low)

    def test_jit_compatible(self):
        """compute_qh is already @jax.jit decorated; verify it runs."""
        from tengri.components.nebular._shared import compute_qh

        wave = jnp.linspace(100.0, 2000.0, 200)
        flux = jnp.ones_like(wave) * 1e-15
        qh = compute_qh(wave, flux)
        assert jnp.isfinite(qh)


# ── compute_analytic_nebular_continuum ────────────────────────────


class TestComputeAnalyticNebularContinuum:
    """Tests for the analytic nebular continuum (free-free + two-photon)."""

    @pytest.fixture
    def wave_optical(self):
        """Optical wavelength grid (1000–12 000 Å)."""
        return jnp.linspace(1000.0, 12000.0, 600)

    def test_finite_non_negative(self, wave_optical):
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        cont = compute_analytic_nebular_continuum(wave_optical, q_h=1e49, log_z_abs=-1.848)
        chex.assert_tree_all_finite(cont)
        assert jnp.all(cont >= 0.0)

    def test_output_shape(self, wave_optical):
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        cont = compute_analytic_nebular_continuum(wave_optical, q_h=1e49, log_z_abs=-1.848)
        chex.assert_equal_shape([cont, wave_optical])

    def test_zero_qh_gives_zero(self, wave_optical):
        """No ionizing photons → no nebular continuum."""
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        cont = compute_analytic_nebular_continuum(wave_optical, q_h=0.0, log_z_abs=-1.848)
        assert jnp.allclose(cont, 0.0, atol=1e-60)

    def test_scales_linearly_with_qh(self, wave_optical):
        """Q_H doubles → continuum doubles (linear normalization)."""
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        cont1 = compute_analytic_nebular_continuum(wave_optical, q_h=1e49, log_z_abs=-1.848)
        cont2 = compute_analytic_nebular_continuum(wave_optical, q_h=2e49, log_z_abs=-1.848)
        ratio = jnp.sum(cont2) / jnp.sum(cont1)
        np.testing.assert_allclose(float(ratio), 2.0, rtol=1e-5)

    def test_two_photon_longward_of_lya(self, wave_optical):
        """Two-photon emission is only present at λ > 1216 Å (λ_Lyα)."""
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        # Wave grid that straddles Lya
        wave = jnp.linspace(900.0, 2000.0, 300)
        cont = compute_analytic_nebular_continuum(wave, q_h=1e49, log_z_abs=-1.848)
        # All finite and non-negative
        chex.assert_tree_all_finite(cont)
        assert jnp.all(cont >= 0.0)

    def test_temperature_changes_continuum_shape(self, wave_optical):
        """Different electron temperatures produce different SED shapes."""
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        cont_t4 = compute_analytic_nebular_continuum(
            wave_optical, q_h=1e49, log_z_abs=-1.848, temperature=1e4
        )
        cont_t5 = compute_analytic_nebular_continuum(
            wave_optical, q_h=1e49, log_z_abs=-1.848, temperature=2e4
        )
        assert not jnp.allclose(cont_t4, cont_t5, rtol=0.01)

    def test_gradient_wrt_qh(self, wave_optical):
        """FD check: ∂(∑continuum)/∂q_h."""
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        def loss(q_h):
            return jnp.sum(
                compute_analytic_nebular_continuum(wave_optical, q_h=q_h, log_z_abs=-1.848)
            )

        g_jax = float(jax.grad(loss)(1e49))
        g_fd = fd_grad(loss, 1e49, eps=1e46)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_jit_compatible(self, wave_optical):
        """compute_analytic_nebular_continuum is JIT-compilable."""
        from tengri.components.nebular._shared import compute_analytic_nebular_continuum

        cont = assert_jit_matches_eager(
            compute_analytic_nebular_continuum, wave_optical, 1e49, -1.848
        )
        chex.assert_tree_all_finite(cont)


# ── NebularContinuumFallback ──────────────────────────────────────


class _MockLineOnlyBackend:
    """Minimal mock for a line-only nebular backend."""

    has_continuum = False
    has_free_params = False
    name = "mock_line_only"

    def __init__(self, wave, line_sed=None):
        self._wave = wave
        self._line_sed = line_sed if line_sed is not None else jnp.zeros_like(wave)

    def predict_nebular_sed(self, *args, **kwargs):
        return self._line_sed


class _MockContinuumBackend:
    """Minimal mock for a continuum-capable backend."""

    has_continuum = True
    name = "mock_continuum"

    def __init__(self, wave, cont_sed):
        self._cont_sed = cont_sed

    def predict_nebular_sed(self, *args, **kwargs):
        return self._cont_sed


class TestNebularContinuumFallback:
    """Tests for the four-tier NebularContinuumFallback cascade."""

    @pytest.fixture
    def wave(self):
        return jnp.linspace(1000.0, 10000.0, 300)

    def test_invalid_fallback_mode_raises(self, wave):
        """fallback_mode other than 'error'/'warn' raises ValueError."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        primary = _MockLineOnlyBackend(wave)
        with pytest.raises(ValueError, match="fallback_mode must be"):
            NebularContinuumFallback(primary, fallback_mode="silent")

    def test_has_continuum_true(self, wave):
        """Wrapper always reports has_continuum=True."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        primary = _MockLineOnlyBackend(wave)
        wrapped = NebularContinuumFallback(primary)
        assert wrapped.has_continuum is True

    def test_name_attribute(self, wave):
        """Name includes primary backend name."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        primary = _MockLineOnlyBackend(wave)
        wrapped = NebularContinuumFallback(primary)
        assert "mock_line_only" in wrapped.name

    def test_tier1_secondary_backend(self, wave):
        """Tier 1: secondary backend continuum is added to primary lines."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        lines = jnp.ones_like(wave) * 1e-30
        cont = jnp.ones_like(wave) * 2e-30
        primary = _MockLineOnlyBackend(wave, lines)
        fallback = _MockContinuumBackend(wave, cont)
        wrapped = NebularContinuumFallback(primary, fallback=fallback)

        result = wrapped.predict_nebular_sed()
        expected = lines + cont
        np.testing.assert_array_equal(np.array(result), np.array(expected))

    def test_tier2_analytic_continuum(self, wave):
        """Tier 2: analytic continuum is added when ssp_wave + gas_logqion provided."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        lines = jnp.zeros_like(wave)
        primary = _MockLineOnlyBackend(wave, lines)
        wrapped = NebularContinuumFallback(primary, fallback_mode="warn")

        result = wrapped.predict_nebular_sed(ssp_wave=wave, gas_logqion=49.0)
        # Should have non-zero continuum added
        assert jnp.any(result > 0.0)
        chex.assert_tree_all_finite(result)

    def test_tier3_error_mode_raises(self, wave):
        """Tier 3: fallback_mode='error' raises NebularContinuumUnavailableError."""
        from tengri.components.nebular._protocol import NebularContinuumUnavailableError
        from tengri.components.nebular._shared import NebularContinuumFallback

        primary = _MockLineOnlyBackend(wave)
        wrapped = NebularContinuumFallback(primary, fallback_mode="error")

        with pytest.raises(NebularContinuumUnavailableError):
            wrapped.predict_nebular_sed()  # no ssp_wave or gas_logqion

    def test_tier4_warn_mode_returns_lines_only(self, wave):
        """Tier 4: fallback_mode='warn' returns lines only and emits UserWarning."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        lines = jnp.ones_like(wave) * 1e-30
        primary = _MockLineOnlyBackend(wave, lines)
        wrapped = NebularContinuumFallback(primary, fallback_mode="warn")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = wrapped.predict_nebular_sed()  # no ssp_wave or gas_logqion

        # Should warn
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        # Should return lines unchanged
        np.testing.assert_array_equal(np.array(result), np.array(lines))

    def test_attribute_delegation(self, wave):
        """Unknown attributes are delegated to the primary backend."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        primary = _MockLineOnlyBackend(wave)
        primary.custom_attr = "test_value"
        wrapped = NebularContinuumFallback(primary)
        assert wrapped.custom_attr == "test_value"

    def test_tier1_takes_priority_over_tier2(self, wave):
        """When both fallback and ssp_wave+gas_logqion are present, Tier 1 wins."""
        from tengri.components.nebular._shared import NebularContinuumFallback

        lines = jnp.zeros_like(wave)
        cont_tier1 = jnp.ones_like(wave) * 5e-30
        primary = _MockLineOnlyBackend(wave, lines)
        fallback = _MockContinuumBackend(wave, cont_tier1)
        wrapped = NebularContinuumFallback(primary, fallback=fallback)

        # Pass ssp_wave + gas_logqion — Tier 1 should still dominate
        result = wrapped.predict_nebular_sed(ssp_wave=wave, gas_logqion=49.0)
        # Result should equal lines + cont_tier1 (Tier 1), not lines + analytic
        expected = lines + cont_tier1
        np.testing.assert_array_equal(np.array(result), np.array(expected))


# ── _interp_index_weight ──────────────────────────────────────────


class TestInterpIndexWeight:
    """Tests for the private piecewise-linear index/weight helper."""

    @staticmethod
    def _grid(n=10):
        return jnp.linspace(0.0, 1.0, n)

    def test_midpoint_weight(self):
        """x at midpoint of first cell → idx=0, w≈0.5."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = self._grid(5)  # [0, 0.25, 0.5, 0.75, 1.0]
        idx, w = _interp_index_weight(0.125, grid)  # midpoint of [0, 0.25]
        assert int(idx) == 0
        assert float(w) == pytest.approx(0.5, abs=1e-6)

    def test_at_grid_node_is_zero_weight(self):
        """x exactly at a node → weight = 0 (left edge of cell)."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = self._grid(5)
        idx, w = _interp_index_weight(0.25, grid)  # exactly grid[1]
        assert int(idx) == 1
        assert float(w) == pytest.approx(0.0, abs=1e-6)

    def test_at_last_grid_point(self):
        """x = grid[-1] → idx = n-2, w = 1.0."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = self._grid(5)
        idx, w = _interp_index_weight(1.0, grid)
        assert int(idx) == 3  # len(grid) - 2
        assert float(w) == pytest.approx(1.0, abs=1e-6)

    def test_clamp_below_grid(self):
        """x < grid[0] is clamped → idx=0, w=0.0."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = self._grid(5)
        idx, w = _interp_index_weight(-10.0, grid)
        assert int(idx) == 0
        assert float(w) == pytest.approx(0.0, abs=1e-6)

    def test_clamp_above_grid(self):
        """x > grid[-1] is clamped → idx=n-2, w=1.0."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = self._grid(5)
        idx, w = _interp_index_weight(100.0, grid)
        assert int(idx) == 3
        assert float(w) == pytest.approx(1.0, abs=1e-6)

    def test_weight_in_unit_interval(self):
        """w is always in [0, 1] for any x."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = self._grid(10)
        for x_val in [-1.0, 0.0, 0.33, 0.67, 1.0, 2.0]:
            _idx, w = _interp_index_weight(x_val, grid)
            assert 0.0 <= float(w) <= 1.0

    def test_linear_reconstruction(self):
        """Piecewise-linear interpolation recovers x for f(x) = x."""
        from tengri.components.nebular._shared import _interp_index_weight

        grid = jnp.linspace(0.0, 1.0, 11)
        x = 0.63
        idx, w = _interp_index_weight(x, grid)
        reconstructed = float(grid[idx]) * (1 - float(w)) + float(grid[idx + 1]) * float(w)
        assert reconstructed == pytest.approx(x, abs=1e-6)


# ── Metallicity convention converters ─────────────────────────────


class TestMetallicityConverters:
    """Tests for neb_logzsol_to_log_z_abs / _cloudy_logoh / _mappings_zeta."""

    def test_log_z_abs_at_solar(self):
        """logzsol=0 → log_z_abs = log10(Z_sun) ≈ -1.848."""
        from tengri.components.nebular._shared import neb_logzsol_to_log_z_abs

        result = float(neb_logzsol_to_log_z_abs(0.0))
        assert result == pytest.approx(_LOG10_ZSUN, rel=1e-6)

    def test_log_z_abs_sub_solar(self):
        """logzsol=-1 → log_z_abs shifts by -1 from solar."""
        from tengri.components.nebular._shared import neb_logzsol_to_log_z_abs

        result = float(neb_logzsol_to_log_z_abs(-1.0))
        assert result == pytest.approx(_LOG10_ZSUN - 1.0, rel=1e-6)

    def test_log_z_abs_super_solar(self):
        """logzsol=0.3 → log_z_abs = LOG10_ZSUN + 0.3."""
        from tengri.components.nebular._shared import neb_logzsol_to_log_z_abs

        result = float(neb_logzsol_to_log_z_abs(0.3))
        assert result == pytest.approx(_LOG10_ZSUN + 0.3, rel=1e-6)

    def test_cloudy_logoh_at_solar(self):
        """logzsol=0 → log(O/H) = _LOG10_ZSUN - _LOG_OH_OFFSET ≈ -3.07."""
        from tengri.components.nebular._shared import neb_logzsol_to_cloudy_logoh

        result = float(neb_logzsol_to_cloudy_logoh(0.0))
        expected = _LOG10_ZSUN - _LOG_OH_OFFSET
        assert result == pytest.approx(expected, rel=1e-6)

    def test_cloudy_logoh_shifts_additively(self):
        """logzsol offset propagates additively to log(O/H)."""
        from tengri.components.nebular._shared import neb_logzsol_to_cloudy_logoh

        r0 = float(neb_logzsol_to_cloudy_logoh(0.0))
        r1 = float(neb_logzsol_to_cloudy_logoh(1.0))
        assert r1 == pytest.approx(r0 + 1.0, rel=1e-6)

    def test_mappings_zeta_solar(self):
        """logzsol=0 → zeta = 1.0 (solar)."""
        from tengri.components.nebular._shared import neb_logzsol_to_mappings_zeta

        result = float(neb_logzsol_to_mappings_zeta(0.0))
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_mappings_zeta_subsolar(self):
        """logzsol=-1 → zeta = 0.1."""
        from tengri.components.nebular._shared import neb_logzsol_to_mappings_zeta

        result = float(neb_logzsol_to_mappings_zeta(-1.0))
        assert result == pytest.approx(0.1, rel=1e-5)

    def test_mappings_zeta_supersolar(self):
        """logzsol=1 → zeta = 10.0."""
        from tengri.components.nebular._shared import neb_logzsol_to_mappings_zeta

        result = float(neb_logzsol_to_mappings_zeta(1.0))
        assert result == pytest.approx(10.0, rel=1e-5)

    def test_all_converters_return_finite(self):
        """All three converters return finite values for scalar solar input."""
        from tengri.components.nebular._shared import (
            neb_logzsol_to_cloudy_logoh,
            neb_logzsol_to_log_z_abs,
            neb_logzsol_to_mappings_zeta,
        )

        for fn in (
            neb_logzsol_to_log_z_abs,
            neb_logzsol_to_cloudy_logoh,
            neb_logzsol_to_mappings_zeta,
        ):
            assert jnp.isfinite(fn(0.0))
