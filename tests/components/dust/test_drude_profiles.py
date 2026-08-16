# SPDX-License-Identifier: BSD-3-Clause
"""Tests for PAH Drude-profile decomposition (drude_profiles.py)."""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax.numpy as jnp

from tengri.components.dust.drude_profiles import (
    N_PAH_FEATURES,
    SMITH2007_PAH_FEATURES,
    compute_pah_template,
    decompose_pah,
    drude_profile,
)
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd

# Reference wavelength grid: 2–20 μm at 0.01 μm spacing.
_WAVE_UM = jnp.linspace(2.0, 20.0, 1800)


class TestFeatureTable:
    def test_feature_count(self):
        assert len(SMITH2007_PAH_FEATURES) == 18
        assert N_PAH_FEATURES == 18

    def test_reference_feature_is_strongest(self):
        """7.60 μm feature is the reference (strength = 1.0)."""

        feat = next(f for f in SMITH2007_PAH_FEATURES if abs(f.wave_um - 7.60) < 0.01)
        assert feat.strength == pytest.approx(1.0)

    def test_all_strengths_positive(self):
        for f in SMITH2007_PAH_FEATURES:
            assert f.strength > 0.0

    def test_all_gammas_positive(self):
        for f in SMITH2007_PAH_FEATURES:
            assert f.gamma > 0.0

    def test_features_are_namedtuple(self):
        f = SMITH2007_PAH_FEATURES[0]
        assert hasattr(f, "wave_um")
        assert hasattr(f, "gamma")
        assert hasattr(f, "strength")


class TestDrudeProfile:
    def test_peak_at_center(self):
        """Single Drude profile peaks exactly at its central wavelength."""
        center_um = 7.6
        gamma = 0.044
        # Fine grid around the center
        wave = jnp.linspace(center_um - 1.0, center_um + 1.0, 2001)
        profile = drude_profile(wave, center_um, gamma)
        peak_idx = int(jnp.argmax(profile))
        assert abs(float(wave[peak_idx]) - center_um) < 0.002

    def test_peak_value_is_unity(self):
        """Peak of an un-strengthed Drude profile is exactly 1."""
        center_um = 7.6
        gamma = 0.044
        wave = jnp.array([center_um])
        val = drude_profile(wave, center_um, gamma)
        assert float(val[0]) == pytest.approx(1.0, rel=1e-6)

    def test_fwhm_width(self):
        """FWHM of profile matches gamma * center."""
        center_um = 6.22
        gamma = 0.030
        expected_fwhm = gamma * center_um
        wave = jnp.linspace(center_um - 1.0, center_um + 1.0, 4001)
        profile = drude_profile(wave, center_um, gamma)
        half_max = 0.5
        # Find the two half-max crossings
        above = profile > half_max
        left_idx = int(jnp.argmax(above))
        right_idx = int(len(above) - 1 - jnp.argmax(above[::-1]))
        measured_fwhm = float(wave[right_idx]) - float(wave[left_idx])
        assert measured_fwhm == pytest.approx(expected_fwhm, rel=0.05)

    def test_output_shape(self):
        wave = jnp.linspace(3.0, 15.0, 500)
        out = drude_profile(wave, 7.6, 0.044)
        chex.assert_shape(out, (500,))

    def test_non_negative(self):
        wave = jnp.linspace(0.1, 30.0, 3000)
        out = drude_profile(wave, 7.6, 0.044)
        assert_non_negative(out, name="out")


class TestPAHTemplate:
    def test_output_shape(self):
        out = compute_pah_template(_WAVE_UM)
        chex.assert_equal_shape([out, _WAVE_UM])

    def test_non_negative_everywhere(self):
        out = compute_pah_template(_WAVE_UM)
        assert_non_negative(out, name="out")

    def test_default_and_explicit_strengths_match(self):
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        out_default = compute_pah_template(_WAVE_UM)
        out_explicit = compute_pah_template(_WAVE_UM, strengths=s)
        assert jnp.allclose(out_default, out_explicit, atol=1e-12)

    def test_zero_strengths_gives_zero(self):
        s = jnp.zeros(N_PAH_FEATURES)
        out = compute_pah_template(_WAVE_UM, strengths=s)
        assert jnp.allclose(out, 0.0, atol=1e-30)

    def test_scale_linearity(self):
        """Doubling all strengths doubles the output."""
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        out1 = compute_pah_template(_WAVE_UM, strengths=s)
        out2 = compute_pah_template(_WAVE_UM, strengths=2.0 * s)
        assert jnp.allclose(out2, 2.0 * out1, rtol=1e-6)

    def test_gradient_is_finite(self):
        """Gradient of sum(compute_pah_template) w.r.t. strengths is finite."""
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])

        def total(strengths):
            return jnp.sum(compute_pah_template(_WAVE_UM, strengths=strengths))

        g = assert_grad_matches_fd(total, s)
        chex.assert_tree_all_finite(g)

    def test_major_features_are_peaks(self):
        """The 7.60 and 6.22 μm features produce local maxima in a coarse SED."""
        wave = jnp.linspace(5.0, 15.0, 1001)
        out = compute_pah_template(wave)
        # Find feature values vs nearby background — just check they're non-negligible
        idx_760 = int(jnp.argmin(jnp.abs(wave - 7.6)))
        idx_622 = int(jnp.argmin(jnp.abs(wave - 6.22)))
        assert float(out[idx_760]) > 0.5 * float(jnp.max(out))
        assert float(out[idx_622]) > 0.1 * float(jnp.max(out))


class TestDecomposePAH:
    def _synthetic_pah(self, wave_um, strengths):
        """Build a noiseless synthetic PAH SED for round-trip tests."""
        return compute_pah_template(wave_um, strengths=strengths)

    def test_output_keys(self):
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        sed = self._synthetic_pah(_WAVE_UM, s)
        result = decompose_pah(_WAVE_UM, sed)
        assert set(result.keys()) == {"strengths", "fitted_pah", "residual"}

    def test_output_shapes(self):
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        sed = self._synthetic_pah(_WAVE_UM, s)
        result = decompose_pah(_WAVE_UM, sed)
        assert result["strengths"].shape == (N_PAH_FEATURES,)
        assert result["fitted_pah"].shape == _WAVE_UM.shape
        assert result["residual"].shape == _WAVE_UM.shape

    def test_fitted_plus_residual_equals_input(self):
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        sed = self._synthetic_pah(_WAVE_UM, s)
        result = decompose_pah(_WAVE_UM, sed)
        reconstructed = result["fitted_pah"] + result["residual"]
        assert jnp.allclose(reconstructed, sed, atol=1e-8)

    def test_round_trip_strengths(self):
        """Round-trip: decompose recovers input strengths within 1% for prominent features."""
        s_true = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        sed = self._synthetic_pah(_WAVE_UM, s_true)
        result = decompose_pah(_WAVE_UM, sed)
        # Relative error per feature (skip features with very small strength)
        mask = s_true > 0.05
        rel_err = jnp.abs(result["strengths"][mask] - s_true[mask]) / s_true[mask]
        assert jnp.all(rel_err < 0.01), f"Max rel error: {float(jnp.max(rel_err)):.4f}"

    def test_strengths_differentiable(self):
        """strengths from decompose_pah are JAX-differentiable w.r.t. the SED."""
        s_true = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        sed = self._synthetic_pah(_WAVE_UM, s_true)

        def total_strength(sed):
            return jnp.sum(decompose_pah(_WAVE_UM, sed)["strengths"])

        g = assert_grad_matches_fd(total_strength, sed)
        chex.assert_tree_all_finite(g)

    def test_continuum_subtraction(self):
        """With a flat continuum added, continuum= kwarg should recover same strengths."""
        s = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])
        pah = self._synthetic_pah(_WAVE_UM, s)
        continuum = jnp.ones_like(_WAVE_UM) * 0.5 * float(jnp.mean(pah))
        sed_with_cont = pah + continuum
        result_no_sub = decompose_pah(_WAVE_UM, pah)
        result_with_sub = decompose_pah(_WAVE_UM, sed_with_cont, continuum=continuum)
        assert jnp.allclose(result_no_sub["strengths"], result_with_sub["strengths"], rtol=1e-4)
