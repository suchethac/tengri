"""Tests for emission line design matrix and marginalization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri.models.observation.eline_marginalization import (
    apply_doublet_constraints,
    build_eline_design_matrix,
    expand_constrained_amplitudes,
    marginalize_emission_lines,
)
from tengri.models.observation.line_catalog import LineCatalog


class TestDesignMatrix:
    def test_shape(self):
        wave = jnp.linspace(4000, 7000, 1000)
        lines = jnp.array([4862.68, 5008.24, 6564.61])  # vacuum Hβ, [OIII]5007, Hα
        G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
        assert G.shape == (1000, 3)

    def test_normalized_profiles(self):
        """Each column should integrate to approximately 1."""
        wave = jnp.linspace(4000, 7000, 5000)
        lines = jnp.array([4862.68, 5008.24, 6564.61])  # vacuum Hβ, [OIII]5007, Hα
        G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
        dlam = float(jnp.mean(jnp.diff(wave)))
        for i in range(3):
            integral = float(jnp.sum(G[:, i])) * dlam
            assert abs(integral - 1.0) < 0.05

    def test_velocity_broadening_widens_profile(self):
        wave = jnp.linspace(6500, 6620, 500)
        ha = jnp.array([6564.61])  # vacuum Hα
        G_narrow = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_sigma_kms=0.0)
        G_broad = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_sigma_kms=200.0)
        fwhm_narrow = int(jnp.sum(G_narrow[:, 0] > 0.5 * G_narrow[:, 0].max()))
        fwhm_broad = int(jnp.sum(G_broad[:, 0] > 0.5 * G_broad[:, 0].max()))
        assert fwhm_broad > fwhm_narrow

    def test_velocity_offset_shifts_center(self):
        wave = jnp.linspace(6500, 6650, 1000)
        ha = jnp.array([6564.61])  # vacuum Hα
        G_zero = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_delta_v_kms=0.0)
        G_red = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_delta_v_kms=300.0)
        peak_zero = float(wave[jnp.argmax(G_zero[:, 0])])
        peak_red = float(wave[jnp.argmax(G_red[:, 0])])
        assert peak_red > peak_zero

    def test_redshift_shifts_lines(self):
        wave = jnp.linspace(9000, 10000, 500)
        ha = jnp.array([6564.61])  # vacuum Hα
        z = 0.5
        G = build_eline_design_matrix(wave, ha, 1000.0, z)
        peak = float(wave[jnp.argmax(G[:, 0])])
        assert abs(peak - 6564.61 * 1.5) < 20.0

    def test_backward_compatible_4arg_call(self):
        """Old 4-argument call must still work."""
        wave = jnp.linspace(4000, 7000, 500)
        lines = jnp.array([6564.61])  # vacuum Hα
        G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
        assert G.shape == (500, 1)


class TestDoubletConstraints:
    def test_constraint_reduces_columns(self):
        cat = LineCatalog.default_optical()
        C = cat.build_constraint_matrix()
        wave = jnp.linspace(4000, 7000, 1000)
        G_full = build_eline_design_matrix(wave, cat.wavelengths, 2000.0, 0.0)
        G_eff = apply_doublet_constraints(G_full, C)
        assert G_eff.shape[1] == cat.n_independent
        assert G_eff.shape[1] < G_full.shape[1]

    def test_doublet_ratio_enforced(self):
        """After marginalization with constraints, OIII ratio should be 2.98."""
        cat = LineCatalog.default_optical()
        C = cat.build_constraint_matrix()
        wave = jnp.linspace(4800, 5100, 1000)
        G_full = build_eline_design_matrix(wave, cat.wavelengths, 2000.0, 0.0)

        # Inject OIII at correct ratio
        true_amps = jnp.zeros(cat.n_lines)
        i_5007 = list(cat.names).index("OIII_5007")
        i_4959 = list(cat.names).index("OIII_4959")
        true_amps = true_amps.at[i_5007].set(10.0)
        true_amps = true_amps.at[i_4959].set(10.0 / 2.98)

        spectrum = G_full @ true_amps
        noise = jnp.full_like(spectrum, 0.01)
        G_eff = apply_doublet_constraints(G_full, C)
        prior_var = jnp.full(G_eff.shape[1], 1000.0**2)
        _, a_hat, a_cov = marginalize_emission_lines(
            spectrum, noise, G_eff, prior_variance=prior_var
        )
        a_full, _ = expand_constrained_amplitudes(a_hat, a_cov, C)
        ratio = float(a_full[i_5007]) / float(a_full[i_4959])
        assert abs(ratio - 2.98) < 0.01


class TestMarginalization:
    def test_recovers_injected_line(self):
        wave = jnp.linspace(6400, 6700, 500)
        ha = jnp.array([6564.61])  # vacuum Hα
        G = build_eline_design_matrix(wave, ha, 2000.0, 0.0)
        true_flux = jnp.array([5.0])
        continuum = jnp.ones_like(wave) * 10.0
        key = jax.random.PRNGKey(0)
        noise_arr = jnp.full_like(wave, 0.1)
        data = continuum + G @ true_flux + 0.01 * jax.random.normal(key, wave.shape)
        prior_var = jnp.full(1, 100.0**2)
        _, a_hat, _ = marginalize_emission_lines(
            data - continuum, noise_arr, G, prior_variance=prior_var
        )
        assert abs(float(a_hat[0]) - 5.0) < 0.5

    def test_gradient_through_marginalization(self):
        """Gradient of -ln_L w.r.t. continuum level must be finite and correctly signed.

        Physical check: marginalizing over a line amplitude, the continuum-level
        gradient must point toward the true continuum.
        - Under-subtracted (level < truth): residual inflated → increasing level
          reduces -ln_L → d(-ln_L)/d(level) < 0.
        - Over-subtracted (level > truth): residual negative → increasing level
          makes it worse → d(-ln_L)/d(level) > 0.
        """
        wave = jnp.linspace(6400, 6700, 200)
        ha = jnp.array([6564.61])  # vacuum Hα
        noise_arr = jnp.full_like(wave, 0.1)

        G = build_eline_design_matrix(wave, ha, 2000.0, 0.0)
        # Inject a true Hα signal on top of a flat continuum=1.0
        data = G @ jnp.array([5.0]) + 1.0

        def neg_log_like(level):
            residual = data - level * jnp.ones_like(wave)
            prior_var = jnp.full(1, 100.0**2)
            ln_l, _, _ = marginalize_emission_lines(
                residual, noise_arr, G, prior_variance=prior_var
            )
            return -ln_l

        # At correct continuum, gradient is near zero (minimum of -ln_L)
        g_at_true = jax.grad(neg_log_like)(1.0)
        assert jnp.isfinite(g_at_true), f"Non-finite gradient at true continuum: {g_at_true}"
        assert abs(float(g_at_true)) < 2.0, f"Gradient too large at true continuum: {g_at_true}"

        # Under-subtracted: d(-ln_L)/d(level) < 0
        g_low = jax.grad(neg_log_like)(0.0)
        assert jnp.isfinite(g_low), f"Non-finite gradient at under-subtracted level: {g_low}"
        assert float(g_low) < 0.0, (
            f"Expected negative gradient when continuum under-subtracted, got {float(g_low):.4f}"
        )

        # Over-subtracted: d(-ln_L)/d(level) > 0
        g_high = jax.grad(neg_log_like)(3.0)
        assert jnp.isfinite(g_high), f"Non-finite gradient at over-subtracted level: {g_high}"
        assert float(g_high) > 0.0, (
            f"Expected positive gradient when continuum over-subtracted, got {float(g_high):.4f}"
        )

    def test_jit_compiles(self):
        wave = jnp.linspace(4000, 7000, 500)
        lines = jnp.array([4862.68, 5008.24, 6564.61])  # vacuum Hβ, [OIII]5007, Hα

        @jax.jit
        def run(data, noise_arr):
            G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
            prior_var = jnp.full(3, 100.0**2)
            return marginalize_emission_lines(
                data - jnp.ones_like(data), noise_arr, G, prior_variance=prior_var
            )

        data = jnp.ones(500) * 10.0
        noise_arr = jnp.ones(500) * 0.1
        ln_l, a_hat, _a_cov = run(data, noise_arr)
        assert jnp.isfinite(ln_l)
        assert a_hat.shape == (3,)
