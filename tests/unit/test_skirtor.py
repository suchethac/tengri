"""Tests for the SKIRTOR clumpy torus model (Stalevski+2012, 2016).

Validates:
- Registration in AGN_MODELS
- Output shape, positivity, finiteness
- Type 1 vs Type 2 SED differences
- Silicate 9.7 um feature (absorption edge-on, emission face-on)
- Parameter sensitivity (all 5 SKIRTOR params affect the SED)
- Gradient flow through all 5 parameters
- JIT compatibility
- Energy conservation (torus_frac scaling)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


class TestSKIRTORRegistration:
    """Test that SKIRTOR is properly registered."""

    def test_skirtor_in_agn_models(self):
        from tengri.models.agn import AGN_MODELS

        assert "skirtor" in AGN_MODELS

    def test_get_agn_model_skirtor(self):
        from tengri.models.agn import resolve_agn_model

        model_fn = resolve_agn_model("skirtor")
        assert callable(model_fn)

    def test_skirtor_analytic_importable(self):
        from tengri.models.agn import skirtor_analytic

        assert callable(skirtor_analytic)

    def test_create_skirtor_from_grid_importable(self):
        from tengri.models.agn import create_skirtor_from_grid

        assert callable(create_skirtor_from_grid)


class TestSKIRTORAnalytic:
    """Test the analytic SKIRTOR torus model."""

    @pytest.fixture
    def wave(self):
        """Wavelength grid spanning UV through FIR (100 A to 10000 um)."""
        return jnp.logspace(2, 7, 500)

    @pytest.fixture
    def ir_wave(self):
        """IR wavelength grid (1 um to 1000 um)."""
        return jnp.logspace(4, 7, 300)

    def test_output_shape(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave)
        assert sed.shape == wave.shape

    def test_output_positive(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave)
        assert jnp.all(sed >= 0), "SKIRTOR SED should be non-negative"
        assert float(jnp.max(sed)) > 0, "SKIRTOR SED should have positive values"

    def test_output_finite(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave)
        assert jnp.all(jnp.isfinite(sed)), "SKIRTOR SED should be finite everywhere"

    def test_type1_vs_type2_spectral_shape(self, ir_wave):
        """Face-on (Type 1) and edge-on (Type 2) SEDs should differ meaningfully.

        Key physical differences:
        - Different silicate feature behavior (emission vs absorption)
        - Different component weighting (hot dust visibility)
        """
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_type1 = skirtor_analytic(ir_wave, agn_cos_inc=0.95)  # face-on
        sed_type2 = skirtor_analytic(ir_wave, agn_cos_inc=0.05)  # edge-on

        # Normalized SEDs should be meaningfully different
        norm1 = sed_type1 / jnp.max(sed_type1)
        norm2 = sed_type2 / jnp.max(sed_type2)
        max_diff = float(jnp.max(jnp.abs(norm1 - norm2)))
        assert max_diff > 0.05, (
            f"Type 1 and Type 2 normalized SEDs should differ (max diff = {max_diff:.4f})"
        )

        # The silicate region (around 9.7 um = 97000 A) should differ
        # between Type 1 and Type 2
        idx_97 = jnp.argmin(jnp.abs(ir_wave - 97000.0))
        sil_ratio = float(norm1[idx_97] / jnp.maximum(norm2[idx_97], 1e-30))
        assert sil_ratio != 1.0, "Silicate feature should differ between Type 1 and Type 2"

    def test_silicate_absorption_edge_on(self, ir_wave):
        """Edge-on viewing should show silicate absorption at 9.7 um."""
        from tengri.models.agn.skirtor import skirtor_analytic

        # High tau, edge-on
        sed = skirtor_analytic(
            ir_wave,
            agn_tau_skirtor=11.0,
            agn_cos_inc=0.05,
        )

        # Find the 9.7 um region (97000 A)
        idx_97 = jnp.argmin(jnp.abs(ir_wave - 97000.0))
        # Find continuum points on either side (~7 um and ~13 um)
        idx_7 = jnp.argmin(jnp.abs(ir_wave - 70000.0))
        idx_13 = jnp.argmin(jnp.abs(ir_wave - 130000.0))

        # Interpolate continuum at 9.7 um
        continuum_at_97 = 0.5 * (float(sed[idx_7]) + float(sed[idx_13]))
        flux_at_97 = float(sed[idx_97])

        # Silicate absorption: flux at 9.7 um should be below interpolated continuum
        assert flux_at_97 < continuum_at_97, (
            f"Silicate absorption expected: flux at 9.7um ({flux_at_97:.2e}) "
            f"should be below continuum ({continuum_at_97:.2e})"
        )

    def test_silicate_emission_face_on(self, ir_wave):
        """Face-on viewing should show silicate emission (or at least no deep absorption)."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(
            ir_wave,
            agn_tau_skirtor=7.0,
            agn_cos_inc=0.95,
        )

        idx_97 = jnp.argmin(jnp.abs(ir_wave - 97000.0))
        idx_7 = jnp.argmin(jnp.abs(ir_wave - 70000.0))
        idx_13 = jnp.argmin(jnp.abs(ir_wave - 130000.0))

        continuum_at_97 = 0.5 * (float(sed[idx_7]) + float(sed[idx_13]))
        flux_at_97 = float(sed[idx_97])

        # Face-on: silicate feature should be in emission (above continuum)
        # or at minimum not deeply absorbed
        assert flux_at_97 >= 0.8 * continuum_at_97, (
            f"Face-on should not show deep silicate absorption: "
            f"flux ({flux_at_97:.2e}) vs continuum ({continuum_at_97:.2e})"
        )

    def test_tau_affects_sed(self, wave):
        """Different tau values should produce different SEDs."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_low = skirtor_analytic(wave, agn_tau_skirtor=3.0)
        sed_high = skirtor_analytic(wave, agn_tau_skirtor=11.0)
        assert not jnp.allclose(sed_low, sed_high, atol=1e-20), "tau should affect the SED"

    def test_p_affects_sed(self, wave):
        """Different p values should produce different SEDs."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_low = skirtor_analytic(wave, agn_p_skirtor=0.0)
        sed_high = skirtor_analytic(wave, agn_p_skirtor=1.5)
        assert not jnp.allclose(sed_low, sed_high, atol=1e-20), "p should affect the SED"

    def test_q_affects_sed(self, wave):
        """Different q values should produce different SEDs."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_low = skirtor_analytic(wave, agn_q_skirtor=0.0)
        sed_high = skirtor_analytic(wave, agn_q_skirtor=1.5)
        assert not jnp.allclose(sed_low, sed_high, atol=1e-20), "q should affect the SED"

    def test_oa_affects_sed(self, wave):
        """Different opening angles should produce different SEDs."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_narrow = skirtor_analytic(wave, agn_oa_skirtor=20.0)
        sed_wide = skirtor_analytic(wave, agn_oa_skirtor=60.0)
        assert not jnp.allclose(sed_narrow, sed_wide, atol=1e-20), (
            "opening angle should affect the SED"
        )

    def test_cos_inc_affects_sed(self, wave):
        """Different inclinations should produce different SEDs."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_face = skirtor_analytic(wave, agn_cos_inc=0.95)
        sed_edge = skirtor_analytic(wave, agn_cos_inc=0.05)
        assert not jnp.allclose(sed_face, sed_edge, atol=1e-20), (
            "inclination should affect the SED"
        )

    def test_luminosity_scaling(self, wave):
        """SED should scale with bolometric luminosity."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_low = skirtor_analytic(wave, agn_log_lbol=43.0)
        sed_high = skirtor_analytic(wave, agn_log_lbol=44.0)
        ratio = float(jnp.max(sed_high) / jnp.max(sed_low))
        np.testing.assert_allclose(ratio, 10.0, rtol=0.1)

    def test_torus_frac_scaling(self, wave):
        """Torus emission should scale with covering fraction."""
        from tengri.models.agn.skirtor import skirtor_analytic

        sed_low = skirtor_analytic(wave, agn_torus_frac=0.2)
        sed_high = skirtor_analytic(wave, agn_torus_frac=0.8)
        ratio = float(jnp.max(sed_high) / jnp.max(sed_low))
        np.testing.assert_allclose(ratio, 4.0, rtol=0.15)


class TestSKIRTORGradients:
    """Test that gradients flow through all 5 SKIRTOR parameters."""

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 300)

    def test_gradient_tau(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        g = jax.grad(lambda tau: jnp.sum(skirtor_analytic(wave, agn_tau_skirtor=tau)))(7.0)
        assert jnp.isfinite(g), "tau gradient should be finite"
        assert abs(float(g)) > 0, "tau gradient should be nonzero"

    def test_gradient_p(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        g = jax.grad(lambda p: jnp.sum(skirtor_analytic(wave, agn_p_skirtor=p)))(1.0)
        assert jnp.isfinite(g), "p gradient should be finite"
        assert abs(float(g)) > 0, "p gradient should be nonzero"

    def test_gradient_q(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        g = jax.grad(lambda q: jnp.sum(skirtor_analytic(wave, agn_q_skirtor=q)))(1.0)
        assert jnp.isfinite(g), "q gradient should be finite"
        assert abs(float(g)) > 0, "q gradient should be nonzero"

    def test_gradient_oa(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        g = jax.grad(lambda oa: jnp.sum(skirtor_analytic(wave, agn_oa_skirtor=oa)))(40.0)
        assert jnp.isfinite(g), "oa gradient should be finite"
        assert abs(float(g)) > 0, "oa gradient should be nonzero"

    def test_gradient_cos_inc(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        g = jax.grad(lambda ci: jnp.sum(skirtor_analytic(wave, agn_cos_inc=ci)))(0.5)
        assert jnp.isfinite(g), "cos_inc gradient should be finite"
        assert abs(float(g)) > 0, "cos_inc gradient should be nonzero"

    def test_gradient_all_params_simultaneously(self, wave):
        """All 5 SKIRTOR gradients should be finite and nonzero simultaneously."""
        from tengri.models.agn.skirtor import skirtor_analytic

        def loss(tau, p, q, oa, ci):
            return jnp.sum(
                skirtor_analytic(
                    wave,
                    agn_tau_skirtor=tau,
                    agn_p_skirtor=p,
                    agn_q_skirtor=q,
                    agn_oa_skirtor=oa,
                    agn_cos_inc=ci,
                )
            )

        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(7.0, 1.0, 1.0, 40.0, 0.5)
        names = ["tau", "p", "q", "oa", "cos_inc"]
        for i, name in enumerate(names):
            assert jnp.isfinite(grads[i]), f"{name} gradient is not finite"
            assert abs(float(grads[i])) > 0, f"{name} gradient is zero"

    def test_gradient_registered_model(self, wave):
        """Gradient should flow through the registered 'skirtor' model."""
        from tengri.models.agn import AGN_MODELS

        model_fn = AGN_MODELS["skirtor"]

        def loss(tau, ci):
            return jnp.sum(
                model_fn(
                    wave,
                    agn_log_lbol=44.0,
                    agn_tau_skirtor=tau,
                    agn_cos_inc=ci,
                )
            )

        g_tau, g_ci = jax.grad(loss, argnums=(0, 1))(7.0, 0.5)
        assert jnp.isfinite(g_tau) and abs(float(g_tau)) > 0
        assert jnp.isfinite(g_ci) and abs(float(g_ci)) > 0


class TestSKIRTORJIT:
    """Test JIT compilation compatibility."""

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 200)

    def test_jit_analytic(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        fn = jax.jit(lambda tau, ci: skirtor_analytic(wave, agn_tau_skirtor=tau, agn_cos_inc=ci))
        sed = fn(7.0, 0.5)
        assert sed.shape == wave.shape
        assert jnp.all(jnp.isfinite(sed))

    def test_jit_registered_model(self, wave):
        from tengri.models.agn import AGN_MODELS

        model_fn = AGN_MODELS["skirtor"]
        fn = jax.jit(
            lambda tau, p, q, oa, ci: model_fn(
                wave,
                agn_tau_skirtor=tau,
                agn_p_skirtor=p,
                agn_q_skirtor=q,
                agn_oa_skirtor=oa,
                agn_cos_inc=ci,
            )
        )
        sed = fn(7.0, 1.0, 1.0, 40.0, 0.5)
        assert sed.shape == wave.shape
        assert jnp.all(jnp.isfinite(sed))

    def test_jit_grad_combined(self, wave):
        """JIT of grad should work."""
        from tengri.models.agn.skirtor import skirtor_analytic

        fn = jax.jit(jax.grad(lambda tau: jnp.sum(skirtor_analytic(wave, agn_tau_skirtor=tau))))
        g = fn(7.0)
        assert jnp.isfinite(g)


class TestSKIRTOREdgeCases:
    """Test edge cases and boundary parameter values."""

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 200)

    def test_extreme_tau_low(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_tau_skirtor=3.0)
        assert jnp.all(jnp.isfinite(sed))

    def test_extreme_tau_high(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_tau_skirtor=11.0)
        assert jnp.all(jnp.isfinite(sed))

    def test_fully_face_on(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_cos_inc=1.0)
        assert jnp.all(jnp.isfinite(sed))
        assert float(jnp.max(sed)) > 0

    def test_fully_edge_on(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_cos_inc=0.0)
        assert jnp.all(jnp.isfinite(sed))
        assert float(jnp.max(sed)) > 0

    def test_narrow_opening_angle(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_oa_skirtor=20.0)
        assert jnp.all(jnp.isfinite(sed))

    def test_wide_opening_angle(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_oa_skirtor=60.0)
        assert jnp.all(jnp.isfinite(sed))

    def test_zero_p_and_q(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_p_skirtor=0.0, agn_q_skirtor=0.0)
        assert jnp.all(jnp.isfinite(sed))

    def test_max_p_and_q(self, wave):
        from tengri.models.agn.skirtor import skirtor_analytic

        sed = skirtor_analytic(wave, agn_p_skirtor=1.5, agn_q_skirtor=1.5)
        assert jnp.all(jnp.isfinite(sed))
