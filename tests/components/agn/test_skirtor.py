# SPDX-License-Identifier: BSD-3-Clause
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
- NPZ loading path in create_skirtor_from_grid
- FileNotFoundError path in skirtor_analytic
- Missing required keys error in create_skirtor_from_grid
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tests._bounds import assert_non_negative


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestSKIRTORRegistration:
    """Test that SKIRTOR is properly registered."""

    def test_skirtor_in_agn_models(self):
        import warnings

        from tengri.components.agn import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = resolve_agn_model("skirtor")
        assert callable(fn)

    def test_get_agn_model_skirtor(self):
        from tengri.components.agn import resolve_agn_model

        model_fn = resolve_agn_model("skirtor")
        assert callable(model_fn)

    def test_skirtor_analytic_importable(self):
        from tengri.components.agn import skirtor_analytic

        assert callable(skirtor_analytic)

    def test_create_skirtor_from_grid_importable(self):
        from tengri.components.agn import create_skirtor_from_grid

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

    def test_shape_finite_positive_and_golden_values(self, wave):
        """Shape, finiteness, positivity, and frozen golden values.

        Golden values frozen from the current implementation (test-audit PR, 2026-07).
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sed = skirtor_analytic(wave)

        # Shape, finiteness, and positivity
        chex.assert_equal_shape([sed, wave])
        chex.assert_tree_all_finite(sed)
        assert jnp.all(sed >= 0), "SKIRTOR SED should be non-negative"
        assert float(jnp.max(sed)) > 0, "SKIRTOR SED should have positive values"

        # Frozen golden values at indices [0, n//3, 2n//3, -1].
        # Re-frozen after the standalone default agn_log_lbol was corrected from
        # the non-physical 44.0 (old log10 erg/s value) to 10.0 (log10 L_bol/Lsun,
        # matching the canonical param default) — a pure 1e-34 rescale of the SED
        # (l_scale is linear in 10**agn_log_lbol). The values are now physical
        # (~1e29 erg/s/Hz for a 1e10 Lsun AGN).
        #
        # Re-frozen again when template wavelength resampling moved from linear
        # (``jnp.interp``) to log lambda + log flux (``resample_template``). Only
        # the two mid-interval samples move, and they move DOWN, which is the
        # expected sign: a straight chord across a convex log-log curve sits
        # above it, so the old values were biased high.
        #
        #   idx 166 (0.4606 um, 0.389 native cells from a node): -6.24 %
        #   idx 333 (21.71 um,  0.068 native cells from a node): -0.069 %
        #   idx 499 (1000 um,   0.000 cells — exactly ON a node): unchanged to 1e-15
        #   idx 0   (0.01 um, below template support):            0.0, unchanged
        #
        # The node-exact sample not moving at all is the load-bearing check here:
        # it shows this is an interpolation change between nodes, not a rescale.
        indices = [0, len(wave) // 3, 2 * len(wave) // 3, -1]
        golden_values = [0.0, 5.982589e24, 3.523055e29, 3.607845e24]
        for idx, golden in zip(indices, golden_values):
            np.testing.assert_allclose(
                float(sed[idx]),
                golden,
                rtol=1e-6,
                atol=1e-70,  # Allow for zero values
                err_msg=f"Golden value mismatch at index {idx}",
            )

    def test_type1_vs_type2_spectral_shape(self, ir_wave):
        """Face-on (Type 1) and edge-on (Type 2) SEDs should differ meaningfully.
        Key physical differences:
        - Different silicate feature behavior (emission vs absorption)
        - Different component weighting (hot dust visibility)
        """
        from tengri.components.agn.skirtor import skirtor_analytic

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
        from tengri.components.agn.skirtor import skirtor_analytic

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
        from tengri.components.agn.skirtor import skirtor_analytic

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
        """Different tau values should produce different SEDs.

        Physical: Optical depth controls dust absorption and re-emission.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sed_low = skirtor_analytic(wave, agn_tau_skirtor=3.0)
            sed_high = skirtor_analytic(wave, agn_tau_skirtor=11.0)
        int_low = float(np.trapezoid(np.array(sed_low), np.array(wave)))
        int_high = float(np.trapezoid(np.array(sed_high), np.array(wave)))
        if int_low > 0:
            frac_change = abs(int_high - int_low) / int_low
            assert frac_change > 0.01, (
                f"tau should produce >1% change, got {frac_change * 100:.2f}%"
            )

    def test_p_affects_sed(self, wave):
        """Different p values should produce different SEDs.

        Physical: Radial dust distribution gradient affects torus geometry and emission.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sed_low = skirtor_analytic(wave, agn_p_skirtor=0.0)
            sed_high = skirtor_analytic(wave, agn_p_skirtor=1.5)
        int_low = float(np.trapezoid(np.array(sed_low), np.array(wave)))
        int_high = float(np.trapezoid(np.array(sed_high), np.array(wave)))
        if int_low > 0:
            frac_change = abs(int_high - int_low) / int_low
            assert frac_change > 0.01, f"p should produce >1% change, got {frac_change * 100:.2f}%"

    def test_q_affects_sed(self, wave):
        """Different q values should produce different SEDs.

        Physical: Angular dust gradient affects SED shape and normalization.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sed_low = skirtor_analytic(wave, agn_q_skirtor=0.0)
            sed_high = skirtor_analytic(wave, agn_q_skirtor=1.5)
        int_low = float(np.trapezoid(np.array(sed_low), np.array(wave)))
        int_high = float(np.trapezoid(np.array(sed_high), np.array(wave)))
        if int_low > 0:
            frac_change = abs(int_high - int_low) / int_low
            assert frac_change > 0.01, f"q should produce >1% change, got {frac_change * 100:.2f}%"

    def test_oa_affects_sed(self, wave):
        """Different opening angles should produce different SEDs.

        Physical: Torus opening angle controls the geometry of obscuration/emission.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sed_narrow = skirtor_analytic(wave, agn_oa_skirtor=20.0)
            sed_wide = skirtor_analytic(wave, agn_oa_skirtor=60.0)
        int_narrow = float(np.trapezoid(np.array(sed_narrow), np.array(wave)))
        int_wide = float(np.trapezoid(np.array(sed_wide), np.array(wave)))
        if int_narrow > 0:
            frac_change = abs(int_wide - int_narrow) / int_narrow
            assert frac_change > 0.01, (
                f"oa should produce >1% change, got {frac_change * 100:.2f}%"
            )

    def test_cos_inc_affects_sed(self, wave):
        """Different inclinations should produce different SEDs.

        Physical: Viewing angle controls visibility of hot/cool dust and silicate features.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sed_face = skirtor_analytic(wave, agn_cos_inc=0.95)
            sed_edge = skirtor_analytic(wave, agn_cos_inc=0.05)
        int_face = float(np.trapezoid(np.array(sed_face), np.array(wave)))
        int_edge = float(np.trapezoid(np.array(sed_edge), np.array(wave)))
        if int_face > 0:
            frac_change = abs(int_edge - int_face) / int_face
            assert frac_change > 0.01, f"cos_inc >1% change: {frac_change * 100:.2f}%"

    def test_luminosity_scaling(self, wave):
        """SED should scale with bolometric luminosity."""
        from tengri.components.agn.skirtor import skirtor_analytic

        sed_low = skirtor_analytic(wave, agn_log_lbol=43.0)
        sed_high = skirtor_analytic(wave, agn_log_lbol=44.0)
        ratio = float(jnp.max(sed_high) / jnp.max(sed_low))
        np.testing.assert_allclose(ratio, 10.0, rtol=0.1)

    def test_torus_frac_scaling(self, wave):
        """Torus emission should scale with covering fraction."""
        from tengri.components.agn.skirtor import skirtor_analytic

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
        """FD check: ∂(∑SED)/∂tau_skirtor. Optical depth gradient."""
        from tengri.components.agn.skirtor import skirtor_analytic

        def f(tau):
            return float(jnp.sum(skirtor_analytic(wave, agn_tau_skirtor=tau)))

        g = float(jax.grad(lambda tau: jnp.sum(skirtor_analytic(wave, agn_tau_skirtor=tau)))(7.0))
        np.testing.assert_allclose(
            g,
            fd_grad(f, 7.0, eps=0.1),
            rtol=1e-3,
            err_msg="skirtor_analytic: FD check ∂/∂tau_skirtor",
        )

    def test_gradient_p(self, wave):
        """FD check: ∂(∑SED)/∂p_skirtor. Radial dust gradient."""
        from tengri.components.agn.skirtor import skirtor_analytic

        def f(p):
            return float(jnp.sum(skirtor_analytic(wave, agn_p_skirtor=p)))

        g = float(jax.grad(lambda p: jnp.sum(skirtor_analytic(wave, agn_p_skirtor=p)))(1.0))
        np.testing.assert_allclose(
            g,
            fd_grad(f, 1.0),
            rtol=1e-3,
            err_msg="skirtor_analytic: FD check ∂/∂p_skirtor",
        )

    def test_gradient_q(self, wave):
        """FD check: ∂(∑SED)/∂q_skirtor. Angular dust gradient."""
        from tengri.components.agn.skirtor import skirtor_analytic

        def f(q):
            return float(jnp.sum(skirtor_analytic(wave, agn_q_skirtor=q)))

        g = float(jax.grad(lambda q: jnp.sum(skirtor_analytic(wave, agn_q_skirtor=q)))(1.0))
        np.testing.assert_allclose(
            g,
            fd_grad(f, 1.0),
            rtol=1e-3,
            err_msg="skirtor_analytic: FD check ∂/∂q_skirtor",
        )

    def test_gradient_oa(self, wave):
        """FD check: ∂(∑SED)/∂oa_skirtor. Opening angle gradient."""
        from tengri.components.agn.skirtor import skirtor_analytic

        def f(oa):
            return float(jnp.sum(skirtor_analytic(wave, agn_oa_skirtor=oa)))

        g = float(jax.grad(lambda oa: jnp.sum(skirtor_analytic(wave, agn_oa_skirtor=oa)))(40.0))
        np.testing.assert_allclose(
            g,
            fd_grad(f, 40.0, eps=0.1),
            rtol=1e-3,
            err_msg="skirtor_analytic: FD check ∂/∂oa_skirtor",
        )

    def test_gradient_cos_inc(self, wave):
        """FD check: ∂(∑SED)/∂cos_inc. Inclination gradient."""
        from tengri.components.agn.skirtor import skirtor_analytic

        def f(ci):
            return float(jnp.sum(skirtor_analytic(wave, agn_cos_inc=ci)))

        g = float(jax.grad(lambda ci: jnp.sum(skirtor_analytic(wave, agn_cos_inc=ci)))(0.5))
        np.testing.assert_allclose(
            g,
            fd_grad(f, 0.5),
            rtol=1e-3,
            err_msg="skirtor_analytic: FD check ∂/∂cos_inc",
        )

    def test_gradient_all_params_simultaneously(self, wave):
        """All 5 SKIRTOR gradients agree with FD simultaneously."""
        from tengri.components.agn.skirtor import skirtor_analytic

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
        params = [
            ("tau", 7.0, 0.1),
            ("p", 1.0, 1e-4),
            ("q", 1.0, 1e-4),
            ("oa", 40.0, 0.1),
            ("cos_inc", 0.5, 1e-4),
        ]
        fs = [
            lambda tau: float(loss(tau, 1.0, 1.0, 40.0, 0.5)),
            lambda p: float(loss(7.0, p, 1.0, 40.0, 0.5)),
            lambda q: float(loss(7.0, 1.0, q, 40.0, 0.5)),
            lambda oa: float(loss(7.0, 1.0, 1.0, oa, 0.5)),
            lambda ci: float(loss(7.0, 1.0, 1.0, 40.0, ci)),
        ]
        for i, (name, x0, eps) in enumerate(params):
            np.testing.assert_allclose(
                float(grads[i]),
                fd_grad(fs[i], x0, eps=eps),
                rtol=1e-3,
                err_msg=f"skirtor_analytic joint: FD check ∂/∂{name}",
            )

    def test_gradient_registered_model(self, wave):
        """Gradient should flow through the registered 'skirtor' model."""
        import warnings

        from tengri.components.agn import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model_fn = resolve_agn_model("skirtor")

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

        def f_tau(tau: float) -> float:
            return float(loss(tau, 0.5))

        def f_ci(ci: float) -> float:
            return float(loss(7.0, ci))

        np.testing.assert_allclose(
            float(g_tau),
            fd_grad(f_tau, 7.0),
            rtol=5e-3,
            err_msg="skirtor (registry): FD check ∂/∂agn_tau_skirtor",
        )
        np.testing.assert_allclose(
            float(g_ci),
            fd_grad(f_ci, 0.5),
            rtol=5e-3,
            err_msg="skirtor (registry): FD check ∂/∂agn_cos_inc",
        )


class TestSKIRTORJIT:
    """Test JIT compilation compatibility."""

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 200)

    def test_jit_parity_analytic(self, wave):
        """JIT eager output match (parity) and output shape for skirtor_analytic.

        Physical: JIT compilation should preserve numerical outputs.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            def test_fn(tau, ci):
                return skirtor_analytic(wave, agn_tau_skirtor=tau, agn_cos_inc=ci)

            sed_eager = test_fn(7.0, 0.5)
            fn_jit = jax.jit(test_fn)
            sed_jit = fn_jit(7.0, 0.5)
            chex.assert_equal_shape([sed_jit, wave])
            # JIT parity to 1 ppm of the SED's own scale. This is a torus-only
            # template: the SED is ~0 across the UV/optical (dust emits nothing
            # there) and peaks in the IR. A bare rtol=1e-6, atol=0 would demand
            # bit-exact eager/jit agreement on that near-zero noise floor —
            # values ~40 dex below the IR peak where XLA legitimately reorders
            # sub-ULP terms differently across platforms (a false failure seen
            # only on CI-Linux, #1195). Scaling atol to the peak asserts the
            # physically meaningful statement: JIT preserves the SED.
            atol = 1e-6 * float(jnp.max(jnp.abs(sed_eager)))
            chex.assert_trees_all_close(sed_eager, sed_jit, rtol=1e-6, atol=atol)

    def test_jit_parity_registered_model(self, wave):
        """JIT parity for registered 'skirtor' model with all 5 parameters."""
        import warnings

        from tengri.components.agn import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model_fn = resolve_agn_model("skirtor")

            def test_fn(tau, p, q, oa, ci):
                return model_fn(
                    wave,
                    agn_log_lbol=44.0,
                    agn_tau_skirtor=tau,
                    agn_p_skirtor=p,
                    agn_q_skirtor=q,
                    agn_oa_skirtor=oa,
                    agn_cos_inc=ci,
                )

            sed_eager = test_fn(7.0, 1.0, 1.0, 40.0, 0.5)
            fn_jit = jax.jit(test_fn)
            sed_jit = fn_jit(7.0, 1.0, 1.0, 40.0, 0.5)
            chex.assert_equal_shape([sed_jit, wave])
            # JIT parity to 1 ppm of the SED's own scale (see the note in
            # test_jit_parity_analytic): atol=0 would demand bit-exact agreement
            # on the near-zero noise floor, which XLA does not guarantee across
            # platforms for fused float ops (#1195).
            atol = 1e-6 * float(jnp.max(jnp.abs(sed_eager)))
            chex.assert_trees_all_close(sed_eager, sed_jit, rtol=1e-6, atol=atol)

    def test_jit_grad_combined(self, wave):
        """JIT of grad agrees with FD for agn_tau_skirtor."""
        from tengri.components.agn.skirtor import skirtor_analytic

        fn = jax.jit(jax.grad(lambda tau: jnp.sum(skirtor_analytic(wave, agn_tau_skirtor=tau))))
        grad_jax = float(fn(7.0))

        def f_scalar(tau: float) -> float:
            return float(jnp.sum(skirtor_analytic(wave, agn_tau_skirtor=tau)))

        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f_scalar, 7.0),
            rtol=5e-3,
            err_msg="skirtor_analytic: FD check ∂(∑SED)/∂agn_tau_skirtor",
        )


class TestSKIRTOREdgeCases:
    """Test edge cases and boundary parameter values.

    Collapsed test: all boundary parameter combinations produce finite SEDs with positive max.
    """

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 200)

    def test_all_boundary_params_produce_finite_nonzero_sed(self, wave):
        """All boundary parameter combinations produce finite positive SEDs.

        Physical: Even at extreme torus geometries (thin tau, wide opening angle, edge-on viewing,
        etc.), the model must produce physically valid (finite, non-negative) spectra.
        """
        import warnings

        from tengri.components.agn.skirtor import skirtor_analytic

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            test_cases = [
                {"agn_tau_skirtor": 3.0},  # Low optical depth
                {"agn_tau_skirtor": 11.0},  # High optical depth
                {"agn_cos_inc": 1.0},  # Face-on
                {"agn_cos_inc": 0.0},  # Edge-on
                {"agn_oa_skirtor": 20.0},  # Narrow opening angle
                {"agn_oa_skirtor": 60.0},  # Wide opening angle
                {"agn_p_skirtor": 0.0, "agn_q_skirtor": 0.0},  # No radial/angular gradients
                {"agn_p_skirtor": 1.5, "agn_q_skirtor": 1.5},  # Max gradients
            ]

            for kw in test_cases:
                sed = skirtor_analytic(wave, **kw)
                assert jnp.all(jnp.isfinite(sed)), f"Non-finite SED for params {kw}"
                assert float(jnp.max(sed)) > 0.0, f"Max SED should be positive for params {kw}"


class TestNovikovThorneEfficiency:
    """Spin-dependent radiative efficiency from Novikov & Thorne (1973).
    The accretion efficiency η = 1 - sqrt(1 - 2/(3 r_isco)) is the fraction of
    rest-mass energy radiated by matter falling from the ISCO to the event horizon
    (Novikov & Thorne 1973, Black Holes, Les Houches; eq. 5.6.23).
    r_isco shrinks with increasing prograde spin, raising η.  Reference values:
    - Schwarzschild (a=0):   r_isco = 6 R_g, η ≈ 0.0572  (Bardeen+1972)
    - Near-maximal (a=0.998): η ≈ 0.321                  (Thorne 1974)
    """

    def test_schwarzschild_isco(self):
        """At a=0 (Schwarzschild BH): r_isco = 6 R_g exactly (Bardeen+1972)."""
        from tengri.components.agn.disc import _isco_radius

        r_isco_0 = float(_isco_radius(0.0))
        np.testing.assert_allclose(
            r_isco_0,
            6.0,
            rtol=1e-5,
            err_msg="Schwarzschild ISCO: r_isco = 6 R_g (Bardeen, Press & Teukolsky 1972)",
        )

    def test_schwarzschild_efficiency(self):
        """At a=0: η = 1 - sqrt(8/9) ≈ 0.0572."""
        from tengri.components.agn.disc import _isco_radius

        r_isco_0 = float(_isco_radius(0.0))
        eta_0 = 1.0 - float(jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_0)))
        np.testing.assert_allclose(
            eta_0,
            0.0572,
            rtol=0.01,
            err_msg="Novikov-Thorne: η(a=0) ≈ 0.0572 (Schwarzschild)",
        )

    def test_maximal_spin_efficiency(self):
        """At a=0.998 (near-maximal prograde spin): η ≈ 0.321 (Thorne 1974 limit)."""
        from tengri.components.agn.disc import _isco_radius

        r_isco_998 = float(_isco_radius(0.998))
        eta_998 = 1.0 - float(jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_998)))
        np.testing.assert_allclose(
            eta_998,
            0.321,
            rtol=0.05,
            err_msg="Novikov-Thorne: η(a=0.998) ≈ 0.321 (Thorne 1974)",
        )

    def test_higher_spin_higher_efficiency(self):
        """Monotonicity: higher prograde spin → smaller ISCO → higher η."""
        from tengri.components.agn.disc import _isco_radius

        def eta_from_spin(a: float) -> float:
            r = float(_isco_radius(a))
            return 1.0 - float(jnp.sqrt(1.0 - 2.0 / (3.0 * r)))

        r_0 = float(_isco_radius(0.0))
        r_5 = float(_isco_radius(0.5))
        r_9 = float(_isco_radius(0.9))
        assert r_0 > r_5 > r_9, (
            f"ISCO must shrink with prograde spin: "
            f"r(0)={r_0:.4f}, r(0.5)={r_5:.4f}, r(0.9)={r_9:.4f}"
        )
        eta_0 = eta_from_spin(0.0)
        eta_5 = eta_from_spin(0.5)
        eta_9 = eta_from_spin(0.9)
        assert eta_9 > eta_5 > eta_0, (
            f"Efficiency must grow with prograde spin: "
            f"η(0)={eta_0:.4f}, η(0.5)={eta_5:.4f}, η(0.9)={eta_9:.4f}"
        )


class TestCreateSkirtorFromGridNpz:
    """Tests for the NPZ loading path in create_skirtor_from_grid.
    The default skirtor_analytic auto-loads an HDF5 file (skirtor_templates_v2.h5)
    which is always found first in the candidate list, so the NPZ code path
    (create_skirtor_from_grid with a .npz suffix) is never exercised by the
    existing physics tests.  These tests target that branch directly.
    """

    @pytest.fixture
    def mock_npz(self, tmp_path):
        """Create a minimal valid SKIRTOR .npz file for testing."""
        rng = np.random.default_rng(42)
        n_tau, n_p, n_q, n_oa, n_inc, n_wave = 3, 3, 3, 3, 3, 50
        wave = np.geomspace(1e3, 1e7, n_wave)
        # Positive grid values so energy integral > 0
        grid = rng.uniform(0.1, 1.0, (n_tau, n_p, n_q, n_oa, n_inc, n_wave))
        data = {
            "wavelength": wave,
            "tau": np.array([3.0, 7.0, 11.0]),
            "p": np.array([0.0, 1.0, 2.0]),
            "q": np.array([0.0, 1.0, 2.0]),
            "oa": np.array([20.0, 40.0, 60.0]),
            "cos_inc": np.array([0.1, 0.5, 0.9]),
            "grid": grid,
        }
        path = tmp_path / "skirtor_test.npz"
        np.savez(str(path), **data)
        return str(path)

    def test_npz_returns_callable(self, mock_npz):
        """create_skirtor_from_grid('.npz') must return a callable."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(mock_npz)
        assert callable(fn), "NPZ loader must return a callable interpolator"

    def test_npz_output_shape(self, mock_npz):
        """Returned callable must produce correct output shape."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(mock_npz)
        wave = jnp.logspace(4.0, 6.0, 80)
        out = fn(wave, agn_log_lbol=12.0)
        assert out.shape == wave.shape, (
            f"NPZ path: output shape {out.shape} does not match wave shape {wave.shape}"
        )

    def test_npz_output_finite(self, mock_npz):
        """NPZ-loaded interpolator must produce finite output."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(mock_npz)
        wave = jnp.logspace(4.0, 6.0, 80)
        out = fn(wave, agn_log_lbol=12.0)
        chex.assert_tree_all_finite(out)

    def test_npz_output_non_negative(self, mock_npz):
        """NPZ-loaded interpolator must produce non-negative luminosities."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(mock_npz)
        wave = jnp.logspace(4.0, 6.0, 80)
        out = fn(wave, agn_log_lbol=12.0)
        assert_non_negative(out, name="out", msg="NPZ path: negative L_nu values")

    def test_npz_missing_key_raises(self, tmp_path):
        """create_skirtor_from_grid must raise KeyError when NPZ lacks required keys."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        # Intentionally omit 'cos_inc' and 'oa'
        rng = np.random.default_rng(0)
        n_wave = 20
        incomplete = {
            "wavelength": np.geomspace(1e3, 1e7, n_wave),
            "tau": np.array([3.0, 7.0]),
            "p": np.array([0.0, 1.0]),
            "q": np.array([0.0, 1.0]),
            "grid": rng.uniform(0.1, 1.0, (2, 2, 2, 2, 2, n_wave)),
            # 'oa' and 'cos_inc' intentionally missing
        }
        bad_path = tmp_path / "incomplete.npz"
        np.savez(str(bad_path), **incomplete)
        with pytest.raises(KeyError, match="missing keys"):
            create_skirtor_from_grid(str(bad_path))

    def test_npz_gradient_flows(self, mock_npz):
        """JAX autodiff must flow through the NPZ-loaded interpolator."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        fn = create_skirtor_from_grid(mock_npz)
        wave = jnp.logspace(4.0, 6.0, 40)
        g_jax = float(jax.grad(lambda lbol: jnp.sum(fn(wave, agn_log_lbol=lbol)))(12.0))
        g_fd = fd_grad(lambda lbol: float(jnp.sum(fn(wave, agn_log_lbol=lbol))), 12.0)
        assert np.isfinite(g_jax), "NPZ path: autodiff gradient is not finite"
        np.testing.assert_allclose(
            g_jax, g_fd, rtol=1e-2, err_msg="NPZ path: autodiff gradient differs from FD by >1%"
        )


class TestSkirtorAnalyticErrorPaths:
    """Tests for the FileNotFoundError branch in skirtor_analytic.
    The lazy-init singleton (_skirtor_default) is reset to None and all
    candidate template paths are patched away so the 'not found' branch
    fires without requiring removal of real data files.
    """

    def test_file_not_found_raises(self, monkeypatch, tmp_path):
        """skirtor_analytic must raise FileNotFoundError when no template file exists."""
        import tengri.components.agn.skirtor as _mod

        # Clear functools.cache so the loaders re-run on next call. skirtor_sed
        # loads the threadable grid via _load_skirtor_default_grid (#1178); the
        # legacy closure loader _load_skirtor_default is also cleared.
        _mod._load_skirtor_default.cache_clear()
        _mod._load_skirtor_default_grid.cache_clear()
        from pathlib import Path

        def fake_is_file(self):
            return False

        monkeypatch.setattr(Path, "is_file", fake_is_file)
        try:
            with pytest.raises(FileNotFoundError, match="SKIRTOR templates not found"):
                _mod.skirtor_analytic(jnp.logspace(3, 6, 50), agn_log_lbol=12.0)
        finally:
            # Clear again so the error result is not cached for subsequent tests
            _mod._load_skirtor_default.cache_clear()
            _mod._load_skirtor_default_grid.cache_clear()
