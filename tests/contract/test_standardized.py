# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for standardized reparameterization (inference/standardized.py).

StandardizedForwardModel absorbs all prior structure into coordinate transforms
so every sampler sees H(ξ) = ½χ² + ½ξᵀξ.  Tests verify domain shape contracts,
roundtrip invertibility, loss structure, and consistency with mock models.
No SSP data required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


# ── Helpers: build minimal specs + mock models without SSP data ───


def _make_spec():
    """Build a minimal DPL Parameters spec with no SSP data needed."""
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Uniform

    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
        met_logzsol=Uniform(-2.0, 0.5),
    )


def _make_stochastic_spec(n_grid=16):
    """Build a minimal stochastic-field Parameters spec (DPL smooth + field)."""
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Uniform

    return Parameters(
        mean_sfh_type=["dpl", "field"],
        n_grid=n_grid,
        sfh_field_psd_sigma=Uniform(0.1, 5.0),
        sfh_field_psd_tau_myr=Uniform(10.0, 500.0),
        met_logzsol=Uniform(-2.0, 0.5),
    )


def _make_mock_model(spec, n_filters=5):
    """Build a mock SEDModel that has spec + predict_photometry returning random flux."""
    model = MagicMock()
    model.spec = spec
    model.predict_photometry.return_value = jnp.ones(n_filters) * 1e-18
    model.predict_spectrum.return_value = jnp.ones(100) * 1e-18
    return model


# ── Domain and n_latent ───────────────────────────────────────────


class TestDomain:
    """Domain structure: keys match free params, shapes are scalar or grid."""

    def test_domain_keys_match_free_params(self):
        """Domain keys are exactly the free parameters (no stochastic field)."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        free = set(spec.free_params)
        domain_keys = set(smodel.domain.keys())
        assert free == domain_keys

    def test_n_latent_equals_n_free_params(self):
        """n_latent is the count of scalar free parameters."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        assert smodel.n_latent == len(spec.free_params)

    def test_stochastic_domain_includes_field_grid(self):
        """Stochastic SFH adds sfh_field_xi with shape (n_grid,)."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_stochastic_spec(n_grid=16)
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        assert "sfh_field_xi" in smodel.domain
        assert smodel.domain["sfh_field_xi"] == (16,)

    def test_stochastic_n_latent_includes_grid(self):
        """n_latent for stochastic includes scalar params + grid size."""
        from tengri.inference.standardized import StandardizedForwardModel

        n_grid = 16
        spec = _make_stochastic_spec(n_grid=n_grid)
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        n_scalar_free = len(spec.free_params)
        assert smodel.n_latent == n_scalar_free + n_grid


# ── xi_to_params / params_to_xi roundtrip ─────────────────────────


class TestRoundtrip:
    """Roundtrip invertibility: params_to_xi ∘ xi_to_params ≈ identity."""

    def _make_zero_xi(self, smodel):
        """Build zero ξ dict matching domain."""
        return {
            name: jnp.zeros(shape) if shape != () else jnp.array(0.0)
            for name, shape in smodel.domain.items()
        }

    def test_params_to_xi_roundtrip(self):
        """params_to_xi(xi_to_params(ξ)) ≈ ξ for all free scalar params."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        # Use non-zero xi to exercise the transforms
        rng = np.random.default_rng(42)
        xi_in = {name: jnp.array(rng.standard_normal()) for name in spec.free_params}

        params = smodel.xi_to_params(xi_in)
        xi_out = smodel.params_to_xi(params)

        for name in spec.free_params:
            np.testing.assert_allclose(
                float(xi_out[name]),
                float(xi_in[name]),
                rtol=1e-5,
                err_msg=f"Roundtrip failed for param '{name}'",
            )

    def test_fixed_params_present_in_output(self):
        """Fixed parameters must appear in xi_to_params output."""
        from tengri.inference.standardized import StandardizedForwardModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed, Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 5.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
            met_logzsol=Fixed(-0.5),
        )
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        params = smodel.xi_to_params(xi)

        assert "met_logzsol" in params


# ── build_standardized_loss — structure of the Hamiltonian ────────


class TestBuildStandardizedLoss:
    """Loss structure: H(ξ) = ½χ² + ½ξᵀξ at perfect fit."""

    def _make_smodel_and_loss(self, n_filters=5):
        from tengri.inference.standardized import StandardizedForwardModel, build_standardized_loss

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=n_filters)
        smodel = StandardizedForwardModel(model)

        data = jnp.ones(n_filters) * 1e-18
        noise = jnp.ones(n_filters) * 1e-19

        loss_fn, unravel_fn = build_standardized_loss(smodel, data, noise)
        return smodel, loss_fn, unravel_fn

    def test_prior_term_is_half_norm_squared(self):
        """At perfect fit (predicted == data), loss = ½ξᵀξ."""
        from tengri.inference.standardized import StandardizedForwardModel, build_standardized_loss

        n_filters = 5
        spec = _make_spec()
        # Predicted always == data: residual = 0, loss = ½ξᵀξ
        target_flux = jnp.ones(n_filters) * 1e-18
        model = MagicMock()
        model.spec = spec
        model.predict_photometry.return_value = target_flux

        smodel = StandardizedForwardModel(model)
        data = target_flux
        noise = jnp.ones(n_filters) * 1e-19

        loss_fn, _ = build_standardized_loss(smodel, data, noise)

        xi_flat = jnp.ones(smodel.n_latent) * 2.0
        expected_prior = 0.5 * float(jnp.sum(xi_flat**2))
        actual = float(loss_fn(xi_flat))

        np.testing.assert_allclose(actual, expected_prior, rtol=1e-6)

    def test_loss_increases_with_larger_residuals(self):
        """Bigger data-model mismatch → larger loss."""
        from tengri.inference.standardized import StandardizedForwardModel, build_standardized_loss

        n_filters = 5
        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=n_filters)

        smodel = StandardizedForwardModel(model)
        xi_flat = jnp.zeros(smodel.n_latent)

        # Small noise → large chi2
        noise_small = jnp.ones(n_filters) * 1e-20
        noise_large = jnp.ones(n_filters) * 1e-16
        data = jnp.ones(n_filters) * 5e-18  # offset from predicted 1e-18

        loss_small, _ = build_standardized_loss(smodel, data, noise_small)
        loss_large, _ = build_standardized_loss(smodel, data, noise_large)

        assert float(loss_small(xi_flat)) > float(loss_large(xi_flat))


# ── build_hierarchical_loss ───────────────────────────────────────


class TestBuildHierarchicalLoss:
    """Hierarchical loss: per-galaxy params + shared PSD hyperparams."""

    def _make_galaxies(self, n_gal=3, n_filters=5):
        rng = np.random.default_rng(0)
        return [
            {
                "flux_obs": rng.uniform(1e-19, 1e-17, n_filters),
                "noise": np.ones(n_filters) * 1e-19,
            }
            for _ in range(n_gal)
        ]

    def test_hierarchical_loss_is_finite(self):
        """Hierarchical loss returns finite scalar at zero ξ."""
        from tengri.inference.standardized import StandardizedForwardModel, build_hierarchical_loss

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)
        galaxies = self._make_galaxies(n_gal=3)

        loss_fn, _ = build_hierarchical_loss(smodel, galaxies)
        n_total = smodel.n_latent * 3  # per-galaxy × n_gal (no shared in DPL)
        xi_flat = jnp.zeros(n_total)

        val = loss_fn(xi_flat)
        assert jnp.isfinite(val)
        assert val.shape == ()

    def test_hierarchical_shared_params_default_to_psd(self):
        """For stochastic SFH, shared_names defaults to PSD hyperparams."""
        from tengri.inference.standardized import StandardizedForwardModel, build_hierarchical_loss

        spec = _make_stochastic_spec(n_grid=8)
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        model.predict_photometry.return_value = jnp.ones(5) * 1e-18

        galaxies = self._make_galaxies(n_gal=2)
        loss_fn, _unravel_fn = build_hierarchical_loss(smodel, galaxies)

        # PSD sigma and tau are shared — the loss should have fewer params
        # than n_gal * n_latent
        n_shared = 2  # sigma + tau_myr
        n_per_gal = smodel.n_latent - n_shared  # remaining scalars + xi field
        n_expected = n_shared + 2 * n_per_gal

        xi_flat = jnp.zeros(n_expected)
        val = loss_fn(xi_flat)
        assert jnp.isfinite(val)


# ── predict method ────────────────────────────────────────────────


class TestPredict:
    """Predict surface: calls through to model and validates interface."""

    def test_predict_unknown_data_type_raises(self):
        """Predict with unknown data_type raises ValueError."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        with pytest.raises(ValueError, match="Unknown data_type"):
            smodel.predict(xi, data_type="hyperspectral")

    def test_call_shortcut_equals_predict_photometry(self):
        """Calling smodel(xi) is equivalent to predict(xi, data_type='photometry')."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        r1 = smodel(xi)
        r2 = smodel.predict(xi, data_type="photometry")

        np.testing.assert_array_equal(r1, r2)

    def test_predict_does_not_pass_legacy_mode_kwarg(self):
        """Predict must not pass the removed mode= kwarg to model.predict_photometry.

        Phase 6-prep (2026-05-20) collapsed the kernel-cascade modes (auto/exact/
        hybrid/compositional/traced) into a single delegate through
        predict_observables_jit. The mode= kwarg was subsequently removed; the
        JIT-safe property is now structurally enforced. This test guards against
        any future code that re-introduces a non-orchestrator call.
        """
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        smodel.predict(xi, data_type="photometry")

        call_kwargs = model.predict_photometry.call_args
        assert "mode" not in call_kwargs.kwargs


# ── Custom PSD model injection ────────────────────────────────────


class TestCustomPSDModel:
    """Custom PSD function can be injected and is called during xi_to_params."""

    def test_custom_psd_model_is_called(self):
        """Custom psd_model callable is invoked during parameter transform."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_stochastic_spec(n_grid=8)
        model = _make_mock_model(spec, n_filters=5)

        calls = []

        def custom_psd(sigma, tau_yr, n_grid, log_ages):
            calls.append((float(sigma), float(tau_yr)))
            # Return a plausible sqrt_power array
            from tengri.components.stellar.sfh.gp_sfh import compute_sqrt_power_drw

            return compute_sqrt_power_drw(sigma, tau_yr, n_grid, log_ages)

        smodel = StandardizedForwardModel(model, psd_model=custom_psd)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        xi["sfh_field_xi"] = jnp.zeros(8)
        smodel.xi_to_params(xi)

        assert len(calls) == 1, "Custom PSD model should have been called once"
