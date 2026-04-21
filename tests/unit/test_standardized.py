"""Tests for the standardized reparameterization bridge (inference/standardized.py).

StandardizedForwardModel absorbs all prior structure into coordinate transforms
so every sampler sees H(ξ) = ½χ² + ½ξᵀξ.  Tests here are deliberately
lightweight: they use mock forward models and simple Parameters specs so no
SSP data files are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ── Helpers: build minimal specs + mock models without SSP data ───


def _make_spec():
    """Build a minimal DPL Parameters spec with no SSP data needed."""
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Uniform

    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 3.0),
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
    def test_domain_keys_match_free_params(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        free = set(spec.free_params)
        domain_keys = set(smodel.domain.keys())
        # domain should contain exactly the free params (no stochastic field)
        assert free == domain_keys

    def test_domain_shapes_are_scalar(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        for name, shape in smodel.domain.items():
            assert shape == (), f"Expected scalar shape for {name}, got {shape}"

    def test_n_latent_equals_n_free_params(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        assert smodel.n_latent == len(spec.free_params)

    def test_stochastic_domain_includes_xi_field(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_stochastic_spec(n_grid=16)
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        assert "sfh_field_xi" in smodel.domain
        assert smodel.domain["sfh_field_xi"] == (16,)

    def test_stochastic_n_latent_includes_grid(self):
        from tengri.inference.standardized import StandardizedForwardModel

        n_grid = 16
        spec = _make_stochastic_spec(n_grid=n_grid)
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        # n_free_scalars + n_grid
        n_scalar_free = len(spec.free_params)
        assert smodel.n_latent == n_scalar_free + n_grid


# ── xi_to_params / params_to_xi roundtrip ─────────────────────────


class TestRoundtrip:
    def _make_zero_xi(self, smodel):
        """Build zero ξ dict matching domain."""
        return {
            name: jnp.zeros(shape) if shape != () else jnp.array(0.0)
            for name, shape in smodel.domain.items()
        }

    def test_xi_to_params_returns_dict(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = self._make_zero_xi(smodel)
        params = smodel.xi_to_params(xi)

        assert isinstance(params, dict)

    def test_xi_to_params_has_all_free_params(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = self._make_zero_xi(smodel)
        params = smodel.xi_to_params(xi)

        for name in spec.free_params:
            assert name in params, f"Missing free param '{name}' in output"

    def test_xi_to_params_values_are_finite(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = self._make_zero_xi(smodel)
        params = smodel.xi_to_params(xi)

        for name, val in params.items():
            arr = jnp.asarray(val)
            assert jnp.all(jnp.isfinite(arr)), f"Non-finite value for param '{name}': {val}"

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
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 3.0),
            met_logzsol=Fixed(-0.5),
        )
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        params = smodel.xi_to_params(xi)

        assert "met_logzsol" in params

    def test_params_to_xi_missing_key_defaults_to_zero(self):
        """params_to_xi should not crash on missing param — default ξ=0."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        # Provide an empty params dict
        xi_out = smodel.params_to_xi({})

        for name in spec.free_params:
            assert name in xi_out
            assert float(xi_out[name]) == 0.0


# ── build_standardized_loss — structure of the Hamiltonian ────────


class TestBuildStandardizedLoss:
    def _make_smodel_and_loss(self, n_filters=5):
        from tengri.inference.standardized import StandardizedForwardModel, build_standardized_loss

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=n_filters)
        smodel = StandardizedForwardModel(model)

        data = jnp.ones(n_filters) * 1e-18
        noise = jnp.ones(n_filters) * 1e-19

        loss_fn, unravel_fn = build_standardized_loss(smodel, data, noise)
        return smodel, loss_fn, unravel_fn

    def test_loss_returns_scalar(self):
        smodel, loss_fn, _ = self._make_smodel_and_loss()
        xi_flat = jnp.zeros(smodel.n_latent)
        val = loss_fn(xi_flat)
        assert val.shape == ()

    def test_loss_is_finite_at_zero(self):
        smodel, loss_fn, _ = self._make_smodel_and_loss()
        xi_flat = jnp.zeros(smodel.n_latent)
        val = loss_fn(xi_flat)
        assert jnp.isfinite(val)

    def test_loss_is_non_negative(self):
        smodel, loss_fn, _ = self._make_smodel_and_loss()
        xi_flat = jnp.zeros(smodel.n_latent)
        val = loss_fn(xi_flat)
        assert float(val) >= 0.0

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

    def test_gradient_is_finite(self):
        smodel, loss_fn, _ = self._make_smodel_and_loss()
        xi_flat = jnp.zeros(smodel.n_latent)
        grad = jax.grad(loss_fn)(xi_flat)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_shape_matches_latent(self):
        smodel, loss_fn, _ = self._make_smodel_and_loss()
        xi_flat = jnp.zeros(smodel.n_latent)
        grad = jax.grad(loss_fn)(xi_flat)
        assert grad.shape == (smodel.n_latent,)

    def test_unravel_fn_recovers_domain_keys(self):
        smodel, _, unravel_fn = self._make_smodel_and_loss()
        xi_flat = jnp.zeros(smodel.n_latent)
        xi_dict = unravel_fn(xi_flat)
        for name in smodel.domain:
            assert name in xi_dict

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
    def _make_galaxies(self, n_gal=3, n_filters=5):
        rng = np.random.default_rng(0)
        return [
            {
                "flux_obs": rng.uniform(1e-19, 1e-17, n_filters),
                "noise": np.ones(n_filters) * 1e-19,
            }
            for _ in range(n_gal)
        ]

    def test_hierarchical_loss_returns_scalar(self):
        from tengri.inference.standardized import StandardizedForwardModel, build_hierarchical_loss

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)
        galaxies = self._make_galaxies(n_gal=3)

        loss_fn, unravel_fn = build_hierarchical_loss(smodel, galaxies)
        n_dim = unravel_fn.__self__.n if hasattr(unravel_fn, "__self__") else None

        # Determine flat vector size from unravel_fn's template
        # Probe dimension by raveling unravel_fn on zero and checking shape
        # Build a flat zero vector of the right size via a test ravel
        xi_test = jnp.zeros(1)  # will fail — use loss gradient shape instead
        # Instead, just call with a zero vector of sufficient size and check finiteness
        # We can determine size from loss_fn's gradient
        n_total = smodel.n_latent * 3  # per-galaxy × n_gal (no shared in DPL)
        xi_flat = jnp.zeros(n_total)
        val = loss_fn(xi_flat)
        assert val.shape == ()
        assert jnp.isfinite(val)

    def test_hierarchical_loss_gradient_finite(self):
        from tengri.inference.standardized import StandardizedForwardModel, build_hierarchical_loss

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)
        galaxies = self._make_galaxies(n_gal=2)

        loss_fn, _ = build_hierarchical_loss(smodel, galaxies)
        n_total = smodel.n_latent * 2
        xi_flat = jnp.zeros(n_total)

        grad = jax.grad(loss_fn)(xi_flat)
        assert jnp.all(jnp.isfinite(grad))

    def test_hierarchical_shared_params_default_to_psd(self):
        """For stochastic SFH, shared_names defaults to PSD hyperparams."""
        from tengri.inference.standardized import StandardizedForwardModel, build_hierarchical_loss

        spec = _make_stochastic_spec(n_grid=8)
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        # Patch predict to avoid actual SSP computation
        model.predict_photometry.return_value = jnp.ones(5) * 1e-18

        galaxies = self._make_galaxies(n_gal=2)
        loss_fn, _unravel_fn = build_hierarchical_loss(smodel, galaxies)

        # PSD sigma and tau are shared — the loss should have fewer params
        # than n_gal * n_latent
        n_total_naive = smodel.n_latent * 2
        # Shared: psd_sigma + psd_tau (2 scalars). Per-galaxy: logzsol + n_grid xi field
        n_shared = 2  # sigma + tau_myr
        n_per_gal = smodel.n_latent - n_shared  # remaining scalars + xi field
        n_expected = n_shared + 2 * n_per_gal

        xi_flat = jnp.zeros(n_expected)
        val = loss_fn(xi_flat)
        assert jnp.isfinite(val)


# ── predict method ────────────────────────────────────────────────


class TestPredict:
    def test_predict_photometry_calls_model(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        result = smodel.predict(xi, data_type="photometry")

        model.predict_photometry.assert_called_once()
        assert result.shape == (5,)

    def test_call_shortcut_equals_predict_photometry(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        r1 = smodel(xi)
        r2 = smodel.predict(xi, data_type="photometry")

        np.testing.assert_array_equal(r1, r2)

    def test_predict_unknown_data_type_raises(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        with pytest.raises(ValueError, match="Unknown data_type"):
            smodel.predict(xi, data_type="hyperspectral")

    def test_predict_uses_traceable_mode(self):
        """predict() must call predict_photometry with mode='_traceable'."""
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_spec()
        model = _make_mock_model(spec, n_filters=5)
        smodel = StandardizedForwardModel(model)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        smodel.predict(xi, data_type="photometry")

        call_kwargs = model.predict_photometry.call_args
        assert call_kwargs.kwargs.get("mode") == "_traceable" or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "_traceable"
        )


# ── Custom PSD model injection ────────────────────────────────────


class TestCustomPSDModel:
    def test_custom_psd_model_is_called(self):
        from tengri.inference.standardized import StandardizedForwardModel

        spec = _make_stochastic_spec(n_grid=8)
        model = _make_mock_model(spec, n_filters=5)

        calls = []

        def custom_psd(sigma, tau_yr, n_grid, log_ages):
            calls.append((float(sigma), float(tau_yr)))
            # Return a plausible sqrt_power array
            from tengri.components.sfh.gp_sfh import compute_sqrt_power_drw

            return compute_sqrt_power_drw(sigma, tau_yr, n_grid, log_ages)

        smodel = StandardizedForwardModel(model, psd_model=custom_psd)

        xi = {name: jnp.array(0.0) for name in spec.free_params}
        xi["sfh_field_xi"] = jnp.zeros(8)
        smodel.xi_to_params(xi)

        assert len(calls) == 1, "Custom PSD model should have been called once"
