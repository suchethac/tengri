"""Tests for ParamSpec parameter specification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.distributions import Fixed, Gaussian, Uniform
from diffsed.param_spec import VALID_PARAM_NAMES, ParamSpec


class TestConstruction:
    def test_basic_parametric(self):
        spec = ParamSpec(
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=1.0,
            redshift=0.1,
            stochastic=False,
        )
        assert not spec.stochastic
        assert "sfh_alpha" in spec.free_params
        assert "sfh_beta" in spec.fixed_params

    def test_basic_stochastic(self):
        spec = ParamSpec(
            psd_sigma=Uniform(0.1, 3.0),
            psd_tau_myr=Uniform(1.0, 300.0),
            stochastic=True,
            n_grid=128,
        )
        assert spec.stochastic
        assert spec.n_grid == 128

    def test_shorthand_scalar(self):
        spec = ParamSpec(dust_slope=-0.7)
        dist = spec.get_distribution("dust_slope")
        assert isinstance(dist, Fixed)
        assert dist.value == -0.7

    def test_shorthand_tuple(self):
        spec = ParamSpec(sfh_peak_sfr=(0.01, 200.0))
        dist = spec.get_distribution("sfh_peak_sfr")
        assert isinstance(dist, Uniform)
        assert dist.lo == 0.01
        assert dist.hi == 200.0

    def test_distribution_passthrough(self):
        g = Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2)
        spec = ParamSpec(met_logzsol=g)
        assert spec.get_distribution("met_logzsol") is g

    def test_defaults_applied(self):
        spec = ParamSpec()
        # All params should have defaults
        assert len(spec.all_params) == len(VALID_PARAM_NAMES)

    def test_invalid_param_name(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParamSpec(bad_name=1.0)


class TestValidation:
    def test_sfh_alpha_must_be_positive(self):
        with pytest.raises(ValueError, match="lo > 0"):
            ParamSpec(sfh_alpha=Uniform(-1.0, 5.0))

    def test_dust_tau_bc_nonnegative(self):
        with pytest.raises(ValueError, match="lo >= 0"):
            ParamSpec(dust_tau_bc=Uniform(-0.5, 4.0))

    def test_redshift_nonnegative(self):
        with pytest.raises(ValueError, match="lo >= 0"):
            ParamSpec(redshift=Uniform(-1.0, 6.0))

    def test_stochastic_without_psd_uses_defaults(self):
        # Default psd_sigma=Fixed(0.0), psd_tau_myr=Fixed(50.0) — should be fine
        spec = ParamSpec(stochastic=True)
        assert spec.stochastic

    def test_validate_good_params(self):
        spec = ParamSpec(sfh_alpha=Uniform(0.5, 3.0))
        spec.validate({"sfh_alpha": jnp.array(1.5)})

    def test_validate_out_of_bounds(self):
        spec = ParamSpec(sfh_alpha=Uniform(0.5, 3.0))
        with pytest.raises(ValueError, match="outside bounds"):
            spec.validate({"sfh_alpha": jnp.array(5.0)})


class TestProperties:
    def test_free_params_list(self):
        spec = ParamSpec(
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=1.0,
            dust_slope=-0.7,
            redshift=0.1,
        )
        free = spec.free_params
        assert "sfh_alpha" in free
        assert "sfh_beta" not in free

    def test_fixed_params_list(self):
        spec = ParamSpec(
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=1.0,
        )
        fixed = spec.fixed_params
        assert "sfh_beta" in fixed
        assert "sfh_alpha" not in fixed

    def test_n_free(self):
        spec = ParamSpec(
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=Uniform(0.3, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )
        # sfh_alpha, sfh_beta are explicitly free
        # defaults: sfh_tau_peak_gyr, sfh_peak_sfr, met_logzsol,
        # dust_tau_bc, dust_tau_diff are Uniform (free)
        # psd_sigma, psd_tau_myr are Fixed by default
        # dust_slope, redshift are Fixed
        assert spec.n_free >= 2  # at least the two we set

    def test_all_params_complete(self):
        spec = ParamSpec()
        assert set(spec.all_params) == VALID_PARAM_NAMES

    def test_get_fixed_values(self):
        spec = ParamSpec(dust_slope=-0.7, redshift=0.1)
        fixed = spec.get_fixed_values()
        assert fixed["dust_slope"] == -0.7
        assert fixed["redshift"] == 0.1

    def test_get_distribution(self):
        spec = ParamSpec(sfh_alpha=Uniform(0.5, 3.0))
        d = spec.get_distribution("sfh_alpha")
        assert isinstance(d, Uniform)
        assert d.lo == 0.5

    def test_get_distribution_unknown(self):
        spec = ParamSpec()
        with pytest.raises(KeyError):
            spec.get_distribution("nonexistent")


class TestSampling:
    def test_sample_all_keys_present(self):
        spec = ParamSpec(stochastic=False)
        params = spec.sample(jax.random.PRNGKey(0))
        for name in VALID_PARAM_NAMES:
            assert name in params, f"Missing key: {name}"

    def test_sample_stochastic_has_xi(self):
        spec = ParamSpec(stochastic=True, n_grid=64)
        params = spec.sample(jax.random.PRNGKey(0))
        assert "psd_xi" in params
        assert params["psd_xi"].shape == (64,)

    def test_sample_parametric_no_xi(self):
        spec = ParamSpec(stochastic=False)
        params = spec.sample(jax.random.PRNGKey(0))
        assert "psd_xi" not in params

    def test_sample_fixed_returns_value(self):
        spec = ParamSpec(dust_slope=-0.7)
        params = spec.sample(jax.random.PRNGKey(0))
        np.testing.assert_allclose(float(params["dust_slope"]), -0.7)

    def test_sample_free_in_bounds(self):
        spec = ParamSpec(sfh_alpha=Uniform(0.5, 3.0))
        params = spec.sample(jax.random.PRNGKey(0))
        val = float(params["sfh_alpha"])
        assert 0.5 <= val <= 3.0

    def test_sample_batch_shapes(self):
        spec = ParamSpec(stochastic=False)
        batch = spec.sample_batch(jax.random.PRNGKey(0), 50)
        for name in VALID_PARAM_NAMES:
            assert batch[name].shape[0] == 50, f"{name} batch dim wrong"

    def test_sample_batch_stochastic_xi_shape(self):
        spec = ParamSpec(stochastic=True, n_grid=32)
        batch = spec.sample_batch(jax.random.PRNGKey(0), 10)
        assert batch["psd_xi"].shape == (10, 32)


class TestRepr:
    def test_repr_contains_params(self):
        spec = ParamSpec(sfh_alpha=Uniform(0.5, 3.0), redshift=0.1)
        r = repr(spec)
        assert "sfh_alpha" in r
        assert "Uniform" in r
        assert "redshift" in r
