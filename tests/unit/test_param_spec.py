"""Tests for ParamSpec parameter specification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.distributions import Fixed, Gaussian, Uniform
from diffsed.param_spec import ParamSpec


class TestConstruction:
    def test_default_is_dpl_field(self):
        spec = ParamSpec()
        assert spec.mean_sfh_type == ["dpl", "field"]
        assert spec.stochastic is True

    def test_parametric_tsnorm(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
            redshift=0.1,
        )
        assert not spec.stochastic
        assert "sfh_tsnorm_log_peak_sfr" in spec.free_params

    def test_basic_stochastic(self):
        spec = ParamSpec(
            mean_sfh_type=["tsnorm", "field"],
            sfh_field_psd_sigma=Uniform(0.1, 1.0),
            sfh_field_psd_tau_myr=Uniform(10.0, 300.0),
        )
        assert spec.stochastic
        assert spec.n_grid == 64

    def test_shorthand_scalar(self):
        spec = ParamSpec(dust_slope=-0.7)
        dist = spec.get_distribution("dust_slope")
        assert isinstance(dist, Fixed)
        assert dist.value == -0.7

    def test_shorthand_tuple(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=(0.01, 3.0),
        )
        dist = spec.get_distribution("sfh_tsnorm_log_peak_sfr")
        assert isinstance(dist, Uniform)
        assert dist.lo == 0.01
        assert dist.hi == 3.0

    def test_distribution_passthrough(self):
        g = Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2)
        spec = ParamSpec(met_logzsol=g)
        assert spec.get_distribution("met_logzsol") is g

    def test_defaults_applied(self):
        spec = ParamSpec()
        # All params should have defaults
        assert len(spec.all_params) > 0

    def test_invalid_param_name(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParamSpec(bad_name=1.0)

    def test_dpl_model(self):
        spec = ParamSpec(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_peak_sfr=Uniform(-1, 2),
            redshift=0.1,
        )
        assert "sfh_dpl_alpha" in spec.free_params
        assert not spec.stochastic

    def test_dpl_rejects_tsnorm_params(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParamSpec(
                mean_sfh_type="dpl",
                sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
            )

    def test_tsnorm_rejects_dpl_params(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParamSpec(
                mean_sfh_type="tsnorm",
                sfh_dpl_alpha=Uniform(0.5, 3.0),
            )

    def test_field_params_without_field_are_ignored(self):
        """Field params are silently dropped when field is not in the model.

        This enables backward compat where psd_sigma=Fixed(0) was
        passed alongside stochastic=False.
        """
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_field_psd_sigma=0.0,
        )
        assert "sfh_field_psd_sigma" not in spec.all_params


class TestLegacyCompat:
    def test_legacy_stochastic_true(self):
        spec = ParamSpec(stochastic=True)
        assert spec.stochastic
        assert "field" in spec.mean_sfh_type
        assert "dpl" in spec.mean_sfh_type

    def test_legacy_stochastic_false(self):
        spec = ParamSpec(stochastic=False)
        assert not spec.stochastic
        assert "field" not in spec.mean_sfh_type
        assert "dpl" in spec.mean_sfh_type

    def test_legacy_double_powerlaw(self):
        spec = ParamSpec(mean_sfh_type="double_powerlaw")
        assert spec.mean_sfh_type == ["dpl"]

    def test_legacy_param_aliases(self):
        """Old param names (sfh_alpha etc.) should resolve to new names."""
        spec = ParamSpec(
            mean_sfh_type="dpl",
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=Uniform(0.3, 2.0),
        )
        assert "sfh_dpl_alpha" in spec.free_params
        assert "sfh_dpl_beta" in spec.free_params


class TestValidation:
    def test_dpl_alpha_must_be_positive(self):
        with pytest.raises(ValueError, match="lo > 0"):
            ParamSpec(mean_sfh_type="dpl", sfh_dpl_alpha=Uniform(-1.0, 5.0))

    def test_dust_tau_bc_nonnegative(self):
        with pytest.raises(ValueError, match="lo >= 0"):
            ParamSpec(dust_tau_bc=Uniform(-0.5, 4.0))

    def test_redshift_nonnegative(self):
        with pytest.raises(ValueError, match="lo >= 0"):
            ParamSpec(redshift=Uniform(-1.0, 6.0))

    def test_validate_good_params(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        )
        spec.validate({"sfh_tsnorm_log_peak_sfr": jnp.array(0.5)})

    def test_validate_out_of_bounds(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        )
        with pytest.raises(ValueError, match="outside bounds"):
            spec.validate({"sfh_tsnorm_log_peak_sfr": jnp.array(5.0)})


class TestProperties:
    def test_free_params_list(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 3.0),
            dust_slope=-0.7,
            redshift=0.1,
        )
        free = spec.free_params
        assert "sfh_tsnorm_log_peak_sfr" in free
        assert "dust_slope" not in free

    def test_fixed_params_list(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
            dust_slope=-0.7,
        )
        fixed = spec.fixed_params
        assert "dust_slope" in fixed
        assert "sfh_tsnorm_log_peak_sfr" not in fixed

    def test_n_free(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
            dust_slope=-0.7,
            redshift=0.1,
        )
        assert spec.n_free >= 1

    def test_get_fixed_values(self):
        spec = ParamSpec(dust_slope=-0.7, redshift=0.1)
        fixed = spec.get_fixed_values()
        assert fixed["dust_slope"] == -0.7
        assert fixed["redshift"] == 0.1

    def test_get_distribution(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        )
        d = spec.get_distribution("sfh_tsnorm_log_peak_sfr")
        assert isinstance(d, Uniform)
        assert d.lo == -1

    def test_get_distribution_unknown(self):
        spec = ParamSpec()
        with pytest.raises(KeyError):
            spec.get_distribution("nonexistent")

    def test_valid_param_names_property(self):
        spec = ParamSpec(mean_sfh_type="tsnorm")
        assert "sfh_tsnorm_log_peak_sfr" in spec.valid_param_names
        assert "sfh_dpl_alpha" not in spec.valid_param_names

    def test_mean_sfh_type_property(self):
        spec = ParamSpec(mean_sfh_type=["tsnorm", "field"])
        assert spec.mean_sfh_type == ["tsnorm", "field"]


class TestSampling:
    def test_sample_all_keys_present(self):
        spec = ParamSpec(mean_sfh_type="tsnorm")
        params = spec.sample(jax.random.PRNGKey(0))
        for name in spec.valid_param_names:
            assert name in params, f"Missing key: {name}"

    def test_sample_stochastic_has_xi(self):
        spec = ParamSpec(mean_sfh_type=["tsnorm", "field"], n_grid=64)
        params = spec.sample(jax.random.PRNGKey(0))
        assert "sfh_field_xi" in params
        assert params["sfh_field_xi"].shape == (64,)

    def test_sample_parametric_no_xi(self):
        spec = ParamSpec(mean_sfh_type="tsnorm")
        params = spec.sample(jax.random.PRNGKey(0))
        assert "sfh_field_xi" not in params

    def test_sample_fixed_returns_value(self):
        spec = ParamSpec(dust_slope=-0.7)
        params = spec.sample(jax.random.PRNGKey(0))
        np.testing.assert_allclose(float(params["dust_slope"]), -0.7)

    def test_sample_free_in_bounds(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        )
        params = spec.sample(jax.random.PRNGKey(0))
        val = float(params["sfh_tsnorm_log_peak_sfr"])
        assert -1 <= val <= 2

    def test_sample_batch_shapes(self):
        spec = ParamSpec(mean_sfh_type="tsnorm")
        batch = spec.sample_batch(jax.random.PRNGKey(0), 50)
        for name in spec.valid_param_names:
            assert batch[name].shape[0] == 50, f"{name} batch dim wrong"

    def test_sample_batch_stochastic_xi_shape(self):
        spec = ParamSpec(mean_sfh_type=["tsnorm", "field"], n_grid=32)
        batch = spec.sample_batch(jax.random.PRNGKey(0), 10)
        assert batch["sfh_field_xi"].shape == (10, 32)


class TestRepr:
    def test_repr_contains_params(self):
        spec = ParamSpec(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
            redshift=0.1,
        )
        r = repr(spec)
        assert "sfh_tsnorm_log_peak_sfr" in r
        assert "Uniform" in r
        assert "redshift" in r
