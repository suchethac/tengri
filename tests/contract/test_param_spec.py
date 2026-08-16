# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Parameters parameter specification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.contract

from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Gaussian, Uniform


class TestConstruction:
    def test_default_is_dpl_field(self):
        spec = Parameters()
        assert spec.mean_sfh_type == ["dpl", "field"]
        assert spec.stochastic is True

    def test_parametric_tsnorm(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            redshift=0.1,
        )
        assert not spec.stochastic
        assert "sfh_tsnorm_log_total_mass" in spec.free_params

    def test_basic_stochastic(self):
        spec = Parameters(
            mean_sfh_type=["tsnorm", "field"],
            sfh_field_psd_sigma=Uniform(0.1, 1.0),
            sfh_field_psd_tau_myr=Uniform(10.0, 300.0),
        )
        assert spec.stochastic
        assert spec.n_grid == 256

    def test_shorthand_scalar(self):
        spec = Parameters(dust_slope=-0.7)
        dist = spec.get_distribution("dust_slope")
        assert isinstance(dist, Fixed)
        assert dist.value == -0.7

    def test_shorthand_tuple(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=(0.01, 3.0),
        )
        dist = spec.get_distribution("sfh_tsnorm_log_total_mass")
        assert isinstance(dist, Uniform)
        assert dist.lo == 0.01
        assert dist.hi == 3.0

    def test_distribution_passthrough(self):
        g = Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2)
        spec = Parameters(met_logzsol=g)
        assert spec.get_distribution("met_logzsol") is g

    def test_defaults_applied(self):
        spec = Parameters()
        # All params should have defaults
        assert len(spec.all_params) > 0

    def test_invalid_param_name(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            Parameters(bad_name=1.0)

    def test_dpl_model(self):
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            redshift=0.1,
        )
        assert "sfh_dpl_alpha" in spec.free_params
        assert not spec.stochastic

    def test_dpl_rejects_tsnorm_params(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            Parameters(
                mean_sfh_type="dpl",
                sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            )

    def test_tsnorm_rejects_dpl_params(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            Parameters(
                mean_sfh_type="tsnorm",
                sfh_dpl_alpha=Uniform(0.5, 3.0),
            )

    def test_field_params_without_field_are_ignored(self):
        """Field params are silently dropped when field is not in the model.

        This enables backward compat where psd_sigma=Fixed(0) was
        passed alongside stochastic=False.
        """
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_field_psd_sigma=0.0,
        )
        assert "sfh_field_psd_sigma" not in spec.all_params


class TestLegacyCompat:
    def test_legacy_stochastic_true(self):
        spec = Parameters(stochastic=True)
        assert spec.stochastic
        assert "field" in spec.mean_sfh_type
        assert "dpl" in spec.mean_sfh_type

    def test_legacy_stochastic_false(self):
        spec = Parameters(stochastic=False)
        assert not spec.stochastic
        assert "field" not in spec.mean_sfh_type
        assert "dpl" in spec.mean_sfh_type

    def test_legacy_double_powerlaw(self):
        spec = Parameters(mean_sfh_type="double_powerlaw")
        assert spec.mean_sfh_type == ["dpl"]

    def test_legacy_param_aliases(self):
        """Old param names (sfh_alpha etc.) should resolve to new names."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=Uniform(0.3, 2.0),
        )
        assert "sfh_dpl_alpha" in spec.free_params
        assert "sfh_dpl_beta" in spec.free_params

    def test_dust_emission_name_aliases(self):
        """#849: the old Schreiber spellings ``dust_tdust`` / ``dust_fpah``
        resolve to the canonical ``dust_T`` / ``dust_f_pah`` (with a warning)."""
        with pytest.warns(DeprecationWarning):
            spec = Parameters(
                mean_sfh_type="const",
                dust_emission="schreiber2018",
                dust_tdust=Uniform(15.0, 99.0),
                dust_fpah=Uniform(0.0, 1.0),
            )
        assert "dust_T" in spec.free_params
        assert "dust_f_pah" in spec.free_params
        assert "dust_tdust" not in spec.free_params
        assert "dust_fpah" not in spec.free_params


class TestValidation:
    def test_dpl_alpha_must_be_positive(self):
        with pytest.raises(ValueError, match="lo > 0"):
            Parameters(mean_sfh_type="dpl", sfh_dpl_alpha=Uniform(-1.0, 5.0))

    def test_dust_tau_bc_nonnegative(self):
        with pytest.raises(ValueError, match="lo >= 0"):
            Parameters(dust_tau_bc=Uniform(-0.5, 4.0))

    def test_redshift_nonnegative(self):
        with pytest.raises(ValueError, match="lo >= 0"):
            Parameters(redshift=Uniform(-1.0, 6.0))

    def test_validate_good_params_does_not_raise(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        )
        # validate() raises on invalid params — no raise IS the assertion
        spec.validate({"sfh_tsnorm_log_total_mass": jnp.array(10.0)})

    def test_validate_out_of_bounds(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        )
        with pytest.raises(ValueError, match="outside bounds"):
            spec.validate({"sfh_tsnorm_log_total_mass": jnp.array(5.0)})


class TestProperties:
    def test_free_params_list(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            dust_slope=-0.7,
            redshift=0.1,
        )
        free = spec.free_params
        assert "sfh_tsnorm_log_total_mass" in free
        assert "dust_slope" not in free

    def test_fixed_params_list(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            dust_slope=-0.7,
        )
        fixed = spec.fixed_params
        assert "dust_slope" in fixed
        assert "sfh_tsnorm_log_total_mass" not in fixed

    def test_n_free(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            dust_slope=-0.7,
            redshift=0.1,
        )
        assert spec.n_free >= 1

    def test_get_fixed_values(self):
        spec = Parameters(dust_slope=-0.7, redshift=0.1)
        fixed = spec.get_fixed_values()
        assert fixed["dust_slope"] == -0.7
        assert fixed["redshift"] == 0.1

    def test_get_distribution(self):
        prior = Uniform(7.0, 12.5)
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=prior,
        )
        d = spec.get_distribution("sfh_tsnorm_log_total_mass")
        assert isinstance(d, Uniform)
        # Compare against the declared prior, not a repeated literal: the point
        # is that get_distribution round-trips what was set.
        assert d.lo == prior.lo

    def test_get_distribution_unknown(self):
        spec = Parameters()
        with pytest.raises(KeyError):
            spec.get_distribution("nonexistent")

    def test_valid_param_names_property(self):
        spec = Parameters(mean_sfh_type="tsnorm")
        assert "sfh_tsnorm_log_total_mass" in spec.valid_param_names
        assert "sfh_dpl_alpha" not in spec.valid_param_names

    def test_mean_sfh_type_property(self):
        spec = Parameters(mean_sfh_type=["tsnorm", "field"])
        assert spec.mean_sfh_type == ["tsnorm", "field"]


class TestSampling:
    def test_sample_all_keys_present(self):
        spec = Parameters(mean_sfh_type="tsnorm")
        params = spec.sample(jax.random.PRNGKey(0))
        for name in spec.valid_param_names:
            assert name in params, f"Missing key: {name}"

    def test_sample_stochastic_has_xi(self):
        spec = Parameters(mean_sfh_type=["tsnorm", "field"], n_grid=64)
        params = spec.sample(jax.random.PRNGKey(0))
        assert "sfh_field_xi" in params
        assert params["sfh_field_xi"].shape == (64,)

    def test_sample_parametric_no_xi(self):
        spec = Parameters(mean_sfh_type="tsnorm")
        params = spec.sample(jax.random.PRNGKey(0))
        assert "sfh_field_xi" not in params

    def test_sample_fixed_returns_value(self):
        spec = Parameters(dust_slope=-0.7)
        params = spec.sample(jax.random.PRNGKey(0))
        np.testing.assert_allclose(float(params["dust_slope"]), -0.7)

    def test_sample_free_in_bounds(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        )
        params = spec.sample(jax.random.PRNGKey(0))
        val = float(params["sfh_tsnorm_log_total_mass"])
        # Bounds read off the declaration, so this test tracks the prior
        # instead of pinning a range that a rename can silently invalidate.
        d = spec.get_distribution("sfh_tsnorm_log_total_mass")
        assert d.lo <= val <= d.hi

    def test_sample_batch_shapes(self):
        spec = Parameters(mean_sfh_type="tsnorm")
        batch = spec.sample_batch(jax.random.PRNGKey(0), 50)
        for name in spec.valid_param_names:
            assert batch[name].shape[0] == 50, f"{name} batch dim wrong"

    def test_sample_batch_stochastic_xi_shape(self):
        spec = Parameters(mean_sfh_type=["tsnorm", "field"], n_grid=32)
        batch = spec.sample_batch(jax.random.PRNGKey(0), 10)
        assert batch["sfh_field_xi"].shape == (10, 32)


class TestRepr:
    def test_repr_contains_params(self):
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            redshift=0.1,
        )
        r = repr(spec)
        assert "sfh_tsnorm_log_total_mass" in r
        assert "Uniform" in r
        assert "redshift" in r


class TestSampleSubstreamStability:
    """Per-parameter substream sampling (#548/#549 footgun regression).

    `Parameters.sample` derives each parameter's subkey from its name
    (`fold_in(key, crc32(name))`), not from its position in a key split.
    A shared free parameter must therefore sample to the same value for a
    given key regardless of which *other* parameters are free in the spec.
    """

    def _dpl(self, **extra):
        base = dict(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.1, 5.0),
            sfh_dpl_beta=Uniform(0.1, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.1, 12.0),
            sfh_dpl_age_gyr=Uniform(0.5, 13.5),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            redshift=Fixed(0.1),
        )
        base.update(extra)
        return Parameters(**base)

    def test_shared_param_invariant_to_free_set(self):
        """SFH params draw the same value whether or not extra params are free.

        This is the #548/#563 regression: spec_b enables ``dust_emission``,
        which adds free ``dust_T``/``dust_beta_ir``. Under positional key
        splitting this shifted the shared ``sfh_dpl_*`` draws (notably
        ``sfh_dpl_age_gyr``: 13.62 vs 13.08); per-name substreams keep them
        identical.
        """
        key = jax.random.PRNGKey(0)
        spec_a = self._dpl()
        spec_b = self._dpl(
            dust_emission="modified_blackbody",
            dust_T=Uniform(20.0, 60.0),
            dust_beta_ir=Uniform(1.0, 2.5),
        )
        a = spec_a.sample(key)
        b = spec_b.sample(key)
        # dust_emission adds free params to spec_b that spec_a lacks.
        assert set(spec_b.free_params) - set(spec_a.free_params), (
            "test setup: spec_b must add free params (dust_T/dust_beta_ir)"
        )
        # Every param free in BOTH must draw identically.
        shared_free = set(spec_a.free_params) & set(spec_b.free_params)
        assert "sfh_dpl_age_gyr" in shared_free, "age_gyr must be free in both"
        for name in shared_free:
            np.testing.assert_allclose(
                np.asarray(a[name]),
                np.asarray(b[name]),
                rtol=0,
                atol=0,
                err_msg=f"shared free param {name!r} diverged across differing free-sets",
            )

    def test_sample_is_reproducible_across_calls(self):
        """crc32 (not salted hash) → identical draws on repeated calls/processes."""
        spec = self._dpl()
        key = jax.random.PRNGKey(7)
        a = spec.sample(key)
        b = spec.sample(key)
        for name in spec.free_params:
            np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(b[name]))

    def test_distinct_params_get_distinct_draws(self):
        """Different names map to different substreams (no accidental aliasing)."""
        spec = self._dpl()
        s = spec.sample(jax.random.PRNGKey(0))
        vals = [float(s[n]) for n in ("sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr")]
        assert len(set(vals)) == len(vals), f"suspicious identical draws: {vals}"
