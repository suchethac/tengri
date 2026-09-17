# SPDX-License-Identifier: BSD-3-Clause
r"""Posterior.to_param_spec and validate read the posterior, not a hardcoded default (#2180).

Two methods on Posterior consulted a hardcoded default instead of self:

1. ``to_param_spec()`` copied only ``stochastic`` and ``n_grid`` from the model spec,
   so ``Parameters(**kwargs)`` re-defaulted ``mean_sfh_type=['dpl']`` and rejected the
   posterior's own ``sfh_delayed_*`` names at line 717 in parameters.py.

2. ``validate()`` forwarded ``n_steps=n_steps`` into ``self._fitter.run("mcmc_nuts", ...)``
   regardless of the method that produced the posterior, so after a MAP fit it failed on
   NUTS's signature.

The root cause is reading a fixed default where the posterior already knows the answer.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestPosteriorReadsItself:
    """Posterior.to_param_spec and validate must consult self, not hardcoded defaults."""

    @pytest.fixture
    def minimal_map_with_delayed_sfh(self, ssp_data_wne, simple_observation):
        """Minimal real MAP fit with delayed SFH type."""
        from tengri import SEDModel, Fixed, DEFAULT
        from tengri.inference.fitter import Fitter

        # Build a model with delayed SFH (not the default dpl)
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=simple_observation,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )

        # Create minimal synthetic data and run a tiny MAP fit
        # simple_observation creates 3 bands
        n_bands = 3
        photometry = jnp.ones(n_bands)
        noise = jnp.ones(n_bands) * 0.01
        fitter = Fitter(model, data=photometry, noise=noise)
        post = fitter.run("map", n_steps=2)
        return post

    def test_to_param_spec_returns_correct_sfh_type(self, minimal_map_with_delayed_sfh):
        """to_param_spec() must preserve the posterior's SFH type, not default to dpl."""
        post = minimal_map_with_delayed_sfh

        # This should not raise "Unknown parameter 'sfh_delayed_log_total_mass'"
        spec = post.to_param_spec()

        # The spec's all_params must contain the delayed SFH parameters
        assert "sfh_delayed_log_total_mass" in spec.all_params
        # And the SFH type (as stored in the spec) must be 'delayed', not 'dpl'
        # Access via the spec's internal structure (through to_groups or _mean_sfh_type)
        groups = spec.to_groups()
        assert groups["sfh"]["type"] == "delayed"

    def test_validate_does_not_error_on_non_mcmc_posterior(self, minimal_map_with_delayed_sfh):
        """validate() must dispatch on self.method, not always assume mcmc_nuts."""
        post = minimal_map_with_delayed_sfh

        # The posterior was fit with MAP, so it should not error about n_steps
        # (n_steps is a MAP argument, not a NUTS argument)
        # We call validate() which internally calls run("mcmc_nuts", n_steps=n_steps, ...)
        # If it tries to pass n_steps to mcmc_nuts, it will fail
        # The fix ensures it only passes n_steps to MAP, not to mcmc_nuts.

        # For a MAP fit with no samples, validate() should either:
        # - Raise with a helpful message about needing samples, OR
        # - Use the posterior's method (MAP) and not try to run NUTS at all
        # Let's check that it doesn't crash with TypeError about mcmc_nuts
        try:
            # MAP fits have no samples, so validate() might raise ValueError about that
            # But it should NOT raise TypeError about mcmc_nuts not accepting n_steps
            post.validate(n_steps=10)
        except ValueError as e:
            # This is OK — MAP has no samples to validate
            assert "sample" in str(e).lower()
        except TypeError as e:
            # This is the bug we're fixing
            if "mcmc_nuts" in str(e) and "n_steps" in str(e):
                pytest.fail(f"validate() passed n_steps to mcmc_nuts: {e}")
            # Other TypeErrors should still fail
            raise

    def test_non_dpl_sfh_types_round_trip(self, ssp_data_wne, simple_observation):
        """Multiple SFH types must round-trip through to_param_spec."""
        from tengri import SEDModel, Fixed, DEFAULT
        from tengri.inference.fitter import Fitter

        for sfh_type in ["delayed", "tsnorm"]:
            model = SEDModel.build(
                ssp_data=ssp_data_wne,
                observation=simple_observation,
                sfh={"type": sfh_type, "all_params": Fixed(DEFAULT)},
                dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
                neb={"type": "none"},
                redshift=Fixed(0.05),
            )

            # Create a minimal posterior with the correct SFH prefix
            # simple_observation creates 3 bands
            n_bands = 3
            photometry = jnp.ones(n_bands)
            noise = jnp.ones(n_bands) * 0.01
            fitter = Fitter(model, data=photometry, noise=noise)
            post = fitter.run("map", n_steps=2)

            # This should not raise for any SFH type
            spec = post.to_param_spec()
            groups = spec.to_groups()
            assert groups["sfh"]["type"] == sfh_type, f"SFH type mismatch for {sfh_type}"
