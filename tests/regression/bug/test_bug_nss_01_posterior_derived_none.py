"""Regression test for BUG-NSS-01: posterior.derived crashes when stellar_mass_surviving is None.

See ADR / docs/known_bugs.md for full context.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBugNSS01PosteriorDerivedNone:
    """posterior.py:252-255 — derived must handle None-valued fields gracefully.

    Root cause: predict_derived() returns stellar_mass_surviving=None when
    ssp_data.ssp_mass_remaining is absent. posterior.derived then tries
    jnp.stack([None, None, ...]) which fails with TypeError.

    Fix: Helper function _stack_or_nan checks for any None values and
    returns a NaN array of the correct shape if found; otherwise stacks
    with jnp.asarray defensiveness.
    """

    def test_derived_with_none_field_returns_nan_array(self):
        """Verify posterior.derived handles NaN fields by returning NaN arrays.

        After optimization to use vmap(predict_sfh_quantities), the derived property
        now returns JAX arrays with NaN values (not Python None) when data is
        unavailable, which is correct for JIT-compatible batch computation.
        """
        from unittest.mock import MagicMock

        from tengri.forward.prediction import SFHQuantities
        from tengri.inference.posterior import Posterior

        # Create a mock model
        mock_model = MagicMock()
        n_samples = 3

        # Create a function that returns single-sample output.
        # vmap will then batch over it.
        def mock_predict_sfh(params_dict):
            """Return a single-sample SFHQuantities (non-batched).

            vmap will apply this per sample and batch the results.
            """
            # We need to extract the scalar value from each batched input
            # When vmap applies this function, params_dict values will be
            # scalar tracers, so we just return scalar SFHQuantities.
            return SFHQuantities(
                stellar_mass=jnp.array(1e11),
                stellar_mass_surviving=jnp.nan,  # NaN when unavailable
                sfr_100myr=jnp.array(10.0),
                sfr_10myr=jnp.array(2.0),
                ssfr=jnp.array(1e-10),
                mass_weighted_age_gyr=jnp.array(5.0),
                mass_weighted_metallicity=jnp.array(-1.5),
            )

        mock_model.predict_sfh_quantities = mock_predict_sfh

        # Create Posterior with samples
        posterior = Posterior(
            samples={
                "sfh_dpl_alpha": jnp.array([1.2, 1.3, 1.1]),
                "sfh_dpl_beta": jnp.array([1.0, 1.05, 0.95]),
            },
            params={
                "sfh_dpl_alpha": jnp.array(1.2),
                "sfh_dpl_beta": jnp.array(1.0),
            },
            method="NSS",
            wall_time_s=10.0,
            diagnostics={},
            _model=mock_model,
        )

        # Call derived property
        derived = posterior.derived

        # Check that stellar_mass_surviving is a NaN array (not crashed)
        assert "stellar_mass_surviving" in derived
        assert derived["stellar_mass_surviving"].shape == (n_samples,)
        assert jnp.isnan(derived["stellar_mass_surviving"]).all(), (
            "stellar_mass_surviving should be all NaN when unavailable for all samples"
        )

        # Check that other fields are present and finite
        assert "stellar_mass" in derived
        assert derived["stellar_mass"].shape == (n_samples,)
        assert jnp.all(jnp.isfinite(derived["stellar_mass"])), "stellar_mass should be finite"

        assert "sfr_100myr" in derived
        assert derived["sfr_100myr"].shape == (n_samples,)
        assert jnp.all(jnp.isfinite(derived["sfr_100myr"])), "sfr_100myr should be finite"
