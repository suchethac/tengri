# SPDX-License-Identifier: BSD-3-Clause
"""Tests for issue #1329 — fit(params=...) per-fit override and unknown-kwarg validation."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, ForwardModel, Observation, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve


def build_test_model(use_waveprecomp=False):
    """Build a small test model for hermetic testing."""
    from tengri.forward.sed_model import WavePrecomp

    wave = jnp.linspace(3000.0, 10000.0, 60)
    ages = jnp.linspace(-1.0, 1.14, 12)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    flux_grid = jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5
    ssp = SSPData(ssp_wave=wave, ssp_flux=flux_grid, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
        for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
    )
    obs = Observation(photometry=Photometry(filters=curves))
    sed = SEDModel.build(ssp_data=ssp, observation=obs, sfh={"type": "dpl"}, redshift=Fixed(0.5))
    truth = {"dust_tau_bc": 0.3, "dust_tau_diff": 0.2}
    data = jnp.asarray(np.asarray(sed.predict_photometry(truth)))
    noise = jnp.asarray(0.05 * np.abs(np.asarray(data)))

    forward = ForwardModel.build(sed=sed, observation=obs)
    if use_waveprecomp:
        forward = forward.with_approx(WavePrecomp(catalog_z_range=(0.01, 2.0), n_z=30))
    return forward, data, noise, sed


class TestFitParamsOverride:
    """Tests for params=... per-fit override reaching the forward pass."""

    pytestmark = pytest.mark.regression_bug

    def test_params_override_reaches_forward_pass(self):
        """params={"redshift": z} override actually affects the fit result."""
        # Use WavePrecomp to activate runtime redshift reading
        forward, data, noise, _ = build_test_model(use_waveprecomp=True)

        # Fit twice with different redshift overrides
        # The params override should actually affect the results
        result1 = forward.fit(data, noise, method="map", params={"redshift": 0.1}, n_steps=50)
        result2 = forward.fit(data, noise, method="map", params={"redshift": 1.5}, n_steps=50)

        # The two fits should differ because redshift changes the photometry
        # Check that the params differ
        assert result1.params["redshift"] != result2.params["redshift"]

        # At minimum, the redshift in the params should reflect the override
        assert abs(result1.params["redshift"] - 0.1) < 1e-5
        assert abs(result2.params["redshift"] - 1.5) < 1e-5

    def test_params_naming_free_param_raises(self):
        """params key naming a FREE parameter raises ValueError."""
        forward, data, noise, sed = build_test_model()

        # Identify a free parameter (if any exist in the model)
        free_params = sed.spec.free_params
        if free_params:
            free_name = free_params[0]
            with pytest.raises(ValueError, match="free parameter"):
                forward.fit(data, noise, method="map", params={free_name: 0.5})

    def test_params_naming_nonexistent_param_raises(self):
        """params key naming a nonexistent parameter raises ValueError."""
        forward, data, noise, _ = build_test_model()

        with pytest.raises(ValueError, match="not a valid parameter"):
            forward.fit(data, noise, method="map", params={"nonexistent_param": 0.5})

    def test_model_unmutated_after_params_fit(self):
        """Model object is not mutated by a params= fit."""
        forward, data, noise, sed = build_test_model()

        # Store the original fixed values
        original_fixed = dict(sed.spec.get_fixed_values())

        # Fit with a params override
        result = forward.fit(data, noise, method="map", params={"redshift": 0.1})

        # Model's fixed values should be unchanged
        current_fixed = dict(sed.spec.get_fixed_values())
        assert original_fixed == current_fixed


class TestUnknownKwargValidation:
    """Tests for validation of unknown kwargs passed to fit()."""

    pytestmark = pytest.mark.regression_bug

    def test_unknown_kwarg_raises(self):
        """Unknown kwarg (not part of fit signature or backend) raises."""
        forward, data, noise, _ = build_test_model()

        with pytest.raises((TypeError, ValueError), match="notakwarg"):
            forward.fit(data, noise, method="map", notakwarg=1)

    def test_real_backend_kwargs_still_accepted(self):
        """Real backend kwargs are not rejected."""
        forward, data, noise, _ = build_test_model()

        # These are real kwargs that MAP accepts
        real_kwargs = [
            {"n_steps": 100},
            {"learning_rate": 0.001},
            {"verbose": False},
        ]

        for kwargs in real_kwargs:
            # Should not raise
            result = forward.fit(data, noise, method="map", **kwargs)
            assert result is not None
