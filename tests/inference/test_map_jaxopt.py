# SPDX-License-Identifier: BSD-3-Clause
"""Tests for L-BFGS MAP optimizer (optax scan-batch path).

Verifies that L-BFGS converges correctly through the
``fitter.run("map", optimizer="lbfgs")`` API.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pytest

SSP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(SSP_FILE),
        reason="SSP file not available; integration-level fixture",
    ),
    pytest.mark.contract,
]


@pytest.fixture(scope="module")
def fitter_and_mock(ssp_data_wne, sdss_filters):
    from tengri import (
        Fitter,
        Fixed,
        Observation,
        Parameters,
        Photometry,
        SEDModel,
        Uniform,
    )

    spec = Parameters(
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        mean_sfh_type="tsnorm",
    )
    ssp = ssp_data_wne
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    model = SEDModel(spec, ssp, observation=obs)
    true_params = {
        "sfh_tsnorm_log_total_mass": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 3.0,
        "sfh_tsnorm_width_gyr": 1.5,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 3.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(42))
    fitter = Fitter(model, mock.flux_obs, mock.noise)
    return fitter, mock


class TestJaxoptOptimizers:
    """Each jaxopt solver converges and returns a valid Posterior."""

    def test_converges(self, fitter_and_mock):
        from tengri.inference.posterior import Posterior

        fitter, _ = fitter_and_mock
        result = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="lbfgs",
            n_steps=200,
            verbose=False,
        )
        assert isinstance(result, Posterior)
        assert jnp.isfinite(result.diagnostics["final_loss"])
        assert result.diagnostics["n_steps"] > 0

    def test_lower_loss_than_adam(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        lbfgs = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="lbfgs",
            n_steps=200,
            verbose=False,
        )
        adam = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="adam",
            n_steps=500,
            verbose=False,
        )
        assert lbfgs.diagnostics["final_loss"] <= adam.diagnostics["final_loss"]


class TestJaxoptPosteriorStructure:
    """Verify Posterior object fields from jaxopt path."""

    def test_method_name_contains_optimizer(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="lbfgs",
            n_steps=50,
            verbose=False,
        )
        assert "L-BFGS" in result.method

    def test_diagnostics_has_optimizer_name(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="lbfgs",
            n_steps=200,
            verbose=False,
        )
        assert result.diagnostics["optimizer"] == "L-BFGS"
        assert "final_loss" in result.diagnostics
        assert "n_steps" in result.diagnostics

    def test_loss_history_is_array(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="lbfgs",
            n_steps=50,
            verbose=False,
        )
        assert hasattr(result, "loss_history")
        assert result.loss_history.ndim == 1
        assert len(result.loss_history) > 0

    def test_params_are_physical(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run(
            method="map",
            key=jax.random.PRNGKey(0),
            optimizer="lbfgs",
            n_steps=50,
            verbose=False,
        )
        for name, val in result.params.items():
            assert jnp.isfinite(val).all(), f"{name} is not finite"


class TestOptimizerErrors:
    """Error handling for unknown optimizer strings."""

    def test_unknown_optimizer_raises(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        with pytest.raises(ValueError, match="Unknown optimizer"):
            fitter.run(
                method="map",
                key=jax.random.PRNGKey(0),
                optimizer="unknown_solver",
                verbose=False,
            )

    def test_error_mentions_all_options(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        with pytest.raises(ValueError, match="lbfgs"):
            fitter.run(
                method="map",
                key=jax.random.PRNGKey(0),
                optimizer="unknown_solver",
                verbose=False,
            )
