# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Fitter class."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.forward.sed_model import SEDModel
from tengri.inference.fitter import Fitter
from tengri.inference.posterior import Posterior
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Gaussian, Uniform
from tests._grad_parity import assert_grad_matches_fd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = [
    pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found"),
    pytest.mark.contract,
]


@pytest.fixture(scope="session")
def model_and_mock(ssp_data_wne, sdss_filters):
    ssp = ssp_data_wne
    filters = sdss_filters

    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )
    model = SEDModel(spec, ssp, filters=filters)

    true_params = {
        "sfh_dpl_alpha": 1.2,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 4.0,
        "sfh_dpl_log_total_mass": 0.9,  # log10(8 Msun/yr)
        # free (it carries a prior) but never given a truth value — the forward
        # used to substitute the spec default silently. Say it out loud (#1021).
        "sfh_dpl_age_gyr": float(spec.get_distribution("sfh_dpl_age_gyr").default),
        "met_logzsol": -0.3,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))
    return model, mock, true_params


class TestFitterConstruction:
    def test_creates(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        assert fitter.data_type == "photometry"

    def test_free_names(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        assert "sfh_dpl_alpha" in fitter._free_names or "sfh_alpha" in fitter._free_names
        assert "dust_slope" not in fitter._free_names  # fixed


class TestLossFunction:
    def test_loss_finite(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        loss_fn = fitter._build_loss_fn()
        init = fitter._initialize_unbounded(jax.random.PRNGKey(0))
        loss = loss_fn(init, fitter._data_args)
        assert jnp.isfinite(loss)

    def test_loss_gradient_finite(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        loss_fn = fitter._build_loss_fn()
        data_args = fitter._data_args
        init = fitter._initialize_unbounded(jax.random.PRNGKey(0))
        grad = assert_grad_matches_fd(lambda p: loss_fn(p, data_args), init)
        for name, g in grad.items():
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient for {name}"


class TestMAP:
    def test_returns_posterior(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        result = fitter.run("map", n_steps=50, verbose=False)
        assert isinstance(result, Posterior)
        assert "MAP" in result.method
        assert result.samples is None
        assert result.loss_history is not None

    def test_loss_decreases(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        result = fitter.run("map", n_steps=200, verbose=False)
        loss = np.array(result.loss_history)
        assert loss[-1] < loss[0], "Loss should decrease"

    def test_init_from_posterior(self, model_and_mock):
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise)
        result1 = fitter.run("map", n_steps=100, verbose=False)
        result2 = fitter.run("map", n_steps=50, init_from=result1, verbose=False)
        assert isinstance(result2, Posterior)


class TestGaussianPrior:
    def test_gaussian_prior_applied(self, ssp_data_wne, sdss_filters):
        ssp = ssp_data_wne
        filters = sdss_filters

        spec = Parameters(
            met_logzsol=Gaussian(-0.3, 0.1, lo=-2.0, hi=0.2),
            redshift=0.1,
            stochastic=False,
        )
        model = SEDModel(spec, ssp, filters=filters)
        params = spec.sample(jax.random.PRNGKey(0))
        mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))

        fitter = Fitter(model, mock.flux_obs, mock.noise)
        loss_fn = fitter._build_loss_fn()
        init = fitter._initialize_unbounded(jax.random.PRNGKey(2))
        loss = loss_fn(init, fitter._data_args)
        assert jnp.isfinite(loss)
