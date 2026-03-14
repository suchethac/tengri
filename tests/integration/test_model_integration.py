"""Integration tests for the new Model class with real SSP data."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.distributions import Uniform, Gaussian, Fixed
from diffsed.param_spec import ParamSpec
from diffsed.model import Model, MockData
from diffsed.models.sps.dsps_wrapper import load_ssp_data
from diffsed.models.observation.filters import load_filter_set


# ---------------------------------------------------------------------------
# Skip if SSP data not available
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found",
)


@pytest.fixture(scope="session")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="session")
def filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


@pytest.fixture(scope="session")
def parametric_spec():
    return ParamSpec(
        sfh_alpha=Uniform(0.5, 3.0),
        sfh_beta=Uniform(0.3, 2.0),
        sfh_tau_peak_gyr=Uniform(0.5, 10.0),
        sfh_peak_sfr=Uniform(0.1, 50.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
        stochastic=False,
    )


@pytest.fixture(scope="session")
def stochastic_spec():
    return ParamSpec(
        sfh_alpha=Uniform(0.5, 3.0),
        sfh_beta=Uniform(0.3, 2.0),
        sfh_tau_peak_gyr=Uniform(0.5, 10.0),
        sfh_peak_sfr=Uniform(0.1, 50.0),
        psd_sigma=Uniform(0.1, 3.0),
        psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
        stochastic=True,
        n_grid=64,
    )


@pytest.fixture(scope="session")
def parametric_model(parametric_spec, ssp_data, filters):
    return Model(parametric_spec, ssp_data, filters=filters)


@pytest.fixture(scope="session")
def stochastic_model(stochastic_spec, ssp_data, filters):
    return Model(stochastic_spec, ssp_data, filters=filters)


@pytest.fixture(scope="session")
def typical_params():
    return {
        "sfh_alpha": 1.2,
        "sfh_beta": 1.0,
        "sfh_tau_peak_gyr": 4.0,
        "sfh_peak_sfr": 8.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }


# ===================================================================
# SED Predictions
# ===================================================================

class TestPredictSed:
    def test_shape(self, parametric_model, typical_params):
        sed = parametric_model.predict_sed(typical_params)
        assert sed.shape == parametric_model.ssp_data.ssp_wave.shape

    def test_finite(self, parametric_model, typical_params):
        sed = parametric_model.predict_sed(typical_params)
        assert jnp.all(jnp.isfinite(sed))

    def test_positive(self, parametric_model, typical_params):
        sed = parametric_model.predict_sed(typical_params)
        assert jnp.all(sed >= 0)


class TestPredictPhotometry:
    def test_shape(self, parametric_model, typical_params):
        phot = parametric_model.predict_photometry(typical_params)
        assert phot.shape == (5,)  # 5 SDSS bands

    def test_finite_positive(self, parametric_model, typical_params):
        phot = parametric_model.predict_photometry(typical_params)
        assert jnp.all(jnp.isfinite(phot))
        assert jnp.all(phot > 0)

    def test_physical_range(self, parametric_model, typical_params):
        phot = parametric_model.predict_photometry(typical_params)
        # Typical galaxy at z=0.1: ~1e-30 to 1e-26 erg/s/cm²/Hz
        assert jnp.all(phot > 1e-35)
        assert jnp.all(phot < 1e-20)


class TestPredictSfh:
    def test_keys(self, parametric_model, typical_params):
        sfh = parametric_model.predict_sfh(typical_params)
        assert "t_gyr" in sfh
        assert "sfr_mean" in sfh
        assert "sfr_full" in sfh

    def test_positive_sfr(self, parametric_model, typical_params):
        sfh = parametric_model.predict_sfh(typical_params)
        assert jnp.all(sfh["sfr_mean"] >= 0)

    def test_parametric_mean_equals_full(self, parametric_model, typical_params):
        sfh = parametric_model.predict_sfh(typical_params)
        np.testing.assert_allclose(
            np.array(sfh["sfr_mean"]),
            np.array(sfh["sfr_full"]),
            rtol=1e-6,
        )


class TestPredictDerived:
    def test_keys(self, parametric_model, typical_params):
        d = parametric_model.predict_derived(typical_params)
        assert "stellar_mass" in d
        assert "sfr_100myr" in d
        assert "ssfr" in d

    def test_mass_positive(self, parametric_model, typical_params):
        d = parametric_model.predict_derived(typical_params)
        assert float(d["stellar_mass"]) > 0

    def test_mass_reasonable(self, parametric_model, typical_params):
        d = parametric_model.predict_derived(typical_params)
        mass = float(d["stellar_mass"])
        assert 1e7 < mass < 1e13


# ===================================================================
# Stochastic Model
# ===================================================================

class TestStochastic:
    def test_predict_sed_works(self, stochastic_model, stochastic_spec):
        params = stochastic_spec.sample(jax.random.PRNGKey(42))
        sed = stochastic_model.predict_sed(params)
        assert jnp.all(jnp.isfinite(sed))

    def test_sfh_full_differs_from_mean(self, stochastic_model, stochastic_spec):
        params = stochastic_spec.sample(jax.random.PRNGKey(42))
        # Force non-zero psd_sigma
        params = {**params, "psd_sigma": 1.5}
        sfh = stochastic_model.predict_sfh(params)
        # With non-zero sigma and random xi, full != mean
        assert not jnp.allclose(sfh["sfr_mean"], sfh["sfr_full"])


# ===================================================================
# Equivalence: parametric model with sigma=0 ≈ stochastic model with sigma=0
# ===================================================================

class TestEquivalence:
    def test_zero_sigma_equivalence(self, ssp_data, filters):
        spec_param = ParamSpec(
            sfh_alpha=1.2, sfh_beta=1.0, sfh_tau_peak_gyr=4.0,
            sfh_peak_sfr=8.0, met_logzsol=-0.3,
            dust_tau_bc=1.0, dust_tau_diff=0.3, dust_slope=-0.7,
            redshift=0.1, stochastic=False,
        )
        spec_stoch = ParamSpec(
            sfh_alpha=1.2, sfh_beta=1.0, sfh_tau_peak_gyr=4.0,
            sfh_peak_sfr=8.0, psd_sigma=0.0, psd_tau_myr=50.0,
            met_logzsol=-0.3,
            dust_tau_bc=1.0, dust_tau_diff=0.3, dust_slope=-0.7,
            redshift=0.1, stochastic=True, n_grid=64,
        )

        model_p = Model(spec_param, ssp_data, filters=filters)
        model_s = Model(spec_stoch, ssp_data, filters=filters)

        params_p = spec_param.sample(jax.random.PRNGKey(0))
        params_s = spec_stoch.sample(jax.random.PRNGKey(0))

        phot_p = model_p.predict_photometry(params_p)
        phot_s = model_s.predict_photometry(params_s)

        # With sigma=0, GP contribution is zero, so results should match
        np.testing.assert_allclose(
            np.array(phot_p), np.array(phot_s), rtol=0.02,
            err_msg="Parametric and stochastic (sigma=0) should give similar photometry"
        )


# ===================================================================
# Mock Generation
# ===================================================================

class TestMock:
    def test_mock_structure(self, parametric_model, typical_params):
        mock = parametric_model.mock(typical_params, snr=20.0, key=jax.random.PRNGKey(0))
        assert isinstance(mock, MockData)
        assert mock.flux_true.shape == (5,)
        assert mock.flux_obs.shape == (5,)
        assert mock.noise.shape == (5,)

    def test_mock_noise_scaling(self, parametric_model, typical_params):
        mock = parametric_model.mock(typical_params, snr=20.0, key=jax.random.PRNGKey(0))
        expected_noise = mock.flux_true / 20.0
        np.testing.assert_allclose(np.array(mock.noise), np.array(expected_noise))

    def test_mock_batch_shapes(self, parametric_model, parametric_spec):
        batch = parametric_spec.sample_batch(jax.random.PRNGKey(0), 5)
        mock_batch = parametric_model.mock_batch(batch, snr=20.0, key=jax.random.PRNGKey(1))
        assert mock_batch.flux_true.shape == (5, 5)  # 5 galaxies, 5 bands
        assert mock_batch.flux_obs.shape == (5, 5)


# ===================================================================
# Gradient Flow
# ===================================================================

class TestGradients:
    def test_photometry_gradient(self, parametric_model, typical_params):
        def loss(p):
            return jnp.sum(parametric_model.predict_photometry(p))

        grad = jax.grad(loss)(typical_params)
        assert jnp.isfinite(grad["sfh_peak_sfr"])
        assert float(grad["sfh_peak_sfr"]) > 0  # more SFR → more flux

    def test_derived_gradient(self, parametric_model, typical_params):
        def loss(p):
            return parametric_model.predict_derived(p)["stellar_mass"]

        grad = jax.grad(loss)(typical_params)
        assert jnp.isfinite(grad["sfh_peak_sfr"])
