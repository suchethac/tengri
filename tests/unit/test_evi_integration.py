"""Integration tests for EVI inference pipeline.

Validates that the full EVI pipeline (build engine → optimize → sample)
runs without errors and produces reasonable results for a smooth SFH
model with synthetic SSP data.
"""

import jax
import jax.numpy as jnp
import pytest

from tengri import Fitter, Fixed, Model, Observation, ParamSpec, Photometry, Uniform
from tengri.models.observation.photometry import FilterCurve
from tengri.models.sps.dsps_wrapper import SSPData

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic SSP data (no disk I/O)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Minimal synthetic SSP for fast tests (3 Z × 20 ages × 100 λ)."""
    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)  # log10(age/Gyr)

    # Simple power-law SSP: brighter at blue wavelengths, younger ages
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])

    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def simple_spec():
    """Simple smooth SFH spec (D=5, minimal free params)."""
    return ParamSpec(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        met_logzsol=Uniform(-1.5, 0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def simple_observation():
    """Synthetic 3-band observation."""
    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.5 for _ in range(3)]
    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )
    photometry = Photometry(filters=curves)
    return Observation(photometry=photometry)


@pytest.fixture(scope="module")
def model_and_mock(simple_spec, synthetic_ssp, simple_observation):
    """Model + mock data for EVI testing."""
    model = Model(simple_spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(42)
    params = simple_spec.sample(key)
    mock = model.mock(params, snr=20.0, key=key)
    return model, mock, params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEVIRuns:
    """EVI pipeline runs without errors."""

    def test_evi_runs_and_returns_posterior(self, model_and_mock, simple_spec):
        """EVI produces a Posterior object with samples."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_evi",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=50,
            verbose=False,
            key=jax.random.PRNGKey(0),
        )

        # Check it returned something
        assert result is not None
        assert hasattr(result, "samples")
        assert hasattr(result, "diagnostics")

    def test_evi_samples_have_correct_keys(self, model_and_mock, simple_spec):
        """Posterior samples contain all free parameter names."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_evi",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=jax.random.PRNGKey(1),
        )

        for name in simple_spec.free_params:
            assert name in result.samples, f"Missing parameter: {name}"

    def test_evi_samples_finite(self, model_and_mock, simple_spec):
        """All posterior samples are finite (no NaN/inf)."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_evi",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=20,
            verbose=False,
            key=jax.random.PRNGKey(2),
        )

        for name, vals in result.samples.items():
            assert jnp.all(jnp.isfinite(vals)), f"{name} has non-finite samples"

    def test_evi_with_multiseed(self, model_and_mock, simple_spec):
        """EVI with multiple seeds runs without error."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_evi",
            n_iterations=3,
            n_samples=2,
            n_seeds=3,
            n_posterior_samples=10,
            verbose=False,
            key=jax.random.PRNGKey(3),
        )

        assert result is not None


class TestGeoVIMGVIRouting:
    """native_geovi and native_mgvi route through JIT engine."""

    def test_native_geovi_runs(self, model_and_mock):
        """Method 'native_geovi' runs (JIT geoVI engine)."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_geovi",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=10,
            verbose=False,
            key=jax.random.PRNGKey(10),
        )
        assert result is not None

    def test_native_mgvi_runs(self, model_and_mock):
        """Method 'native_mgvi' runs (JIT MGVI engine)."""
        model, mock, _ = model_and_mock
        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

        result = fitter.run(
            "native_mgvi",
            n_iterations=3,
            n_samples=2,
            n_seeds=1,
            n_posterior_samples=10,
            verbose=False,
            key=jax.random.PRNGKey(11),
        )
        assert result is not None
