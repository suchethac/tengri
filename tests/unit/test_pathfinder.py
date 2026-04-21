"""Tests for Pathfinder inference."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from pathlib import Path

from tengri.forward.sed_model import Model
from tengri.inference.fitter import Fitter
from tengri.inference.posterior import Posterior
from tengri.parameters.parameters import ParamSpec
from tengri.parameters.priors import Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = [
    pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found"),
]


def _has_blackjax():
    try:
        import blackjax  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def fitter_and_mock(ssp_data_wne, sdss_filters):
    ssp = ssp_data_wne
    filters = sdss_filters

    spec = ParamSpec(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )
    model = Model(spec, ssp, filters=filters)
    true_params = {
        "sfh_dpl_alpha": 1.2,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 4.0,
        "sfh_dpl_log_peak_sfr": 0.9,
        "met_logzsol": -0.3,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))
    fitter = Fitter(model, mock.flux_obs, mock.noise)
    return fitter, true_params


@pytest.mark.skipif(not _has_blackjax(), reason="blackjax not installed")
class TestPathfinder:
    def test_returns_posterior(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run("pathfinder", n_samples=100, maxiter=10, verbose=False)
        assert isinstance(result, Posterior)
        assert "Pathfinder" in result.method
        assert result.samples is not None
        assert result.wall_time_s > 0

    def test_samples_shape(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        n = 200
        result = fitter.run("pathfinder", n_samples=n, maxiter=10, verbose=False)
        for name, arr in result.samples.items():
            assert arr.shape[0] == n, f"Expected {n} samples for {name}"

    def test_init_from_map(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        map_result = fitter.run("map", n_steps=100, verbose=False)
        result = fitter.run(
            "pathfinder",
            init_from=map_result,
            n_samples=100,
            maxiter=10,
            verbose=False,
        )
        assert isinstance(result, Posterior)

    def test_diagnostics(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run("pathfinder", n_samples=50, maxiter=10, verbose=False)
        assert "maxiter" in result.diagnostics
        assert "mean_log_q" in result.diagnostics
