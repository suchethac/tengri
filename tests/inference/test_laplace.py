# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Laplace approximation inference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pathlib import Path

from tengri.forward.sed_model import SEDModel
from tengri.inference.fitter import Fitter
from tengri.inference.posterior import Posterior
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = [
    pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found"),
    pytest.mark.contract,
]


@pytest.fixture(scope="session")
def fitter_and_map(ssp_data_wne, sdss_filters):
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
        "sfh_dpl_log_total_mass": 0.9,
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
    fitter = Fitter(model, mock.flux_obs, mock.noise)
    map_result = fitter.run("map", n_steps=200, verbose=False)
    return fitter, map_result, true_params


class TestLaplace:
    def test_returns_posterior(self, fitter_and_map):
        fitter, map_result, _ = fitter_and_map
        result = fitter.run("laplace", init_from=map_result, n_samples=100, verbose=False)
        assert isinstance(result, Posterior)
        assert "Laplace" in result.method
        assert result.samples is not None
        assert result.wall_time_s > 0

    def test_samples_shape(self, fitter_and_map):
        fitter, map_result, _ = fitter_and_map
        n = 200
        result = fitter.run("laplace", init_from=map_result, n_samples=n, verbose=False)
        for name, arr in result.samples.items():
            assert arr.shape[0] == n, f"Expected {n} samples for {name}"

    def test_samples_centered_on_map(self, fitter_and_map):
        fitter, map_result, _ = fitter_and_map
        result = fitter.run("laplace", init_from=map_result, n_samples=2000, verbose=False)
        # Posterior mean should be close to MAP params
        for name in map_result.params:
            if name in result.params:
                map_val = float(map_result.params[name])
                laplace_mean = float(result.params[name])
                # Allow generous tolerance (5 sigma of sample std)
                sample_std = float(jnp.std(result.samples[name]))
                if sample_std > 0:
                    assert abs(laplace_mean - map_val) < 5 * sample_std, (
                        f"{name}: mean {laplace_mean:.3f} far from MAP {map_val:.3f}"
                    )

    def test_log_evidence(self, fitter_and_map):
        fitter, map_result, _ = fitter_and_map
        result = fitter.run("laplace", init_from=map_result, n_samples=50, verbose=False)
        assert "log_evidence" in result.diagnostics
        assert np.isfinite(result.diagnostics["log_evidence"])

    def test_eigenvalue_diagnostics(self, fitter_and_map):
        fitter, map_result, _ = fitter_and_map
        result = fitter.run("laplace", init_from=map_result, n_samples=50, verbose=False)
        assert "eigenvalues" in result.diagnostics
        eigs = result.diagnostics["eigenvalues"]
        assert jnp.all(eigs > 0), "All eigenvalues should be positive"
        assert "condition_number" in result.diagnostics

    def test_auto_runs_map(self, fitter_and_map):
        fitter, _, _ = fitter_and_map
        # No init_from — should auto-run MAP internally
        result = fitter.run("laplace", n_samples=50, n_map_steps=50, verbose=False)
        assert isinstance(result, Posterior)
        assert result.samples is not None
