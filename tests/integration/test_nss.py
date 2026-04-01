"""Integration tests for NSS with real forward model.

Requires SSP data files. Skipped gracefully if not found.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_SSP_PATH = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
_SSP_EXISTS = _SSP_PATH.exists()
pytestmark = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


@pytest.fixture(scope="module")
def smooth_fitter():
    """Create a smooth fitter with mock photometric data."""
    from tengri import Fitter, Model, ParamSpec, Uniform, load_filter_set, load_ssp_data

    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        redshift=0.1,
        stochastic=False,
    )
    ssp = load_ssp_data(str(_SSP_PATH))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    model = Model(spec, ssp, filters=filters)
    key = jax.random.PRNGKey(42)
    mock = model.mock(spec.sample(key), snr=20.0, key=jax.random.PRNGKey(1))
    return Fitter(model, mock.flux_obs, mock.noise)


class TestNSSSmooth:
    """NSS on smooth (parametric) SFH models."""

    def test_nss_produces_evidence(self, smooth_fitter):
        """NSS returns finite log-evidence."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        assert result.log_evidence is not None
        assert np.isfinite(result.log_evidence)

    def test_nss_produces_samples(self, smooth_fitter):
        """NSS returns posterior samples for all free params."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        assert result.samples is not None
        for name in smooth_fitter._free_names:
            assert name in result.samples
            assert result.samples[name].shape[0] > 0

    def test_nss_samples_in_bounds(self, smooth_fitter):
        """All NSS samples are within prior bounds."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        for name in smooth_fitter._free_names:
            lo, hi = smooth_fitter._bounds[name]
            vals = np.array(result.samples[name])
            assert np.all(vals >= lo - 1e-6), f"{name}: min={vals.min()}, lo={lo}"
            assert np.all(vals <= hi + 1e-6), f"{name}: max={vals.max()}, hi={hi}"

    def test_nss_diagnostics_complete(self, smooth_fitter):
        """NSS diagnostics contain expected keys."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        expected_keys = {
            "n_live",
            "num_delete",
            "num_inner_steps",
            "n_iterations",
            "n_dead",
            "log_evidence",
            "ess",
        }
        assert expected_keys.issubset(result.diagnostics.keys())

    def test_nss_ess_positive(self, smooth_fitter):
        """NSS effective sample size is positive."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        assert result.diagnostics["ess"] > 0

    def test_nss_method_name(self, smooth_fitter):
        """NSS result has correct method string."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        assert "NSS" in result.method

    def test_nss_summary_table(self, smooth_fitter):
        """summary_table works and includes evidence."""
        result = smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))
        table = result.summary_table()
        assert isinstance(table, str)
        assert "log Z" in table
