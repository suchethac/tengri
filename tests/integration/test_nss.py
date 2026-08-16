# SPDX-License-Identifier: BSD-3-Clause
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
    from tengri import Fitter, Parameters, SEDModel, Uniform, load_filter_set, load_ssp_data

    spec = Parameters(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        redshift=0.1,
        stochastic=False,
    )
    ssp = load_ssp_data(str(_SSP_PATH))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    model = SEDModel(spec, ssp, filters=filters)
    key = jax.random.PRNGKey(42)
    mock = model.mock(spec.sample(key), snr=20.0, key=jax.random.PRNGKey(1))
    return Fitter(model, mock.flux_obs, mock.noise)


@pytest.fixture(scope="module")
def nss_result(smooth_fitter):
    """Run the nested sampler ONCE and share the posterior across the suite.

    Every test below interrogates a different facet of the same NSS run, on a
    fixed key, so the run is deterministic and re-running it per test recomputed
    a bit-identical result. Caching the *result* rather than the fitter turns
    seven ~85 s samplings into one; the assertions are unchanged.
    """
    return smooth_fitter.run("nss", n_live=100, num_delete=10, key=jax.random.PRNGKey(0))


class TestNSSSmooth:
    """NSS on smooth (parametric) SFH models."""

    def test_nss_produces_evidence(self, nss_result):
        """NSS returns finite log-evidence."""
        assert nss_result.log_evidence is not None
        assert np.isfinite(nss_result.log_evidence)

    def test_nss_produces_samples(self, smooth_fitter, nss_result):
        """NSS returns posterior samples for all free params."""
        assert nss_result.samples is not None
        for name in smooth_fitter._free_names:
            assert name in nss_result.samples
            assert nss_result.samples[name].shape[0] > 0

    def test_nss_samples_in_bounds(self, smooth_fitter, nss_result):
        """All NSS samples are within prior bounds."""
        for name in smooth_fitter._free_names:
            lo, hi = smooth_fitter._bounds[name]
            vals = np.array(nss_result.samples[name])
            assert np.all(vals >= lo - 1e-6), f"{name}: min={vals.min()}, lo={lo}"
            assert np.all(vals <= hi + 1e-6), f"{name}: max={vals.max()}, hi={hi}"

    def test_nss_diagnostics_complete(self, nss_result):
        """NSS diagnostics contain expected keys."""
        expected_keys = {
            "n_live",
            "num_delete",
            "num_inner_steps",
            "n_iterations",
            "n_dead",
            "log_evidence",
            "ess",
        }
        assert expected_keys.issubset(nss_result.diagnostics.keys())

    def test_nss_ess_positive(self, nss_result):
        """NSS effective sample size is positive."""
        assert nss_result.diagnostics["ess"] > 0

    def test_nss_method_name(self, nss_result):
        """NSS result has correct method string."""
        assert "NSS" in nss_result.method

    def test_nss_summary_table(self, nss_result):
        """summary_table works and includes evidence."""
        table = nss_result.summary_table()
        assert isinstance(table, str)
        assert "log Z" in table
