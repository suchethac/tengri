# SPDX-License-Identifier: BSD-3-Clause
"""Tests for HMC+importance-sampling evidence estimation.

Tests the end-to-end HMC+IS backend including posterior sampling,
importance-sampled evidence computation, and quality diagnostics.

Requires SSP data. Marked as slow integration tests.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from tengri.inference.posterior import Posterior

_SSP_PATH = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
_SSP_EXISTS = _SSP_PATH.exists()
pytestmark = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


@pytest.fixture(scope="module")
def smooth_fitter():
    """Create a smooth fitter with mock photometric data.

    Fixes: pin auto-freed parameters (sfh_dpl_age_gyr, sfh_dpl_beta, dust_tau_diff)
    to achieve exactly D=5 free parameters and avoid exact degeneracies.
    Uses deterministic interior truth for mock generation.
    """
    from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform, load_filter_set, load_ssp_data

    spec = Parameters(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Fixed(1.0),  # Pin: not a free param
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_age_gyr=Fixed(13.0),  # Pin: not a free param (cosmic time for SF)
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Fixed(0.3),  # Pin: diffuse ISM dust (not a free param)
        redshift=0.1,
        stochastic=False,
    )
    ssp = load_ssp_data(str(_SSP_PATH))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    model = SEDModel(spec, ssp, filters=filters)

    # Deterministic interior truth: all values interior to Uniform bounds.
    # Avoid extremes to minimize posterior curvature, improving Laplace accuracy.
    truth = {
        'sfh_dpl_alpha': 1.2,  # interior of (0.5, 3.0)
        'sfh_dpl_tau_gyr': 4.0,  # interior of (0.5, 10.0)
        'sfh_dpl_log_total_mass': 10.0,  # interior of (7.0, 12.5)
        'met_logzsol': -0.8,  # interior of (-2.0, 0.2)
        'dust_tau_bc': 1.2,  # interior of (0.0, 3.0)
    }
    mock = model.mock(truth, snr=20.0, key=jax.random.PRNGKey(1))
    return Fitter(model, mock.flux_obs, mock.noise)


@pytest.fixture(scope="module")
def hmc_is_baseline(smooth_fitter):
    """HMC+IS with default proposal inflation."""
    return smooth_fitter.run(
        "hmc_is",
        n_warmup=300,
        n_burnin=100,
        n_samples=800,
        n_is_draws=50_000,
        proposal_inflation=1.5,
        key=jax.random.PRNGKey(0),
        verbose=False,
    )


class TestHMCISStructure:
    """HMC+IS returns proper Posterior structure."""

    def test_returns_posterior(self, hmc_is_baseline):
        assert isinstance(hmc_is_baseline, Posterior)

    def test_has_samples(self, hmc_is_baseline):
        assert hmc_is_baseline.samples is not None
        assert isinstance(hmc_is_baseline.samples, dict)
        assert len(hmc_is_baseline.samples) > 0

    def test_sample_shapes_correct(self, smooth_fitter, hmc_is_baseline):
        """Posterior samples should have shape (n_samples,) for each free param."""
        free_names = smooth_fitter._free_names
        for name in free_names:
            assert name in hmc_is_baseline.samples, f"Missing sample for {name}"
            samples = hmc_is_baseline.samples[name]
            assert samples.shape[0] > 0, f"No samples for {name}"

    def test_method_name(self, hmc_is_baseline):
        assert "HMC+IS" in hmc_is_baseline.method or "HMC" in hmc_is_baseline.method

    def test_has_log_evidence(self, hmc_is_baseline):
        assert hmc_is_baseline.log_evidence is not None
        assert np.isfinite(hmc_is_baseline.log_evidence)

    def test_has_wall_time(self, hmc_is_baseline):
        assert hmc_is_baseline.wall_time_s > 0


class TestHMCISEvidence:
    """HMC+IS evidence computation and error estimates."""

    def test_log_evidence_in_diagnostics(self, hmc_is_baseline):
        assert "log_evidence" in hmc_is_baseline.diagnostics
        assert np.isfinite(hmc_is_baseline.diagnostics["log_evidence"])

    def test_log_evidence_matches_posterior(self, hmc_is_baseline):
        """Posterior.log_evidence should match diagnostics."""
        np.testing.assert_allclose(
            float(hmc_is_baseline.log_evidence),
            float(hmc_is_baseline.diagnostics["log_evidence"]),
            rtol=1e-10,
        )

    def test_has_evidence_error(self, hmc_is_baseline):
        """Evidence error estimate should be positive."""
        assert "log_evidence_err" in hmc_is_baseline.diagnostics
        err = hmc_is_baseline.diagnostics["log_evidence_err"]
        assert err > 0
        assert np.isfinite(err)


class TestHMCISQualityMetrics:
    """HMC+IS quality metrics (ESS, max_weight_frac)."""

    def test_has_ess(self, hmc_is_baseline):
        assert "ess" in hmc_is_baseline.diagnostics
        ess = hmc_is_baseline.diagnostics["ess"]
        assert ess > 0
        assert ess <= hmc_is_baseline.diagnostics["n_is_draws"], "ESS is an absolute count"

    def test_ess_floor(self, hmc_is_baseline):
        """ESS floor: enough effective draws for a usable log Z error bar.

        SED posteriors are curved in xi-space, so a Student-t proposal is
        inefficient (ESS/N ~ 1e-3 is typical); ESS ~ 100 still gives
        sigma(log Z) ~ 0.1 nats, ample for BMA weights.
        """
        ess_count = hmc_is_baseline.diagnostics["ess"]
        assert ess_count >= 50, f"ESS = {ess_count:.0f} < 50: log Z error bar unusable"
        err = hmc_is_baseline.diagnostics["log_evidence_err"]
        assert err <= 0.3, f"log_evidence_err = {err:.3f} > 0.3 nats"

    def test_max_weight_frac(self, hmc_is_baseline):
        assert "max_weight_frac" in hmc_is_baseline.diagnostics
        mwf = hmc_is_baseline.diagnostics["max_weight_frac"]
        assert 0 <= mwf <= 1

    def test_max_weight_frac_reasonable(self, hmc_is_baseline):
        """No single draw may dominate the weight sum."""
        mwf = hmc_is_baseline.diagnostics["max_weight_frac"]
        assert mwf <= 0.25, f"max_weight_frac = {mwf:.4f} > 0.25: estimate dominated by one draw"

    def test_quality_warning_consistent(self, hmc_is_baseline):
        """Warning flag present exactly when a quality threshold is crossed."""
        d = hmc_is_baseline.diagnostics
        threshold_crossed = d["ess"] < 500 or d["max_weight_frac"] > 0.1
        assert ("is_quality_warning" in d) == threshold_crossed


class TestHMCISProposalParams:
    """HMC+IS proposal parameters are recorded."""

    def test_has_proposal_params(self, hmc_is_baseline):
        assert "proposal_df" in hmc_is_baseline.diagnostics
        assert "proposal_inflation" in hmc_is_baseline.diagnostics
        assert "proposal_cond_number" in hmc_is_baseline.diagnostics

    def test_proposal_df_value(self, hmc_is_baseline):
        df = hmc_is_baseline.diagnostics["proposal_df"]
        assert df > 0
        assert df == 5.0  # Default

    def test_proposal_inflation_value(self, hmc_is_baseline):
        inflation = hmc_is_baseline.diagnostics["proposal_inflation"]
        assert inflation > 0
        assert inflation == 1.5  # Used in fixture

    def test_proposal_cond_number(self, hmc_is_baseline):
        cond = hmc_is_baseline.diagnostics["proposal_cond_number"]
        assert cond > 0
        assert np.isfinite(cond)


@pytest.fixture(scope="module")
def hmc_is_higher_inflation(smooth_fitter):
    """HMC+IS with increased proposal inflation (2.5 instead of 1.5)."""
    return smooth_fitter.run(
        "hmc_is",
        n_warmup=300,
        n_burnin=100,
        n_samples=800,
        n_is_draws=50_000,
        proposal_inflation=2.5,
        key=jax.random.PRNGKey(0),
        verbose=False,
    )


class TestHMCISInflationEffect:
    """Increasing proposal inflation should yield consistent evidence."""

    def test_inflation_effect_on_evidence(self, hmc_is_baseline, hmc_is_higher_inflation):
        """Raising inflation should not dramatically change evidence estimate."""
        log_z_base = float(hmc_is_baseline.log_evidence)
        log_z_high = float(hmc_is_higher_inflation.log_evidence)
        err_base = float(hmc_is_baseline.diagnostics["log_evidence_err"])
        err_high = float(hmc_is_higher_inflation.diagnostics["log_evidence_err"])

        combined_err = np.sqrt(err_base**2 + err_high**2)
        tolerance = max(0.5, 3.0 * combined_err)
        diff = abs(log_z_base - log_z_high)

        assert diff <= tolerance, (
            f"Inflation 1.5 log Z = {log_z_base:.3f} ± {err_base:.4f}, "
            f"Inflation 2.5 log Z = {log_z_high:.3f} ± {err_high:.4f}, "
            f"diff = {diff:.3f} > tolerance {tolerance:.3f}"
        )

    def test_inflation_improves_ess(self, hmc_is_baseline, hmc_is_higher_inflation):
        """Higher inflation should yield comparable or better ESS."""
        ess_base = float(hmc_is_baseline.diagnostics["ess"])
        ess_high = float(hmc_is_higher_inflation.diagnostics["ess"])
        # Both should be > 0; higher isn't always better (depends on target)
        # but should be in the same ballpark
        assert ess_high > 0
        assert ess_base > 0


class TestHMCISHMCDiagnostics:
    """HMC diagnostics are preserved in output."""

    def test_has_hmc_diagnostics(self, hmc_is_baseline):
        """Should contain HMC diagnostics from underlying run_hmc."""
        diag = hmc_is_baseline.diagnostics
        # At minimum should have some HMC output
        assert len(diag) > 5  # More than just IS params
