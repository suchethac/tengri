# SPDX-License-Identifier: BSD-3-Clause
"""Tests for NUTS with Pathfinder warm-start.

Exercises the ``pathfinder_warmstart=True`` path in
``tengri.inference.backends.mcmc.nuts.run_nuts``, which swaps
``blackjax.window_adaptation`` for ``blackjax.adaptation.pathfinder_adaptation``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.forward.sed_model import SEDModel
from tengri.inference.backends.mcmc import _shared
from tengri.inference.fitter import Fitter
from tengri.inference.posterior import Posterior
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = [pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")]


def _has_blackjax() -> bool:
    try:
        import blackjax  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def fitter_and_mock(ssp_data_wne, sdss_filters):
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )
    model = SEDModel(spec, ssp_data_wne, filters=sdss_filters)
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
    return fitter, true_params


@pytest.fixture(autouse=True)
def _cheap_elbo_draws(monkeypatch):
    """Shrink Pathfinder's ELBO draws for this module (#1028).

    Each ELBO draw is a full forward-model evaluation and BlackJAX vmaps
    ``maxiter (30) x n_draws`` of them, so the shipped default of 25 costs ~10 GB
    standalone and ~17 GB inside the full tier -- more than a 16 GB CI runner has.
    These tests assert *plumbing* (it runs, it labels its diagnostics, its cache key
    separates), none of which depends on how finely Pathfinder ranks the candidate
    Gaussians along its path. 5 draws exercises the same code at ~4.7 GB.

    The production default stays at 25 (Stan's ``num_elbo_draws``); it is asserted in
    ``test_pathfinder_elbo_draws.py``, which does not patch it.
    """
    monkeypatch.setattr(_shared, "_PATHFINDER_ELBO_DRAWS", 5)


@pytest.mark.skipif(not _has_blackjax(), reason="blackjax not installed")
class TestPathfinderWarmstart:
    """Contract: run_nuts(pathfinder_warmstart=True) must produce a valid
    Posterior and label itself in diagnostics."""

    def test_runs_without_error(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run(
            "mcmc_nuts",
            pathfinder_warmstart=True,
            n_warmup=50,
            n_burnin=10,
            n_samples=30,
            verbose=False,
        )
        assert isinstance(result, Posterior)
        assert result.samples is not None
        for arr in result.samples.values():
            assert arr.shape[0] == 30

    def test_diagnostics_label(self, fitter_and_mock):
        fitter, _ = fitter_and_mock
        result = fitter.run(
            "mcmc_nuts",
            pathfinder_warmstart=True,
            n_warmup=50,
            n_burnin=5,
            n_samples=20,
            verbose=False,
        )
        assert result.diagnostics.get("warmup") == "pathfinder"

    def test_window_adaptation_still_default(self, fitter_and_mock):
        """Regression: default path must still use window adaptation."""
        fitter, _ = fitter_and_mock
        result = fitter.run(
            "mcmc_nuts",
            n_warmup=50,
            n_burnin=5,
            n_samples=20,
            verbose=False,
        )
        assert result.diagnostics.get("warmup") == "window"

    def test_cache_key_separation(self, fitter_and_mock):
        """Window-adapted and pathfinder-adapted runs must not share cache."""
        fitter, _ = fitter_and_mock
        r1 = fitter.run(
            "mcmc_nuts",
            pathfinder_warmstart=False,
            n_warmup=50,
            n_burnin=5,
            n_samples=20,
            verbose=False,
        )
        r2 = fitter.run(
            "mcmc_nuts",
            pathfinder_warmstart=True,
            n_warmup=50,
            n_burnin=5,
            n_samples=20,
            verbose=False,
        )
        assert r1.diagnostics["warmup"] == "window"
        assert r2.diagnostics["warmup"] == "pathfinder"
