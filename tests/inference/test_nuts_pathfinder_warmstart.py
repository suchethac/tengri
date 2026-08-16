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
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
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


#: One NUTS configuration for both warmup paths. A NUTS run here is dominated by
#: the XLA kernel compile, which is independent of the step count -- so shrinking
#: these numbers buys nothing, and every *distinct* setting buys another compile.
#: Four tests previously triggered five runs across three configurations; they
#: assert plumbing (a Posterior comes back, the diagnostics are labeled, the two
#: warmup paths do not share a cache entry), none of which depends on the counts.
_NUTS_KWARGS = dict(n_warmup=50, n_burnin=5, n_samples=20, verbose=False)


@pytest.fixture(scope="module")
def window_result(fitter_and_mock):
    """One window-adapted NUTS run, shared across the class.

    ``pathfinder_warmstart`` is deliberately NOT passed: the default must remain
    window adaptation, so omitting it pins the default rather than merely
    re-stating it.
    """
    fitter, _ = fitter_and_mock
    return fitter.run("mcmc_nuts", **_NUTS_KWARGS)


@pytest.fixture(scope="module")
def pathfinder_result(fitter_and_mock, window_result):
    """One pathfinder-warmstarted NUTS run, shared across the class.

    Depends on ``window_result`` so both adaptations run in the same process and
    against the same fitter -- which is precisely what makes the cache-separation
    assertion meaningful: a shared cache entry would hand this run back the
    window-adapted diagnostics.
    """
    fitter, _ = fitter_and_mock
    return fitter.run("mcmc_nuts", pathfinder_warmstart=True, **_NUTS_KWARGS)


@pytest.mark.skipif(not _has_blackjax(), reason="blackjax not installed")
class TestPathfinderWarmstart:
    """Contract: run_nuts(pathfinder_warmstart=True) must produce a valid
    Posterior and label itself in diagnostics."""

    def test_runs_without_error(self, pathfinder_result):
        assert isinstance(pathfinder_result, Posterior)
        assert pathfinder_result.samples is not None
        for arr in pathfinder_result.samples.values():
            assert arr.shape[0] == _NUTS_KWARGS["n_samples"]

    def test_diagnostics_label(self, pathfinder_result):
        assert pathfinder_result.diagnostics.get("warmup") == "pathfinder"

    def test_window_adaptation_still_default(self, window_result):
        """Regression: default path must still use window adaptation."""
        assert window_result.diagnostics.get("warmup") == "window"

    def test_cache_key_separation(self, window_result, pathfinder_result):
        """Window-adapted and pathfinder-adapted runs must not share cache."""
        assert window_result.diagnostics["warmup"] == "window"
        assert pathfinder_result.diagnostics["warmup"] == "pathfinder"
