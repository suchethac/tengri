# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #3456: degenerate MCMC and SMC runs must not pass silently.

Bug: on a one-parameter toy with an analytic posterior, ``mcmc_ghmc`` returned a
mean 210 grid-sd off with R-hat 1.70 and ESS 0.52 and printed nothing;
``mcmc_mclmc`` failed with an opaque dependency error on a 1-D model; and
``mcmc_smc`` reported 10.4 effective particles of 512 with no warning.

Fix under test: one shared post-run check (``check_sampler_convergence``) warns
once and records ``diagnostics["convergence_warning"]``; MCLMC refuses a
one-dimensional target before any compile; SMC documents ``n_divergent``.

The unit tests build synthetic posteriors and run in milliseconds. The
end-to-end toy tests need the bare-stellar fsps SSP file and are skipped without it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.posterior import Posterior
from tengri.inference.sampler_health import check_sampler_convergence

pytestmark = pytest.mark.regression_bug

_RNG = np.random.default_rng(3456)


def _posterior(samples, diagnostics, method="test"):
    """Build a model-free posterior from draws, the way a backend does."""
    return Posterior(
        samples={k: jnp.asarray(v) for k, v in samples.items()},
        params={k: jnp.asarray(float(np.mean(v))) for k, v in samples.items()},
        method=method,
        wall_time_s=0.0,
        diagnostics=diagnostics,
    )


def _ar1(n, rho, rng):
    """A stationary AR(1) chain: strongly autocorrelated for rho near one."""
    x = np.empty(n)
    x[0] = rng.standard_normal()
    noise = rng.standard_normal(n) * np.sqrt(1.0 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + noise[i]
    return x


class TestSharedConvergenceCheck:
    """The post-run check, on synthetic draws."""

    def test_split_chains_that_disagree_warn_once_and_name_the_parameter(self):
        """Two chains centered far apart: R-hat far above 1.1, one RuntimeWarning."""
        n = 400
        chains = np.concatenate([_RNG.normal(0.0, 1.0, n), _RNG.normal(8.0, 1.0, n)])
        post = _posterior(
            {"sfh_dpl_log_total_mass": chains},
            {"n_chains": 2, "n_samples": n},
        )

        with pytest.warns(RuntimeWarning, match="mcmc_ghmc") as record:
            check_sampler_convergence(post, backend="mcmc_ghmc")

        assert len([w for w in record if issubclass(w.category, RuntimeWarning)]) == 1
        assert "sfh_dpl_log_total_mass" in str(record[0].message)
        assert "R-hat" in str(record[0].message)
        assert post.diagnostics["convergence_warning"] is not None
        assert "R-hat" in post.diagnostics["convergence_warning"]

    def test_strongly_autocorrelated_chain_warns_on_low_ess(self):
        """A single chain with ESS far below 10 % of its draws warns."""
        draws = _ar1(2000, 0.999, _RNG)
        post = _posterior({"x": draws}, {"n_chains": 1, "n_samples": draws.size})

        with pytest.warns(RuntimeWarning, match="ESS"):
            check_sampler_convergence(post, backend="mcmc_ghmc")
        assert "ESS" in post.diagnostics["convergence_warning"]

    def test_healthy_chains_are_silent_and_record_none(self, recwarn):
        """Two well-mixed chains: no warning, and the verdict is recorded as None."""
        n = 500
        draws = _RNG.normal(0.0, 1.0, 2 * n)
        post = _posterior({"x": draws}, {"n_chains": 2, "n_samples": n})

        check_sampler_convergence(post, backend="mcmc_nuts")

        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        assert post.diagnostics["convergence_warning"] is None

    def test_smc_degenerate_population_warns_on_particle_ess(self):
        """SMC with 10.4 effective particles of 512 is flagged, as in #3456."""
        draws = _RNG.normal(0.0, 1.0, 512)
        post = _posterior(
            {"x": draws},
            {"n_chains": 1, "n_samples": 512, "n_particles": 512, "min_ancestor_ess": 10.4},
        )

        with pytest.warns(RuntimeWarning, match="mcmc_smc"):
            check_sampler_convergence(post, backend="mcmc_smc")
        assert "particle ESS" in post.diagnostics["convergence_warning"]

    def test_smc_healthy_population_is_silent(self, recwarn):
        """A population with ancestor ESS near the particle count does not warn."""
        draws = _RNG.normal(0.0, 1.0, 512)
        post = _posterior(
            {"x": draws},
            {"n_chains": 1, "n_samples": 512, "n_particles": 512, "min_ancestor_ess": 400.0},
        )

        check_sampler_convergence(post, backend="mcmc_smc")

        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        assert post.diagnostics["convergence_warning"] is None

    def test_result_without_samples_is_returned_untouched(self):
        """MAP-style results carry no draws; the check leaves them alone."""
        post = Posterior(
            samples=None,
            params={"x": jnp.asarray(1.0)},
            method="map",
            wall_time_s=0.0,
            diagnostics={},
        )
        assert check_sampler_convergence(post, backend="map") is post
        assert "convergence_warning" not in post.diagnostics


def _wne_toy(ssp_data):
    """A one-parameter SED fit: log total mass free, every other input fixed."""
    from tengri import (
        DEFAULT,
        Fixed,
        ForwardModel,
        Observation,
        Photometry,
        SEDModel,
        Uniform,
        recipes,
    )
    from tengri.inference.fitter import Fitter

    bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    obs = Observation(photometry=Photometry.from_names(bands))
    config = recipes.star_forming_photometry()
    config["redshift"] = Fixed(0.05)
    config["sfh"] = {
        "type": "dpl",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(7.0, 12.5),
    }
    config["dust_attenuation"] = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_diff": Fixed(1.0),
    }
    config["met"] = {"all_params": Fixed(DEFAULT), "logzsol": Fixed(-0.5)}
    sed = SEDModel.build(ssp_data=ssp_data, observation=obs, **config)
    forward = ForwardModel.build(sed=sed, observation=obs)
    mock = sed.mock({"sfh_dpl_log_total_mass": 10.0}, key=jax.random.PRNGKey(1))
    return Fitter(forward, mock.flux_obs, mock.noise, profile_mass=False)


@pytest.fixture(scope="module")
def toy_fitter(ssp_data_fsps):
    # Bare-stellar SSP: the wNE grids are refused by the Cue nebular backend.
    return _wne_toy(ssp_data_fsps)


class TestEndToEndToy:
    """The shared check and the MCLMC guard, through ``Fitter.run``."""

    @pytest.mark.slow
    def test_ghmc_on_one_parameter_toy_warns(self, toy_fitter):
        """The #3456 repro: a degenerate GHMC run is flagged, not returned silently."""
        with pytest.warns(RuntimeWarning, match="mcmc_ghmc"):
            post = toy_fitter.run(
                "mcmc_ghmc",
                key=jax.random.PRNGKey(2),
                allow_unvalidated=True,
                n_warmup=200,
                n_burnin=50,
                n_samples=300,
            )
        assert post.diagnostics["convergence_warning"] is not None

    def test_nuts_on_same_toy_is_silent(self, toy_fitter, recwarn):
        """NUTS on the same target mixes, so the shared check stays quiet."""
        post = toy_fitter.run(
            "mcmc_nuts",
            key=jax.random.PRNGKey(2),
            n_warmup=300,
            n_burnin=100,
            n_samples=500,
        )
        assert not [
            w
            for w in recwarn
            if issubclass(w.category, RuntimeWarning) and "convergence" in str(w.message)
        ]
        assert post.diagnostics["convergence_warning"] is None

    @pytest.mark.parametrize("method", ["mcmc_mclmc", "mcmc_adjusted_mclmc"])
    def test_mclmc_refuses_one_dimensional_target_before_sampling(self, toy_fitter, method):
        """A 1-D MCLMC request gets the named ValueError, not the BlackJAX error."""
        with pytest.raises(ValueError, match="MCLMC needs ≥ 2 free parameters"):
            toy_fitter.run(method, key=jax.random.PRNGKey(2), allow_unvalidated=True)
