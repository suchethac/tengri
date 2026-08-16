# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Posterior arithmetic, resampling, and parameter-spec conversion."""

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


# ── BPT line names and fluxes used across several test classes ──────────────
_BPT_NAMES = ("Halpha", "Hbeta", "NII_6584", "OIII_5007")
_BPT_WAVES = jnp.array([6564.61, 4862.68, 6583.45, 5008.24])
# Ha=10, Hb=4, NII=5, OIII=2 → Ha/Hb=2.5, NII/Ha=0.5, OIII/Hb=0.5
_BPT_FLUX_1D = jnp.array([10.0, 4.0, 5.0, 2.0])


@pytest.fixture
def map_posterior():
    return Posterior(
        samples=None,
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "sfh_dpl_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="MAP (Adam)",
        wall_time_s=1.5,
        diagnostics={"n_steps": 100},
        loss_history=jnp.array([10.0, 5.0, 2.0]),
    )


@pytest.fixture
def sampling_posterior():
    key = jax.random.PRNGKey(0)
    n = 100
    return Posterior(
        samples={
            "sfh_dpl_alpha": 1.2 + 0.3 * jax.random.normal(key, (n,)),
            "sfh_dpl_beta": 1.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n,)),
            "met_logzsol": -0.3 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n,)),
        },
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "sfh_dpl_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="NUTS (BlackJAX)",
        wall_time_s=30.0,
        diagnostics={"n_divergent": 0, "n_samples": 100},
    )


class TestResample:
    """Test posterior resampling from MAP and sampling chains."""

    def test_resample_single(self, sampling_posterior):
        draw = sampling_posterior.resample(jax.random.PRNGKey(0), n=1)
        assert "sfh_dpl_alpha" in draw
        assert draw["sfh_dpl_alpha"].ndim == 0  # scalar

    def test_resample_batch(self, sampling_posterior):
        draw = sampling_posterior.resample(jax.random.PRNGKey(0), n=5)
        assert draw["sfh_dpl_alpha"].shape == (5,)

    def test_map_resample(self, map_posterior):
        draw = map_posterior.resample(jax.random.PRNGKey(0), n=1)
        assert float(draw["sfh_dpl_alpha"]) == pytest.approx(1.2)


class TestToParamSpec:
    """Test conversion of posterior to parameter specification."""

    def test_map_to_param_spec(self, map_posterior):
        spec = map_posterior.to_param_spec()
        from tengri.parameters.priors import Fixed

        d = spec.get_distribution("sfh_dpl_alpha")
        assert isinstance(d, Fixed)

    def test_sampling_to_param_spec(self, sampling_posterior):
        spec = sampling_posterior.to_param_spec()
        from tengri.parameters.priors import Gaussian

        d = spec.get_distribution("sfh_dpl_alpha")
        assert isinstance(d, Gaussian)
        assert d.mu == pytest.approx(1.2, abs=0.1)


class TestRepr:
    """Test string representation of posteriors."""

    def test_map_repr(self, map_posterior):
        r = repr(map_posterior)
        assert "MAP" in r
        assert "None" in r  # no samples

    def test_sampling_repr(self, sampling_posterior):
        r = repr(sampling_posterior)
        assert "NUTS" in r


class TestSummaryTable:
    """Test summary table generation for diagnostics."""

    def test_map_table_contains_method(self, map_posterior):
        t = map_posterior.summary_table()
        assert "MAP" in t
        assert "sfh_dpl_alpha" in t

    def test_sampling_table_contains_ess_header(self, sampling_posterior):
        t = sampling_posterior.summary_table()
        assert "ESS" in t
        assert "sfh_dpl_alpha" in t

    def test_sampling_table_shows_accept_rate(self):
        p = Posterior(
            samples={"x": jnp.ones(50)},
            params={"x": jnp.array(1.0)},
            method="mcmc_raytrace",
            wall_time_s=5.0,
            diagnostics={"accept_rate": 0.62},
        )
        t = p.summary_table()
        assert "accept=" in t

    def test_sampling_table_shows_divergences(self):
        p = Posterior(
            samples={"x": jnp.ones(50)},
            params={"x": jnp.array(1.0)},
            method="mcmc_nuts",
            wall_time_s=5.0,
            diagnostics={"n_divergent": 3},
        )
        t = p.summary_table()
        assert "divergences=3" in t

    def test_sampling_table_shows_final_loss(self):
        p = Posterior(
            samples={"x": jnp.ones(50)},
            params={"x": jnp.array(1.0)},
            method="MAP",
            wall_time_s=1.0,
            diagnostics={"final_loss": 12.34},
        )
        t = p.summary_table()
        assert "loss=" in t

    def test_log_evidence_included(self):
        p = Posterior(
            samples=None,
            params={"x": jnp.array(0.5)},
            method="nss",
            wall_time_s=60.0,
            diagnostics={"log_evidence_err": 0.05},
            log_evidence=-42.1,
        )
        t = p.summary_table()
        assert "log Z" in t
        assert "-42.1" in t


class TestDiagnosticsSummary:
    """Test diagnostic summary formatting."""

    def test_map_returns_short_string(self, map_posterior):
        s = map_posterior.diagnostics_summary()
        assert "MAP" in s
        assert "no samples" in s

    def test_sampling_returns_table(self, sampling_posterior):
        s = sampling_posterior.diagnostics_summary()
        assert "Method" in s
        assert "Samples" in s
        assert "sfh_dpl_alpha" in s
