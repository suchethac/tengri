# SPDX-License-Identifier: BSD-3-Clause
"""The forward-chunk budget divides by the live chains a catalog cell carries (#3195).

A catalog cell vmaps ``K * n_ensemble`` chains for ChEES, so an ``"auto"`` K that
divided the budget by K alone would overshoot by the ensemble width. These tests
pin three things: the pure formula, that the catalog ChEES call site passes the
resolved ensemble as ``n_chains``, and that the spectroscopy bucket path resolves
``batch_size="auto"`` through the same resolver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract


class _Reached(Exception):
    """Raised by a recording stub to stop a run at the call under test."""


def _small_model(synthetic_ssp, simple_observation):
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def _two_galaxies(model):
    truth = dict(model.spec.sample(jax.random.PRNGKey(0)))
    truth["sfh_dpl_log_total_mass"] = jnp.array(10.0)
    flux = model.predict_photometry(truth)
    noise = jnp.abs(flux) * 0.02
    return [{"flux_obs": flux, "noise": noise}, {"flux_obs": 2.0 * flux, "noise": 2.0 * noise}]


class TestPureFormula:
    def test_ensemble_width_divides_the_budget(self):
        """With the 2 GB default budget, 8 live chains per galaxy shrink K from 63 to 7."""
        from tengri.inference._batching import resolve_forward_chunk_size

        one_chain = resolve_forward_chunk_size(
            "auto", n_gal=1000, n_data_per_gal=100, n_chains=1, memory_budget_gb=2.0
        )
        eight_chains = resolve_forward_chunk_size(
            "auto", n_gal=1000, n_data_per_gal=100, n_chains=8, memory_budget_gb=2.0
        )
        assert one_chain == 63
        assert eight_chains == 7
        assert eight_chains == one_chain // 8


class TestCatalogChEESCallSite:
    @pytest.mark.parametrize(
        ("n_ensemble", "expected_live_chains"),
        [(None, 8), (16, 16)],
        ids=["default-ensemble", "explicit-ensemble-16"],
    )
    def test_chees_passes_its_resolved_ensemble_as_live_chains(
        self, monkeypatch, synthetic_ssp, simple_observation, n_ensemble, expected_live_chains
    ):
        import tengri.inference.catalog_fitter as catalog_fitter
        from tengri import CatalogFitter

        seen: dict = {}

        def record(requested, **kwargs):
            seen.update(kwargs)
            raise _Reached

        monkeypatch.setattr(catalog_fitter, "resolve_forward_chunk_size", record)
        model = _small_model(synthetic_ssp, simple_observation)
        cat = CatalogFitter(model, _two_galaxies(model), data_type="photometry")
        run_kwargs = {} if n_ensemble is None else {"n_ensemble": n_ensemble}

        with pytest.raises(_Reached):
            cat.run(
                "mcmc_chees",
                key=jax.random.PRNGKey(1),
                n_warmup=5,
                n_burnin=2,
                n_samples=5,
                **run_kwargs,
            )
        assert seen.get("n_chains") == expected_live_chains


class TestSpectroscopyBucketBatchSize:
    def test_auto_resolves_through_the_budget_with_the_live_chains(self, monkeypatch):
        from tengri.inference import batched_spectro
        from tengri.inference.batched_spectro import _resolve_bucket_batch_size

        seen: dict = {}

        def record(requested, **kwargs):
            seen["requested"] = requested
            seen.update(kwargs)
            return 3

        monkeypatch.setattr(batched_spectro, "resolve_forward_chunk_size", record)
        out = _resolve_bucket_batch_size(
            "auto", n_gal=40, n_data_per_gal=12, sampler="chees", n_chains=1, n_ensemble=8
        )
        assert out == 3
        assert seen == {"requested": "auto", "n_gal": 40, "n_data_per_gal": 12, "n_chains": 8}

    def test_an_integer_passes_through_without_consulting_the_budget(self, monkeypatch):
        from tengri.inference import batched_spectro
        from tengri.inference.batched_spectro import _resolve_bucket_batch_size

        def forbid(*args, **kwargs):
            raise AssertionError("an explicit integer must not consult the budget")

        monkeypatch.setattr(batched_spectro, "resolve_forward_chunk_size", forbid)
        out = _resolve_bucket_batch_size(
            5, n_gal=40, n_data_per_gal=12, sampler="nuts", n_chains=1, n_ensemble=8
        )
        assert out == 5

    def test_none_stays_none(self, monkeypatch):
        from tengri.inference import batched_spectro
        from tengri.inference.batched_spectro import _resolve_bucket_batch_size

        def forbid(*args, **kwargs):
            raise AssertionError("None must not consult the budget")

        monkeypatch.setattr(batched_spectro, "resolve_forward_chunk_size", forbid)
        out = _resolve_bucket_batch_size(
            None, n_gal=40, n_data_per_gal=12, sampler="nuts", n_chains=1, n_ensemble=8
        )
        assert out is None
