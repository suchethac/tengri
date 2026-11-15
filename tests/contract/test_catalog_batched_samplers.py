# SPDX-License-Identifier: BSD-3-Clause
"""The batched catalog path: which samplers reach it, and on what convention.

``CatalogFitter._MCMC_VMAPPABLE`` decides whether a sampler runs K galaxies per
``lax.map`` step on the accelerator or falls to ``_run_sequential`` and never
reaches the GPU at all. Three things about that set are load-bearing enough to
pin:

1. **Membership is not the tier.** ``mcmc_chees`` is on the batched path *and*
   stays ``tier="experimental"``; ``mcmc_ghmc`` and ``mcmc_mclmc`` are
   ``tier="broken"`` and must stay off it, because a backend that reports wrong
   answers reports them faster here.
2. **The adaptation convention.** Three incompatible conventions exist in this
   codebase and conflating them is how a galaxy's posterior starts depending on
   which galaxies shared its batch. The catalog path picks per-galaxy adaptation,
   with ChEES's ensemble as an *inner* axis, and that choice is pinned here
   rather than left in a docstring nobody re-reads.
3. **Compile cost stays O(1) in N.** The binding contract in
   ``docs/internal/specs/2026-07-23-inference-prediction-api-final.md``, and the
   reason the whole batched path exists.

These are contract tests: they assert the shape of the seam, not the quality of
any posterior. Convergence is measured in ``bench/reports/``, not asserted here.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _gauss_ld(x, data_args):
    """Standard-normal-shaped log-density centered on ``data_args["mu"]``."""
    return -0.5 * jnp.sum((x - data_args["mu"]) ** 2)


def _gauss_substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err, shared):
    """Per-galaxy ``data_args``: the galaxy's center only."""
    return {"mu": data}


def _gauss_flat(dim):
    """The four-tuple ``flat_logdensity`` seam for a ``dim``-dimensional Gaussian target."""
    return (_gauss_ld, lambda x: x, jnp.zeros(dim), {"mu": jnp.zeros(dim)})


# --------------------------------------------------------------------------
# 1. Membership
# --------------------------------------------------------------------------


class TestWhichSamplersReachTheBatchedPath:
    def test_chees_is_on_the_batched_path(self):
        """The Phase 3 change, stated once as a fact rather than inferred."""
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        assert "mcmc_chees" in _CatalogFitterOriginal._MCMC_VMAPPABLE

    def test_the_quarantined_samplers_are_not(self):
        """A ``tier="broken"`` backend reports wrong answers *faster* here.

        ``mcmc_ghmc`` measured split-R-hat 1.1e10 with acceptance at 0.989 and
        ``mcmc_mclmc`` is quarantined on its energy error. Putting either on the
        batched path would hand a catalog user a fast wrong answer, which is
        strictly worse than a slow one.
        """
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        for name in ("mcmc_ghmc", "mcmc_mclmc"):
            assert name not in _CatalogFitterOriginal._MCMC_VMAPPABLE

    def test_chees_did_not_get_promoted_on_the_way_in(self):
        """Reaching the batched path is a structural property, not a tier bump."""
        from tengri.inference._backend_registry import get_backend

        assert get_backend("mcmc_chees").tier == "experimental"

    def test_every_vmappable_method_has_a_sampler_tag(self):
        """A method in the set with no engine tag is a ``KeyError`` at fit time.

        The dispatch used to read ``"nuts" if method == "mcmc_nuts" else "hmc"``,
        which silently ran HMC for anything else added to the set. A table that
        must be complete is better than a fallback that cannot be wrong.
        """
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal as C

        assert set(C._MCMC_VMAPPABLE) == set(C._MCMC_SAMPLER_TAG)

    def test_the_engine_agrees_about_which_samplers_exist(self):
        from tengri.inference.backends.mcmc.catalog import _SAMPLERS
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal as C

        assert set(C._MCMC_SAMPLER_TAG.values()) <= set(_SAMPLERS)


# --------------------------------------------------------------------------
# 2. The adaptation convention
# --------------------------------------------------------------------------


class TestTheAdaptationConvention:
    """Per galaxy, with ChEES's ensemble inside it -- never across galaxies."""

    def test_the_ensemble_axis_is_chains_within_galaxy(self):
        """LOAD-BEARING, and the reason a catalog ChEES fit is reproducible.

        An ensemble spanning *galaxies* would tune one trajectory length against
        a mixture of posteriors, and each galaxy's draws would then depend on
        which galaxies happened to share its batch. The resolver that enforces
        this is shared with the single-fit path, so the two cannot disagree.
        """
        from tengri.inference.backends.mcmc._shared import _resolve_chees_ensemble

        doc = _resolve_chees_ensemble.__doc__
        assert "chains-within-galaxy" in doc
        assert "galaxies-within-batch" in doc

    def test_the_catalog_default_ensemble_is_a_legal_ensemble(self):
        """Smaller than the single-fit 32 for VRAM, still above the floor.

        The ensemble is an inner axis under the galaxy vmap, so a cell at
        ``forward_chunk_size=K`` carries ``K * n_ensemble`` live chains. Below
        ``_CHEES_MIN_ENSEMBLE`` the cross-chain centered positions ChEES
        differentiates collapse toward zero and the trajectory length stops
        adapting -- silently, which is why the floor is enforced rather than
        documented.
        """
        from tengri.inference.backends.mcmc._shared import (
            _CHEES_DEFAULT_ENSEMBLE,
            _CHEES_MIN_ENSEMBLE,
            _resolve_chees_ensemble,
        )
        from tengri.inference.backends.mcmc.catalog import CATALOG_CHEES_ENSEMBLE

        assert CATALOG_CHEES_ENSEMBLE >= _CHEES_MIN_ENSEMBLE
        assert CATALOG_CHEES_ENSEMBLE < _CHEES_DEFAULT_ENSEMBLE
        assert _resolve_chees_ensemble(CATALOG_CHEES_ENSEMBLE, 1) == CATALOG_CHEES_ENSEMBLE

    def test_an_undersized_catalog_ensemble_is_refused_not_clamped(self):
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

        with pytest.raises(ValueError, match="n_ensemble"):
            build_catalog_mcmc_engine(
                object(), "chees", n_warmup=10, n_burnin=0, n_samples=10, n_ensemble=1
            )

    @pytest.mark.parametrize("sampler", ["nuts", "hmc"])
    def test_window_samplers_run_several_chains_from_one_adaptation(self, sampler):
        """NUTS/HMC adapt once per galaxy and sample several overdispersed chains from it.

        The output contract matches the ChEES path: the chains are flattened to
        ``(n_chains * n_samples, D)``. The chains must differ, or the split R-hat
        over them measures nothing.
        """
        import jax.numpy as jnp
        import numpy as np

        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

        n_chains, n_samples, dim = 2, 12, 2
        run_one, _ = build_catalog_mcmc_engine(
            None,
            sampler,
            n_warmup=40,
            n_burnin=5,
            n_samples=n_samples,
            n_chains=n_chains,
            flat_logdensity=_gauss_flat(dim),
            substitute=_gauss_substitute,
        )
        positions, divergent = run_one(
            jnp.zeros(dim),
            jax.random.PRNGKey(7),
            jnp.ones(dim),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        assert positions.shape == (n_chains * n_samples, dim)
        assert divergent.shape == (n_chains * n_samples,)
        assert bool(jnp.all(jnp.isfinite(positions)))
        chains = np.asarray(positions).reshape(n_chains, n_samples, dim)
        assert not np.allclose(chains[0], chains[1])

    def test_single_window_chain_is_the_direct_run_chain_program(self):
        """``n_chains=1`` keeps the one-chain program, bit for bit.

        The multi-chain form must not perturb the existing single-chain result:
        the draws equal a direct ``run_chain`` call on the same key split.
        """
        import jax.numpy as jnp
        import numpy as np

        from tengri.inference._backend_registry import get_backend
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
        from tengri.inference.backends.mcmc.chain_program import NUTSConfig, run_chain

        n_warmup, n_burnin, n_samples, dim = 40, 5, 12, 2
        n_chain = n_burnin + n_samples
        run_one, _ = build_catalog_mcmc_engine(
            None,
            "nuts",
            n_warmup=n_warmup,
            n_burnin=n_burnin,
            n_samples=n_samples,
            flat_logdensity=_gauss_flat(dim),
            substitute=_gauss_substitute,
        )
        gal_key = jax.random.PRNGKey(11)
        data = jnp.ones(dim)
        positions, _ = run_one(jnp.zeros(dim), gal_key, data, None, None, None, None, None, None)

        warmup_key, chain_key = jax.random.split(gal_key)
        chain_keys = jax.random.split(chain_key, n_chain)
        entry = get_backend("mcmc_nuts")
        adaptation = entry.adaptation_for({"pathfinder_warmstart": False})
        cfg = NUTSConfig(max_doublings=10, use_dense=False, target_accept_rate=0.85)
        direct, _ = run_chain(
            entry.chain_program,
            adaptation,
            jnp.zeros(dim),
            warmup_key,
            None,
            chain_keys,
            _gauss_ld,
            {"mu": data},
            n_warmup,
            cfg,
        )
        np.testing.assert_array_equal(
            np.asarray(positions), np.asarray(direct.position[n_burnin:])
        )

    def test_the_catalog_docstring_names_all_three_conventions(self):
        """The rejected alternatives are recorded where the choice was made.

        ``Fitter._fit_batch_vmap_mcmc`` adapts once on the first galaxy and
        reuses it; that is the convention this path rejects, and a reader who
        does not find it named here will eventually re-propose it.
        """
        from tengri.inference.backends.mcmc import catalog as mod

        doc = mod.__doc__
        assert "_fit_batch_vmap_mcmc" in doc
        assert "chains-within-galaxy" in doc
        assert "Per-galaxy adaptation inside the vmap" in doc

    def test_chees_does_not_inherit_nutss_acceptance_target(self):
        """0.651, not 0.85. Carrying the NUTS value across would be invisible.

        Each ChEES step is a *fixed*-length HMC proposal, whose asymptotically
        optimal acceptance rate is 0.651 (Beskos+2013). The sampler would still
        run at 0.85 -- just with a step size dual-averaged for a different
        proposal -- so the default is resolved per sampler from an ``AUTO``
        sentinel rather than from a shared literal.
        """
        from tengri.inference._batching import AUTO
        from tengri.inference.backends.mcmc.chees import CHEES_TARGET_ACCEPT_RATE
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        sig = inspect.signature(_CatalogFitterOriginal._run_native_mcmc)
        assert sig.parameters["target_accept_rate"].default is AUTO
        assert pytest.approx(0.651) == CHEES_TARGET_ACCEPT_RATE

    def test_chees_dense_metric_changes_the_draws_and_stays_finite(self):
        """``chees_dense`` is applied, not silently ignored.

        The dense metric is a linear change of variables estimated from a pilot run, so
        the same key must give different draws with and without it. Both must be finite
        and of the shared chain-flattened shape.
        """
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

        dim, n_samples = 2, 12
        kwargs = dict(
            n_warmup=40,
            n_burnin=5,
            n_samples=n_samples,
            n_chains=4,
            n_ensemble=4,
            flat_logdensity=_gauss_flat(dim),
            substitute=_gauss_substitute,
        )
        args = (
            jnp.zeros(dim),
            jax.random.PRNGKey(3),
            jnp.ones(dim),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        diagonal_run, _ = build_catalog_mcmc_engine(None, "chees", **kwargs)
        dense_run, _ = build_catalog_mcmc_engine(None, "chees", chees_dense=True, **kwargs)
        diagonal, _ = diagonal_run(*args)
        dense, _ = dense_run(*args)
        assert dense.shape == (4 * n_samples, dim)
        assert bool(jnp.all(jnp.isfinite(dense)))
        assert not bool(jnp.allclose(diagonal, dense))


# --------------------------------------------------------------------------
# 3. The tree-depth cap reaches warmup
# --------------------------------------------------------------------------


class TestTheTreeDepthCapReachesWarmup:
    """#2093 / Phase 3: the cap was applied to the cheap half only.

    BlackJAX's window adaptation runs its **own** NUTS kernel and forwards
    ``**extra_parameters`` into it. Omitting ``max_num_doublings`` left warmup on
    BlackJAX's default of 10 while only the sampling scan honored the caller's
    number -- and warmup is where trees are deepest, because the step size has
    not converged yet. Measured at K = 1 on the throughput fixture: capping the
    sampling half alone took 50 draws from 19 s to 0.1 s and left the 50 warmup
    steps at 36 s; capping both took the whole cell to 2.1 s.
    """

    def test_run_chain_forwards_the_cap_to_the_adaptation(self, monkeypatch):
        """The max tree-depth cap VALUE reaches window_adaptation, not just sampling.

        If it only capped the sampling kernel (not warmup), warmup would use
        BlackJAX's default of 10 doublings (expensive) while sampling uses the
        caller's cap. This test verifies the cap VALUE reaches the adaptation.
        """

        from tengri.inference.backends.mcmc.chain_program import (
            NUTS,
            WINDOW_NUTS,
            NUTSConfig,
            run_chain,
        )

        # Recording stub: capture what gets passed to window_adaptation
        class _Reached(Exception):
            pass

        captured_kwargs = {}

        try:
            import blackjax

            original_adapt = blackjax.window_adaptation
        except ImportError:
            # If blackjax not available or doesn't have window_adaptation,
            # skip this test
            pytest.skip("blackjax.window_adaptation not available")
            return

        def record_adapt(*args, **kwargs):
            captured_kwargs.update(kwargs)
            raise _Reached("window_adaptation_called")

        monkeypatch.setattr(blackjax, "window_adaptation", record_adapt)

        # Distinctive value that won't be a default
        sentinel_cap = 7

        try:
            # Call run_chain with the distinctive cap value
            # (This will fail when it hits our stub, but that's ok)
            import jax
            import jax.random
            import numpy as np

            init_flat = np.array([0.0, 0.0])
            key = jax.random.PRNGKey(0)
            chain_keys = jax.random.split(key, 4)

            def dummy_logdensity(x):
                return -0.5 * np.sum(x**2)

            def dummy_logdensity_2arg(x, _data):
                return dummy_logdensity(x)

            run_chain(
                NUTS,
                WINDOW_NUTS,
                init_flat,
                key,
                None,
                chain_keys,
                dummy_logdensity_2arg,
                {},
                10,
                NUTSConfig(max_doublings=sentinel_cap, use_dense=False, target_accept_rate=0.85),
            )
        except _Reached:
            # Expected: the stub raises once it has recorded the cap.
            pass
        # Deliberately no blanket ``except Exception``. If the scan fails before
        # reaching the adaptation, that error is the diagnosis; swallowing it
        # leaves only "did not reach window_adaptation", which is true of a
        # dropped cap and of an unrelated crash alike.

        # Critical assertion: max_doublings VALUE reached the adaptation
        assert "max_num_doublings" in captured_kwargs or "max_doublings" in captured_kwargs, (
            "max_doublings did not reach window_adaptation"
        )

        cap_arg = captured_kwargs.get("max_num_doublings") or captured_kwargs.get("max_doublings")
        assert cap_arg is sentinel_cap, (
            f"max_doublings={cap_arg} != {sentinel_cap}; cap dropped or default used"
        )

    def test_the_warmup_config_carries_the_cap(self):
        import dataclasses

        from tengri.inference.backends.mcmc._shared import DEFAULT_MAX_NUM_DOUBLINGS
        from tengri.inference.backends.mcmc.chain_program import NUTSConfig

        # Defaulting to BlackJAX's own 10 is what makes this change a no-op for
        # a caller who never asked for a cap.
        assert "max_doublings" in {field.name for field in dataclasses.fields(NUTSConfig)}
        cfg = NUTSConfig(
            max_doublings=DEFAULT_MAX_NUM_DOUBLINGS, use_dense=False, target_accept_rate=0.85
        )
        assert cfg.max_doublings == 10

    def test_the_cap_is_in_the_adaptation_cache_key(self):
        """A step size adapted under one cap is not the one another would find.

        Without this the second fit of a model at a different cap would reuse the
        first cap's step size, and nothing would say so.
        """
        from tengri.inference.backends.mcmc import nuts

        # ``tuning`` used to be a hand-built tuple literal (``tuning = (int(n_warmup),
        # ...)``); it is now built by ``adaptation_method_key(...)`` (#2163 E.5), so
        # the span is bounded by "tuning = " (not "tuning = (") through "adapt_key = (".
        src = inspect.getsource(nuts.run_nuts)
        tuning = src[src.index("    tuning = ") : src.index("adapt_key = (")]
        assert "max_num_doublings" in tuning


def test_stuck_pilot_falls_back_to_its_own_draws(monkeypatch):
    """A pilot that returns constant draws cannot define a metric; the galaxy keeps them.

    The constant pilot has a zero covariance, so the Cholesky factor is not finite. The
    galaxy must then return exactly the pilot's post-burn-in draws, all finite, and not
    NaN. The pilot is replaced by a stub so that the stuck state is exact; the final pass
    is the real ChEES scan and must still run.
    """
    from tengri.inference.backends.mcmc import catalog as catalog_mod
    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

    real_scan = catalog_mod._chees_scan
    dim, n_burnin, n_samples, n_chains, pilot_value = 2, 5, 12, 4, 0.5
    n_chain = n_burnin + n_samples
    calls = []

    def stuck_pilot_then_real(*args):
        calls.append(None)
        if len(calls) > 1:
            return real_scan(*args)
        n_chains_arg, n_chain_arg = args[7], args[8]
        positions = jnp.full((n_chains_arg, n_chain_arg, dim), pilot_value)
        divergent = jnp.zeros((n_chains_arg, n_chain_arg), dtype=bool)
        return positions, divergent, None, None, None

    monkeypatch.setattr(catalog_mod, "_chees_scan", stuck_pilot_then_real)
    run_one, _ = build_catalog_mcmc_engine(
        None,
        "chees",
        n_warmup=40,
        n_burnin=n_burnin,
        n_samples=n_samples,
        n_chains=n_chains,
        n_ensemble=4,
        chees_dense=True,
        flat_logdensity=_gauss_flat(dim),
        substitute=_gauss_substitute,
    )
    positions, divergent = run_one(
        jnp.zeros(dim), jax.random.PRNGKey(9), jnp.ones(dim), None, None, None, None, None, None
    )

    assert len(calls) == 2, "the dense final pass must still be attempted"
    assert positions.shape == (n_chains * n_samples, dim)
    assert bool(jnp.all(jnp.isfinite(positions)))
    np.testing.assert_array_equal(
        np.asarray(positions), np.full((n_chains * n_samples, dim), pilot_value)
    )
    assert not bool(jnp.any(divergent))


def _split_rhat(chains):
    """Gelman-Rubin R-hat per coordinate for ``chains`` of shape (n_chains, n_draws, D)."""
    n_draws = chains.shape[1]
    means = chains.mean(axis=1)
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = n_draws * means.var(axis=0, ddof=1)
    var_hat = (n_draws - 1) / n_draws * within + between / n_draws
    return np.sqrt(var_hat / within)


def test_window_chains_start_at_distinct_overdispersed_positions():
    """Every chain starts somewhere of its own, at about the requested jitter scale."""
    from tengri.inference.backends.mcmc.chain_program import _jittered_starts

    starts = np.asarray(_jittered_starts(jnp.zeros(2), jax.random.PRNGKey(0), 4, 0.1))
    assert starts.shape == (4, 2)
    assert len({tuple(np.round(row, 12)) for row in starts}) == 4
    assert 0.02 < starts.std() < 0.3, f"start spread {starts.std():.3f} is not ~0.1"


@pytest.mark.parametrize("sampler", ["nuts", "hmc"])
def test_window_chains_mix_on_a_well_posed_target(sampler):
    """Four chains from one adaptation agree on the target: split R-hat below 1.1."""
    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

    n_chains, n_samples = 4, 300
    run_one, _ = build_catalog_mcmc_engine(
        None,
        sampler,
        n_warmup=200,
        n_burnin=50,
        n_samples=n_samples,
        n_chains=n_chains,
        flat_logdensity=_gauss_flat(2),
        substitute=_gauss_substitute,
    )
    positions, _ = run_one(
        jnp.zeros(2), jax.random.PRNGKey(5), jnp.ones(2), None, None, None, None, None, None
    )
    chains = np.asarray(positions).reshape(n_chains, n_samples, 2)
    rhat = _split_rhat(chains)
    assert np.all(rhat < 1.1), rhat


def test_single_chain_matches_the_golden_array_from_before_multichain():
    """``n_chains=1`` is bit-identical to the array the pre-multichain engine produced.

    The golden was generated at the parent of the first multi-chain commit, with the same
    key, target and counts, and is pinned as a file so the single-chain path cannot drift
    without the change being visible.
    """
    from pathlib import Path

    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

    golden = np.load(Path(__file__).parent / "data" / "catalog_nuts_single_chain_golden.npy")
    run_one, _ = build_catalog_mcmc_engine(
        None,
        "nuts",
        n_warmup=40,
        n_burnin=5,
        n_samples=12,
        flat_logdensity=_gauss_flat(2),
        substitute=_gauss_substitute,
    )
    positions, _ = run_one(
        jnp.zeros(2), jax.random.PRNGKey(11), jnp.ones(2), None, None, None, None, None, None
    )
    np.testing.assert_array_equal(np.asarray(positions), golden)


_RHO = 0.95
_CORR_COV = np.array([[1.0, _RHO], [_RHO, 1.0]])
_CORR_PREC = np.linalg.inv(_CORR_COV)


def _corr_ld(x, data_args):
    """Log-density of a 2-D Gaussian with unit variances and correlation 0.95."""
    return -0.5 * x @ jnp.asarray(_CORR_PREC) @ x


def _corr_flat():
    return (_corr_ld, lambda x: x, jnp.zeros(2), {"mu": jnp.zeros(2)})


def _ess_1d(chains):
    """Effective sample size of (n_chains, n_draws) chains, pooled, truncated at the first
    negative autocorrelation."""
    m, n = chains.shape
    centered = chains - chains.mean()
    var = (centered**2).mean()
    rho_sum = 0.0
    for lag in range(1, n // 2):
        rho = np.mean(centered[:, :-lag] * centered[:, lag:]) / var
        if rho < 0:
            break
        rho_sum += rho
    return m * n / (1.0 + 2.0 * rho_sum)


def test_chees_dense_recovers_a_correlated_gaussian_and_mixes_better_than_diagonal():
    """Dense ChEES recovers rho=0.95 and resolves the correlated direction better.

    Tolerances: about 2000 draws at ESS of several hundred give a standard error near
    0.06 on a variance and near 0.004 on a correlation, (1-rho^2)/sqrt(ESS) at rho=0.95,
    so 0.2 and 0.03 are roughly 3 and 7 standard errors. The mixing comparison is on the
    projection onto (1,1)/sqrt(2), where the diagonal metric cannot help: the dense run
    must have at least 1.5 times the diagonal run's ESS.
    """
    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

    n_chains, n_samples = 4, 500
    kwargs = dict(
        n_warmup=300,
        n_burnin=100,
        n_samples=n_samples,
        n_chains=n_chains,
        n_ensemble=8,
        max_leapfrog_steps=64,
        flat_logdensity=_corr_flat(),
        substitute=_gauss_substitute,
    )
    args = (jnp.zeros(2), jax.random.PRNGKey(21), jnp.ones(2), None, None, None, None, None, None)
    dense_run, _ = build_catalog_mcmc_engine(None, "chees", chees_dense=True, **kwargs)
    diag_run, _ = build_catalog_mcmc_engine(None, "chees", **kwargs)
    dense, _ = dense_run(*args)
    diagonal, _ = diag_run(*args)
    dense = np.asarray(dense)
    assert np.all(np.isfinite(dense))

    draws = dense.reshape(n_chains, n_samples, 2)
    cov = np.cov(draws.reshape(-1, 2), rowvar=False)
    np.testing.assert_allclose(np.diag(cov), [1.0, 1.0], atol=0.2)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(corr - _RHO) < 0.03, f"sample correlation {corr:.4f} vs {_RHO}"

    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    ess_dense = _ess_1d(draws @ direction)
    ess_diag = _ess_1d(np.asarray(diagonal).reshape(n_chains, n_samples, 2) @ direction)
    assert ess_dense >= 1.5 * ess_diag, f"ESS dense {ess_dense:.1f} vs diagonal {ess_diag:.1f}"
