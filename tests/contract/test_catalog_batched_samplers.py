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

import pytest

pytestmark = pytest.mark.contract


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

    def test_window_adaptation_samplers_refuse_multiple_chains_by_name(self):
        """NUTS/HMC adapt per galaxy, so a second chain would re-run warmup.

        Refused rather than ignored: a caller who asked for four chains and got
        one would have no way to find out, and would read a single chain's split
        R-hat as a four-chain one.
        """
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

        for sampler in ("nuts", "hmc"):
            with pytest.raises(ValueError, match="n_chains"):
                build_catalog_mcmc_engine(
                    object(), sampler, n_warmup=10, n_burnin=0, n_samples=10, n_chains=4
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

    def test_chees_refuses_a_dense_mass_matrix_by_name(self):
        """Its kernel metric is diagonal; a silently-ignored True is worse.

        This is the failure ``run_ghmc``'s ``target_accept_rate`` records: an
        argument the surface accepts and the kernel cannot use.
        """
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        cat = _CatalogFitterOriginal.__new__(_CatalogFitterOriginal)
        with pytest.raises(ValueError, match="dense_mass_matrix"):
            _CatalogFitterOriginal._run_native_mcmc(
                cat, "mcmc_chees", key=None, dense_mass_matrix=True
            )


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

    def test_the_full_scan_forwards_the_cap_to_the_adaptation(self):
        from tengri.inference.backends.mcmc import _shared

        src = inspect.getsource(_shared._nuts_full_scan)
        assert "max_num_doublings" in src, (
            "the cap must reach blackjax.window_adaptation, not only the "
            "sampling kernel; see Finding 3 of the 2026-08-30 GPU report"
        )

    def test_the_warmup_only_entry_point_takes_the_cap(self):
        from tengri.inference.backends.mcmc._shared import (
            DEFAULT_MAX_NUM_DOUBLINGS,
            _nuts_warmup_only,
        )

        sig = inspect.signature(_nuts_warmup_only.__wrapped__)
        assert "max_doublings" in sig.parameters
        # Defaulting to BlackJAX's own 10 is what makes this change a no-op for
        # a caller who never asked for a cap.
        assert sig.parameters["max_doublings"].default == DEFAULT_MAX_NUM_DOUBLINGS
        assert DEFAULT_MAX_NUM_DOUBLINGS == 10

    def test_the_cap_is_in_the_adaptation_cache_key(self):
        """A step size adapted under one cap is not the one another would find.

        Without this the second fit of a model at a different cap would reuse the
        first cap's step size, and nothing would say so.
        """
        from tengri.inference.backends.mcmc import nuts

        src = inspect.getsource(nuts.run_nuts)
        tuning = src[src.index("    tuning = (") : src.index("adapt_key = (")]
        assert "max_num_doublings" in tuning
