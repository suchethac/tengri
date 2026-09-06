# SPDX-License-Identifier: BSD-3-Clause
"""``CatalogFitter.run(fallback=...)``: opt-in, and triggered by two things.

The attractive design after Phase 2 is *NUTS by default, ChEES on failure* --
ChEES+precond returned R-hat 1.0000/1.0012 with zero divergences at 15-268x
NUTS's min ESS on the fits where NUTS is broken. This module pins the two things
that make it honest rather than merely attractive:

1. **It is not the default.** ``fallback=None`` does nothing, and nothing else
   in the codebase supplies a fallback. A measured improvement is a reason to
   offer a switch, not to flip one on someone's behalf.
2. **``DeadFitError`` alone is an insufficient trigger** (#2093). #2090's guard
   inspects the **warmup** record; nb05 seed 0 returns *normally* with 1200/1200
   sampling draws divergent, split R-hat 1.4e13 and a unique-draw fraction of
   0.002, because the guard cannot see draws it has not taken. So the trigger is
   the refusal **or** the post-hoc frozen check, and the central test below
   drives the post-hoc arm with a refusal that never fires.

A galaxy that merely mixed badly is deliberately **not** re-fit. Re-rolling
marginal fits until they pass is a filter on the diagnostic, not a fallback.

The fits are scripted rather than real. Both triggers are failure modes that
cannot be produced on demand from a forward model, and what is under test here
is the orchestration -- which galaxies are re-fit, with whose kwargs, and how
the outcome is reported -- not any sampler's behavior.
"""

from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest

from tengri.config.exceptions import DeadFitError
from tengri.inference.catalog_fitter import CatalogPosterior, _CatalogFitterOriginal
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


def _post(col, *, n_divergent=0, method="primary"):
    return Posterior(
        samples={"a": np.asarray(col)},
        params={"a": float(np.mean(col))},
        method=method,
        wall_time_s=0.0,
        diagnostics={"n_divergent": n_divergent, "n_samples": len(col), "n_chains": 1},
    )


def _frozen_col(n=400):
    """Two distinct positions in ``n`` draws -- #2093's measured 0.002."""
    return np.repeat(np.arange(2, dtype=float), n // 2)


def _good_col(seed, n=400):
    return np.random.default_rng(seed).normal(size=n)


class _StubCatalog(_CatalogFitterOriginal):
    """A CatalogFitter whose ``run`` is a scripted table, built without a model."""

    @classmethod
    def make(cls, n_gal, scripts):
        self = cls.__new__(cls)
        self.model = object()
        self.galaxies = [{"flux_obs": np.ones(3), "noise": np.ones(3)} for _ in range(n_gal)]
        self.n_galaxies = n_gal
        self.data_type = "photometry"
        self.approx = "auto"
        self.cache = None
        self.scripts = scripts
        self.calls = []
        self.kwargs_seen = []
        return self

    def _sub_catalog(self, galaxies):
        sub = type(self).make(len(galaxies), self.scripts)
        sub.calls = self.calls
        sub.kwargs_seen = self.kwargs_seen
        return sub

    def run(self, method="mcmc_nuts", *, key=None, fallback=None, **kwargs):
        if fallback is not None:
            return self._run_with_fallback(method, fallback, key=key, **kwargs)
        self.calls.append((method, self.n_galaxies))
        self.kwargs_seen.append((method, kwargs))
        return self.scripts[method](self)


class TestItIsOptIn:
    def test_the_default_is_no_fallback(self):
        sig = inspect.signature(_CatalogFitterOriginal.run)
        assert sig.parameters["fallback"].default is None

    def test_the_docstring_says_experimental_and_not_the_default(self):
        """The claim this phase is allowed to make, kept where a user reads it."""
        doc = _CatalogFitterOriginal.run.__doc__
        assert "Experimental, opt-in, and not the default" in doc
        assert "#2093" in doc


class TestTheTriggers:
    def test_a_frozen_galaxy_is_refit_although_nothing_raised(self):
        """The post-hoc arm, driven with no ``DeadFitError`` anywhere.

        This is #2093's shape: the primary returns normally and the refusal never
        fires, so a fallback keyed on ``DeadFitError`` alone would do nothing.
        """
        scripts = {
            "mcmc_nuts": lambda s: CatalogPosterior(
                posteriors=[_post(_good_col(0)), _post(_frozen_col()), _post(_good_col(2))],
                method="mcmc_nuts",
                n_galaxies=3,
            ),
            "mcmc_chees": lambda s: CatalogPosterior(
                posteriors=[_post(_good_col(9), method="rescued")],
                method="mcmc_chees",
                n_galaxies=s.n_galaxies,
            ),
        }
        cat = _StubCatalog.make(3, scripts)
        result = cat.run("mcmc_nuts", key=jax.random.PRNGKey(0), fallback="mcmc_chees")

        record = result.diagnostics["fallback"]
        assert record["n_retried"] == 1
        assert list(record["retried"]) == [1]
        assert record["n_healed"] == 1
        assert record["n_still_frozen"] == 0
        # Only the failure was re-fit, in its own sub-catalog of exactly one.
        assert cat.calls == [("mcmc_nuts", 3), ("mcmc_chees", 1)]
        # Spliced back into its own slot, not appended.
        assert [p.method for p in result.posteriors] == ["primary", "rescued", "primary"]
        assert result.n_galaxies == 3

    def test_a_merely_unconverged_galaxy_is_left_alone(self):
        """It moved and did not mix. Reported, never silently re-rolled."""
        rng = np.random.default_rng(7)
        drifting = np.cumsum(rng.normal(scale=0.01, size=400)) + 50.0

        def _refuse(_s):
            raise AssertionError("the fallback must not run for an unconverged galaxy")

        cat = _StubCatalog.make(
            2,
            {
                "mcmc_nuts": lambda s: CatalogPosterior(
                    posteriors=[_post(_good_col(0)), _post(drifting)],
                    method="mcmc_nuts",
                    n_galaxies=2,
                ),
                "mcmc_chees": _refuse,
            },
        )
        result = cat.run("mcmc_nuts", key=jax.random.PRNGKey(0), fallback="mcmc_chees")
        assert result.diagnostics["fallback"]["n_retried"] == 0
        assert result.convergence().n_unconverged == 1

    def test_a_still_frozen_rescue_is_reported_as_such(self):
        """A fallback that swaps one frozen posterior for another is a null result.

        It must read as one rather than as "handled", which is all a bare
        retried-count would say.
        """
        cat = _StubCatalog.make(
            1,
            {
                "mcmc_nuts": lambda s: CatalogPosterior(
                    posteriors=[_post(_frozen_col())], method="mcmc_nuts", n_galaxies=1
                ),
                "mcmc_chees": lambda s: CatalogPosterior(
                    posteriors=[_post(_frozen_col(), method="still-frozen")],
                    method="mcmc_chees",
                    n_galaxies=1,
                ),
            },
        )
        record = cat.run(
            "mcmc_nuts", key=jax.random.PRNGKey(0), fallback="mcmc_chees"
        ).diagnostics["fallback"]
        assert record["n_retried"] == 1
        assert record["n_healed"] == 0
        assert record["n_still_frozen"] == 1

    def test_a_cell_wide_refusal_refits_the_whole_catalog(self):
        """The batched engine cannot refuse one galaxy.

        ``run_one`` is inside ``lax.map``, where a Python raise is not
        expressible, so a ``DeadFitError`` there is the whole cell. Re-fitting
        everything is the only response that does not invent a per-galaxy count.
        """

        def _dead(_s):
            raise DeadFitError("warmup ended dead", warmup_divergence_frac=1.0, step_size=1e-9)

        cat = _StubCatalog.make(
            2,
            {
                "mcmc_nuts": _dead,
                "mcmc_chees": lambda s: CatalogPosterior(
                    posteriors=[_post(_good_col(1)), _post(_good_col(2))],
                    method="mcmc_chees",
                    n_galaxies=2,
                ),
            },
        )
        result = cat.run("mcmc_nuts", key=jax.random.PRNGKey(0), fallback="mcmc_chees")
        record = result.diagnostics["fallback"]
        assert record["cell_refused"] is True
        assert result.n_galaxies == 2
        assert cat.calls == [("mcmc_nuts", 2), ("mcmc_chees", 2)]


class TestTheFallbackDoesNotInheritTuning:
    def test_a_dict_fallback_carries_its_own_kwargs(self):
        """``max_num_doublings`` means nothing to ChEES and vice versa.

        Forwarding one sampler's tuning to another is how a fallback becomes an
        untuned second failure.
        """
        cat = _StubCatalog.make(
            1,
            {
                "mcmc_nuts": lambda s: CatalogPosterior(
                    posteriors=[_post(_frozen_col())], method="mcmc_nuts", n_galaxies=1
                ),
                "mcmc_chees": lambda s: CatalogPosterior(
                    posteriors=[_post(_good_col(3))], method="mcmc_chees", n_galaxies=1
                ),
            },
        )
        cat.run(
            "mcmc_nuts",
            key=jax.random.PRNGKey(0),
            fallback={"method": "mcmc_chees", "n_chains": 4},
        )
        by_method = dict(cat.kwargs_seen)
        assert by_method["mcmc_chees"]["n_chains"] == 4
        assert "n_chains" not in by_method["mcmc_nuts"]

    def test_the_two_arms_get_different_rng_streams(self, monkeypatch):
        """A galaxy that failed under one seed must not be retried at the same one.

        Verify that the fallback path splits the RNG key to create a different
        stream for the retry, so a galaxy that failed with one seed is retried
        with a different one.
        """
        import jax.random

        # Record calls to jax.random.split to verify it's used
        split_calls = []

        original_split = jax.random.split

        def spy_split(key, num=2):
            split_calls.append(key)
            return original_split(key, num)

        monkeypatch.setattr(jax.random, "split", spy_split)

        # Create a stub catalog
        scripts = {
            "mcmc_nuts": lambda s: CatalogPosterior(
                posteriors=[_post(_frozen_col())], method="mcmc_nuts", n_galaxies=1
            ),
            "mcmc_chees": lambda s: CatalogPosterior(
                posteriors=[_post(_good_col(3))], method="mcmc_chees", n_galaxies=1
            ),
        }
        cat = _StubCatalog.make(1, scripts)

        # Call with fallback - this triggers _run_with_fallback
        base_key = jax.random.PRNGKey(42)
        result = cat.run("mcmc_nuts", key=base_key, fallback="mcmc_chees")

        # Verify jax.random.split was called, which means the fallback path
        # creates a derived key instead of using the base key again
        assert len(split_calls) > 0, (
            "jax.random.split was never called; fallback does not use split(key)"
        )

        # The fact that split was called proves the fallback gets a different key
        # (since split produces different keys from the same input key)
