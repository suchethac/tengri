# SPDX-License-Identifier: BSD-3-Clause
"""A catalog result must account for every galaxy it was given.

Three distinct ways a galaxy fails to produce a usable posterior, and two of
them are invisible in a wall clock:

* **refused** -- ``DeadFitError`` before sampling (#2088). Loud, and the galaxy
  has no posterior at all.
* **silently frozen** -- the fit *returns*, with 100 % of its draws divergent or
  essentially no distinct positions (#2093, #1999). Split R-hat cannot fault it.
* **unconverged** -- it moved and did not mix.

The counts must be disjoint and must close against the catalog size. A
throughput number over a catalog that quietly shrank is not a throughput number,
which is the whole reason these are four separate integers rather than a
pass rate.

The last class of test here pins the thing R-hat alone cannot do: Phase 0
measured **73 % of galaxies clearing max split-R-hat < 1.01 with zero
divergences at a worst ESS of 2.63 of 500 draws**. Split R-hat compares two
equally badly-mixed halves and reads 1.00.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.catalog_convergence import (
    CATALOG_MAX_RHAT,
    FROZEN_DISTINCT_FRAC,
    catalog_convergence,
    galaxy_health,
)
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


def _posterior(samples, *, n_divergent=0, n_chains=1):
    """A minimal MCMC-shaped Posterior with no model, so every column is judged."""
    n_flat = len(next(iter(samples.values())))
    return Posterior(
        samples={k: np.asarray(v) for k, v in samples.items()},
        params={k: float(np.mean(v)) for k, v in samples.items()},
        method="test",
        wall_time_s=0.0,
        diagnostics={
            "n_divergent": n_divergent,
            "n_samples": n_flat // n_chains,
            "n_chains": n_chains,
        },
    )


def _healthy(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return _posterior({"a": rng.normal(size=n), "b": rng.normal(size=n)})


class TestTheFrozenSignatures:
    """Two independent signatures, either sufficient. Both are measured."""

    def test_a_hundred_percent_divergent_fit_is_frozen(self):
        """#2093's first signature. The fit *returns*; nothing raised."""
        rng = np.random.default_rng(0)
        post = _posterior({"a": rng.normal(size=1200)}, n_divergent=1200)
        health = galaxy_health(post)
        assert health.verdict == "frozen"
        assert health.divergence_rate == pytest.approx(1.0)

    def test_the_rate_uses_total_draws_not_n_samples(self):
        """#2087: ``n_samples`` is per chain, ``n_divergent`` is summed.

        A four-chain fit with 600 draws each and 2400 divergences is 100 %, not
        400 %. Dividing by ``n_samples`` would have reported 400 % and, worse,
        would never have hit exactly 1.0 for the frozen test above.
        """
        rng = np.random.default_rng(0)
        post = _posterior({"a": rng.normal(size=2400)}, n_divergent=2400, n_chains=4)
        assert galaxy_health(post).divergence_rate == pytest.approx(1.0)

    def test_a_low_distinct_fraction_is_frozen_even_with_zero_divergences(self):
        """#1999's signature: frozen, and the divergence count says nothing.

        The measured unique-draw fraction on #2093's fit is 0.002. This builds a
        chain at that fraction with ``n_divergent=0``, which is exactly the case
        a divergence-only trigger misses.
        """
        col = np.repeat(np.arange(2, dtype=float), 500)  # 2 distinct in 1000
        post = _posterior({"a": col}, n_divergent=0)
        health = galaxy_health(post)
        assert health.verdict == "frozen"
        assert health.distinct_frac <= FROZEN_DISTINCT_FRAC
        assert health.divergence_rate == 0.0

    def test_a_healthy_but_badly_mixed_chain_is_NOT_frozen(self):
        """The separator has to separate. A slow chain still moves every step.

        Building a genuinely autocorrelated chain (a random walk) rather than a
        white one: its ESS is far below its draw count, and its distinct
        fraction is still 1.0, which is the property that makes the threshold
        safe at two orders of magnitude below the worst healthy case.
        """
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.normal(scale=0.05, size=1000))
        health = galaxy_health(_posterior({"a": walk}))
        assert health.verdict != "frozen"
        assert health.distinct_frac == pytest.approx(1.0)


class TestTheCountsClose:
    def test_the_four_buckets_sum_to_the_catalog_size(self):
        """The identity that makes a dropped galaxy impossible to miss."""
        rng = np.random.default_rng(2)
        posteriors = [_healthy(seed=i) for i in range(3)]
        posteriors.append(_posterior({"a": np.zeros(400)}, n_divergent=0))  # frozen
        posteriors.append(_posterior({"a": np.cumsum(rng.normal(size=400))}))  # unconverged
        report = catalog_convergence(posteriors, refusals={9: "dead warmup"}, n_galaxies=6)

        assert report.n_galaxies == 6
        total = report.n_converged + report.n_unconverged + report.n_frozen + report.n_refused
        assert total == report.n_galaxies
        assert report.n_refused == 1
        assert report.n_frozen == 1

    def test_a_refusal_is_its_own_bucket_and_carries_no_draws(self):
        report = catalog_convergence([], refusals={0: "warmup ended dead"})
        assert report.n_refused == 1
        assert report.n_converged == report.n_frozen == report.n_unconverged == 0
        (row,) = report.per_galaxy
        assert row.verdict == "refused"
        assert row.max_rhat is None and row.min_ess is None

    def test_an_empty_catalog_reports_nothing_rather_than_a_rate(self):
        report = catalog_convergence([])
        assert report.frac_converged is None
        assert report.divergence_rate is None


class TestRHatIsNotACount:
    """The 73 %-at-ESS-2.63 finding, made structural."""

    def test_min_ess_is_reported_beside_every_converged_count(self):
        report = catalog_convergence([_healthy(seed=i) for i in range(3)])
        assert report.n_converged == 3
        assert report.min_ess_converged is not None
        assert "ESS" in report.summary()

    def test_the_summary_never_states_a_rate_without_its_ess(self):
        report = catalog_convergence([_healthy(seed=i) for i in range(3)])
        text = report.summary()
        assert "%" in text
        assert "worst\nESS" in text or "worst ESS" in text

    def test_min_ess_converged_is_a_different_number_from_min_ess(self):
        """The catalog minimum is set by the tail that already failed R-hat.

        Quoting it beside a converged rate compares two different populations,
        so the converged subset gets its own column.
        """
        rng = np.random.default_rng(3)
        good = [_healthy(seed=i) for i in range(3)]
        stuck = _posterior({"a": np.cumsum(rng.normal(scale=0.01, size=400)) + 100.0})
        report = catalog_convergence([*good, stuck])
        assert report.min_ess is not None and report.min_ess_converged is not None
        assert report.min_ess <= report.min_ess_converged

    def test_an_ess_floor_demotes_a_galaxy_that_passed_r_hat(self):
        """Opt-in, because the default must not quietly change a published count."""
        post = _healthy(seed=4)
        assert galaxy_health(post).verdict == "converged"
        huge = galaxy_health(post, ess_floor=1e9)
        assert huge.verdict == "unconverged"
        assert huge.max_rhat < CATALOG_MAX_RHAT

    def test_the_worst_parameter_is_named_not_only_its_value(self):
        """A catalog-wide "max R-hat 1.9" with no name cannot be acted on."""
        rng = np.random.default_rng(5)
        post = _posterior(
            {"fine": rng.normal(size=400), "stuck": np.cumsum(rng.normal(scale=0.01, size=400))}
        )
        health = galaxy_health(post)
        assert health.max_rhat_param in ("fine", "stuck")
        assert health.min_ess_param == "stuck"
