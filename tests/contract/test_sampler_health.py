# SPDX-License-Identifier: BSD-3-Clause
"""The shared driver judges convergence for every engine.

``run_sampler`` writes split R-hat, bulk ESS and the fraction of steps that moved into the
diagnostics (chain and ensemble samplers), the ELBO trace's relative change (variational
samplers) and the retained rank (low-rank HMC), and emits one warning per run naming what
failed. The estimators and thresholds are the repo's.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference._backend_registry import _BACKENDS, register_backend
from tengri.inference.sampling.driver import Sampler, SamplerResult, run_sampler
from tengri.inference.sampling.health import (
    chain_diagnostics,
    elbo_diagnostics,
    element_names,
    lowrank_diagnostics,
)
from tengri.inference.sampling.target import FlatTarget, LatentLayout

pytestmark = pytest.mark.contract


def _target(dim=3):
    return FlatTarget(
        lambda x, args: -0.5 * jnp.sum(x**2), LatentLayout.vector(dim), jnp.zeros(dim)
    )


@pytest.fixture
def stuck_sampler():
    """A sampler whose two chains sit at -3 and +3: deterministic, never mixing."""

    def run(target, key, n_chains, metric, cfg):
        n = cfg["n_samples"]
        wiggle = 0.01 * jnp.sin(jnp.arange(n))[None, :, None]
        base = jnp.where(jnp.arange(2) == 0, -3.0, 3.0)[:, None, None]
        return SamplerResult(base + wiggle + jnp.zeros((2, n, target.n_dim)), None, {})

    sampler = Sampler("stuck", "chain", run)
    register_backend("stuck", legacy_fitter=False, sampler=sampler)(run)
    yield sampler
    _BACKENDS.pop("stuck")


def test_a_stuck_pair_of_chains_triggers_one_warning_naming_the_parameter(stuck_sampler):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler("stuck", _target(), key=jax.random.PRNGKey(0), n_samples=200)
    health = [w for w in caught if "R-hat" in str(w.message)]
    assert len(health) == 1
    assert "'x[0]'" in str(health[0].message)
    assert result.diagnostics["rhat_max"] > 1.1
    assert result.diagnostics["convergence_problems"]


def test_the_warning_can_be_silenced_and_the_diagnostics_stay(stuck_sampler):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler(
            "stuck", _target(), key=jax.random.PRNGKey(0), n_samples=200, health_warnings=False
        )
    assert not [w for w in caught if "R-hat" in str(w.message)]
    assert result.diagnostics["rhat_max"] > 1.1


def test_a_healthy_chain_reports_its_diagnostics_without_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler(
            "mcmc_nuts", _target(), key=jax.random.PRNGKey(0), n_chains=4, n_warmup=200, n_samples=500
        )
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert result.diagnostics["rhat_max"] < 1.1
    assert result.diagnostics["ess_bulk_min"] > 100
    assert result.diagnostics["acceptance_rate"] > 0.5


def test_a_chain_that_never_moves_is_reported_as_collapsed():
    draws = np.zeros((1, 100, 2))
    draws[0, :, 1] = np.linspace(0, 1, 100)
    draws[0, ::50, 0] = 1.0  # moves on 4 of 99 steps
    record, problems = chain_diagnostics(draws, ["a", "b"])
    assert record["acceptance_rate"] > 0.05  # coordinate b moves every step
    still = np.zeros((1, 100, 1))
    still[0, 50:, 0] = 1.0
    record, problems = chain_diagnostics(still, ["a"])
    assert any("collapsed" in p for p in problems)


def test_an_unconverged_elbo_trace_is_flagged_and_a_flat_one_is_not():
    rising = np.linspace(0.0, 100.0, 1000)
    _, problems = elbo_diagnostics(rising)
    assert problems and "ELBO" in problems[0]
    record, problems = elbo_diagnostics(np.full(1000, 5.0) + 1e-6 * np.arange(1000))
    assert not problems and record["elbo_rel_change"] < 0.05


def test_a_truncating_low_rank_metric_is_flagged():
    record, problems = lowrank_diagnostics(rank=10, weakest_log_lambda=1.2, dim=22)
    assert record["lowrank_rank"] == 10 and record["lowrank_dim"] == 22
    assert problems and "rank 10 of D=22" in problems[0]
    assert not lowrank_diagnostics(10, 0.05, 22)[1]
    assert not lowrank_diagnostics(10, 1.2, 8)[1]


def test_element_names_follow_the_ravel_order():
    layout = LatentLayout.hierarchical(
        ("log_total_mass",), 2, 3, sigma_bounds=(0.1, 4.0), tau_bounds=(1.0, 300.0)
    )
    names = element_names(layout)
    assert len(names) == layout.n_dim
    assert names[0] == "gal/log_total_mass[0]"
    assert "psd_sigma_u" in names and "gal_xi[1,2]" in names


def test_a_chain_that_turns_to_nan_is_named_not_read_as_healthy():
    draws = np.random.default_rng(0).normal(size=(1, 300, 2))
    draws[0, 100:, 0] = np.nan
    _, problems = chain_diagnostics(draws, ["a", "b"])
    assert any("non-finite draws in 'a'" in p for p in problems), problems
    assert not any("'b'" in p and "non-finite" in p for p in problems)
    only_a, _ = chain_diagnostics(draws[..., :1], ["a"])
    assert only_a["acceptance_rate"] < 0.5  # moves only on the 99 finite steps of 299


def test_one_nan_column_among_finite_ones_is_named():
    draws = np.random.default_rng(1).normal(size=(2, 200, 3))
    draws[1, 50, 2] = np.inf
    _, problems = chain_diagnostics(draws, ["a", "b", "c"])
    assert any("non-finite draws in 'c'" in p for p in problems), problems
    assert not any(
        "non-finite draws in 'a'" in p or "non-finite draws in 'b'" in p for p in problems
    )


def test_a_constant_chain_is_collapsed_not_silent():
    record, problems = chain_diagnostics(np.ones((2, 100, 2)), ["a", "b"])
    assert record["acceptance_rate"] == 0.0
    assert any("collapsed" in p for p in problems), problems


def test_a_nan_step_does_not_count_as_a_move():
    draws = np.full((1, 100, 1), np.nan)
    record, problems = chain_diagnostics(draws, ["a"])
    assert record["acceptance_rate"] == 0.0
    assert any("non-finite draws in 'a'" in p for p in problems)


def test_one_stuck_chain_is_visible_through_the_per_chain_move_fraction():
    rng = np.random.default_rng(2)
    draws = np.stack([rng.normal(size=(300, 1)), np.full((300, 1), 0.5)])
    record, problems = chain_diagnostics(draws, ["a"])
    assert record["acceptance_rate"] > 0.4  # the pooled mean hides it
    assert record["acceptance_rate_min"] == 0.0
    assert any("collapsed" in p for p in problems), problems


def test_ess_is_the_sum_of_the_per_chain_ess():
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(4, 500, 1))
    record, _ = chain_diagnostics(draws, ["a"])
    # The repo's Sokal estimator is conservative on iid draws (about 0.55 of n), per chain.
    assert 0.4 * 2000 < record["ess_bulk_min"] <= 1.3 * 2000
    # Chains with different centers: the flattened series would read one long slow chain.
    shifted = draws + np.array([0.0, 5.0, 0.0, 5.0])[:, None, None]
    shifted_record, _ = chain_diagnostics(shifted, ["a"])
    np.testing.assert_allclose(shifted_record["ess_bulk_min"], record["ess_bulk_min"], rtol=1e-6)


def test_a_dead_chain_is_reported_whatever_its_divergence_flags_say():
    """Every draw identical and every step divergent: the driver names it."""
    dead = np.ones((2, 100, 2))
    record, problems = chain_diagnostics(dead, ["a", "b"], divergent=np.ones((2, 100), dtype=bool))
    assert record["divergent_fraction"] == 1.0
    assert any("divergent" in p for p in problems), problems
    assert any("collapsed" in p for p in problems), problems


def test_a_few_divergences_are_not_a_dead_chain():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(2, 300, 2))
    flags = np.zeros((2, 300), dtype=bool)
    flags[:, :6] = True
    record, problems = chain_diagnostics(draws, ["a", "b"], divergent=flags)
    assert record["divergent_fraction"] == 0.02
    assert not any("divergent" in p for p in problems), problems


@pytest.mark.parametrize("sampler", ["mcmc_nuts", "mcmc_hmc"])
def test_the_driver_reports_a_dead_chain_of_a_real_sampler(sampler):
    """A density that is finite at the start and walled off everywhere else: no step is taken."""
    start = jnp.zeros(2)

    def walled(x, args):
        return jnp.where(jnp.all(x == start), 0.0, -jnp.inf)

    target = FlatTarget(walled, LatentLayout.vector(2), start)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler(
            sampler,
            target,
            key=jax.random.PRNGKey(0),
            n_chains=2,
            n_warmup=20,
            n_burnin=0,
            n_samples=60,
            chain_jitter=0.0,
        )
    problems = result.diagnostics["convergence_problems"]
    assert any("collapsed" in p or "divergent" in p or "step size" in p for p in problems), (
        problems
    )
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)], "no warning raised"


def test_one_dead_galaxy_among_healthy_ones_is_named():
    from tengri.inference.sampling.health import catalog_diagnostics

    rng = np.random.default_rng(5)
    n_chains, n_samples = 2, 150
    positions = rng.normal(size=(3, n_chains * n_samples, 2))
    divergent = np.zeros((3, n_chains * n_samples), dtype=bool)
    positions[1] = 0.25
    divergent[1] = True
    records, flagged = catalog_diagnostics(positions, divergent, n_chains, n_samples, ["a", "b"])
    assert list(flagged) == [1], flagged
    assert records[1]["convergence_problems"] and not records[0]["convergence_problems"]
    assert not records[2]["convergence_problems"]


def test_the_chain_axis_of_a_galaxy_is_the_leading_one():
    """A bad chain among good ones is read as one chain, not smeared over the others."""
    from tengri.inference.sampling.health import catalog_diagnostics

    rng = np.random.default_rng(6)
    n_chains, n_samples = 3, 200
    positions = rng.normal(size=(1, n_chains * n_samples, 1))
    positions[0, n_samples : 2 * n_samples] = 7.0  # the second chain only
    records, flagged = catalog_diagnostics(
        positions, np.zeros((1, n_chains * n_samples), dtype=bool), n_chains, n_samples, ["a"]
    )
    assert flagged and records[0]["acceptance_rate_min"] == 0.0
