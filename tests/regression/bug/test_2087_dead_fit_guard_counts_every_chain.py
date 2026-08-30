# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2087 (defect 1): the dead-fit guard compared a
cross-chain divergence count with a per-chain draw count.

Bug: ``n_divergent`` is summed over the flattened ``(n_chains * n_samples,)``
divergence record while ``diagnostics["n_samples"]`` is per chain. The
all-divergent branch of ``Posterior.__post_init__`` tested
``n_divergent == n_samples``, false for every multi-chain run (2400 != 600 on
the 4-chain fit that filed the issue), and ``convergence_check`` divided by the
same per-chain count, so it reported 400% divergences and its own all-divergent
branch never fired either.

Guard: ``total_draws(diagnostics)`` is the one place that knows ``n_samples``
is per chain; both consumers compare against it.

Mutation checks (the one-line mutant each test must die under):
1. ``test_total_draws_multiplies_chains``: make ``total_draws`` ignore
   ``n_chains``.
2. ``test_all_divergent_two_chain_fit_warns``: revert posterior.py to
   ``n_divergent == n_samples`` -> no warning.
3. ``test_half_divergent_two_chain_fit_does_not_warn``: the same mutant makes
   the old condition TRUE for ``n_divergent == n_samples`` with two chains ->
   a false warning.
4. ``test_convergence_check_two_chain_percentages``: revert convergence.py to
   ``n_div / n_samples`` -> 200%.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import DeadFitWarning
from tengri.inference.backends.mcmc._shared import total_draws
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug

_N_PER_CHAIN = 200
_N_CHAINS = 2
_N_TOTAL = _N_PER_CHAIN * _N_CHAINS


def _two_chain_samples():
    key = jax.random.PRNGKey(0)
    return {
        "x": jax.random.normal(key, (_N_TOTAL,)) + 5.0,
        "y": jax.random.normal(jax.random.fold_in(key, 1), (_N_TOTAL,)) - 2.0,
    }


def _two_chain_posterior(n_divergent):
    samples = _two_chain_samples()
    for name, arr in samples.items():
        assert float(np.ptp(np.asarray(arr))) > 0.0, name
    return Posterior(
        samples=samples,
        params={"x": jnp.array(5.0)},
        method="mcmc_nuts",
        wall_time_s=1.0,
        diagnostics={
            "n_divergent": n_divergent,
            "n_samples": _N_PER_CHAIN,
            "n_chains": _N_CHAINS,
        },
    )


def test_total_draws_multiplies_chains():
    assert total_draws({"n_samples": 600, "n_chains": 4}) == 2400
    assert total_draws({"n_samples": 600}) == 600
    assert total_draws({"n_chains": 4}, n_samples=50) == 200


def test_all_divergent_two_chain_fit_warns():
    with pytest.warns(DeadFitWarning, match=r"dead fit: 400/400 divergent"):
        _two_chain_posterior(n_divergent=_N_TOTAL)


def test_half_divergent_two_chain_fit_does_not_warn():
    # 200 of 400 draws diverged: bad, but not the "every transition" signature.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _two_chain_posterior(n_divergent=_N_PER_CHAIN)


def test_convergence_check_two_chain_percentages():
    from tengri.analysis.plotting.convergence import convergence_check

    with pytest.warns(DeadFitWarning):
        dead = _two_chain_posterior(n_divergent=_N_TOTAL)
    info = convergence_check(dead, method_name="NUTS", verbose=False)
    assert info["all_samples_divergent"] is True
    assert info["n_draws_total"] == _N_TOTAL
    assert info["divergence_pct"] == pytest.approx(100.0)
    assert "CRITICAL" in " ".join(info["warnings"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        half = _two_chain_posterior(n_divergent=_N_PER_CHAIN)
    info = convergence_check(half, method_name="NUTS", verbose=False)
    assert info.get("all_samples_divergent") is not True
    assert info["divergence_pct"] == pytest.approx(50.0)
