# SPDX-License-Identifier: BSD-3-Clause
"""Nested R-hat and tail ESS follow the Margossian et al. and Vehtari et al. definitions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.analysis.diagnostics.autocorrelation import nested_rhat, tail_ess

pytestmark = pytest.mark.contract

N_SUPERCHAINS = 8
N_CHAINS = 16
N_DRAWS = 50


def _gaussian_draws(seed: int, n_chains: int = N_CHAINS, n_draws: int = N_DRAWS) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_chains, n_draws))


def test_nested_rhat_is_slightly_above_one_for_short_mixed_chains():
    draws = jnp.asarray(_gaussian_draws(seed=1))
    value = float(nested_rhat(draws, N_SUPERCHAINS))
    assert 1.0 <= value < 1.02


def test_nested_rhat_is_one_for_long_mixed_gaussian_chains():
    draws = jnp.asarray(_gaussian_draws(seed=1, n_draws=500))
    assert abs(float(nested_rhat(draws, N_SUPERCHAINS)) - 1.0) < 0.01


def test_rank_normalized_nested_rhat_invariant_to_monotone_transform():
    draws = jnp.asarray(_gaussian_draws(seed=3))
    base = float(nested_rhat(draws, N_SUPERCHAINS, fold=False))
    transformed = float(nested_rhat(jnp.expm1(draws), N_SUPERCHAINS, fold=False))
    assert transformed == pytest.approx(base, abs=1e-10)


def test_nested_rhat_detects_superchains_offset_in_mean():
    draws = _gaussian_draws(seed=2)
    chains_per_superchain = N_CHAINS // N_SUPERCHAINS
    for s in range(N_SUPERCHAINS):
        rows = slice(s * chains_per_superchain, (s + 1) * chains_per_superchain)
        draws[rows] += float(s)
    assert float(nested_rhat(jnp.asarray(draws), N_SUPERCHAINS)) > 1.1


def test_tail_ess_of_iid_draws_matches_draw_count():
    n_chains, n_draws = 4, 2000
    draws = _gaussian_draws(seed=4, n_chains=n_chains, n_draws=n_draws)
    ess = tail_ess(draws)
    total = n_chains * n_draws
    assert abs(ess - total) / total < 0.2


def test_n_chains_not_multiple_of_n_superchains_names_both_numbers():
    draws = jnp.asarray(_gaussian_draws(seed=5, n_chains=10))
    with pytest.raises(ValueError, match=r"10.*8|8.*10"):
        nested_rhat(draws, N_SUPERCHAINS)


def test_jit_nested_rhat_matches_eager_value():
    draws = jnp.asarray(_gaussian_draws(seed=6))
    eager = float(nested_rhat(draws, N_SUPERCHAINS))
    jitted = float(jax.jit(nested_rhat, static_argnums=1)(draws, N_SUPERCHAINS))
    assert jitted == pytest.approx(eager, rel=1e-10)
