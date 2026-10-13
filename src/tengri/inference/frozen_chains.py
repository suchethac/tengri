# SPDX-License-Identifier: BSD-3-Clause
"""Detect parameters whose chains move but never mix (#2112).

A chain that jitters by a sliver of the posterior width around its own center,
while the other chains sit around different centers, is not converged and not
dead: it moves, so the all-divergent and exactly-constant guards stay silent,
and R-hat is the only instrument that could see it. This module measures the
two ratios that make the regime visible, per parameter.

Chains are the rows of a ``(n_chains, n_samples)`` reshape of the flat draw
column, which is the layout every MCMC backend writes. A single chain is split
into two halves, so the same test applies.
"""

from __future__ import annotations

import numpy as np

#: Within-chain std over the between-chain std of the chain means. Chains that
#: mix have means that agree to within the Monte Carlo error, so this ratio is
#: of order sqrt(n_samples) on a healthy fit and falls below one only when the
#: chains disagree by more than their own spread.
FROZEN_MIX_RATIO_MAX = 0.02

#: Within-chain std over the pooled std of every draw. A healthy chain explores
#: the posterior, so its own spread is comparable to the pooled one. A frozen
#: chain sits in a sliver of it.
FROZEN_WIDTH_RATIO_MAX = 0.02

#: Fewest draws per chain for the ratios to mean anything. Below this a chain's
#: std is itself too noisy to compare.
FROZEN_MIN_DRAWS_PER_CHAIN = 20


def _chain_matrix(column: np.ndarray, n_chains: int) -> np.ndarray | None:
    """Reshape a flat draw column to ``(n_chains, n_per_chain)``, splitting one chain in half."""
    if column.ndim != 1 or column.size == 0:
        return None
    if n_chains >= 2:
        if column.size % n_chains:
            return None
        return column.reshape(n_chains, -1)
    half = column.size // 2
    if half < FROZEN_MIN_DRAWS_PER_CHAIN:
        return None
    return np.stack([column[:half], column[half : 2 * half]])


def _is_frozen_moving(column: np.ndarray, n_chains: int) -> bool:
    """True when one parameter's chains move within themselves but do not mix."""
    chains = _chain_matrix(column, n_chains)
    if chains is None or chains.shape[1] < FROZEN_MIN_DRAWS_PER_CHAIN:
        return False
    pooled = float(np.std(column, ddof=1))
    if not np.isfinite(pooled) or pooled <= 0.0:
        return False
    within = float(np.mean(np.std(chains, axis=1, ddof=1)))
    between = float(np.std(np.mean(chains, axis=1), ddof=1))
    if not np.isfinite(between) or between <= 0.0:
        return False
    return within / between < FROZEN_MIX_RATIO_MAX and within / pooled < FROZEN_WIDTH_RATIO_MAX


def frozen_moving_params(samples: dict, n_chains: int) -> tuple[str, ...]:
    """Names of the parameters whose chains move but never mix.

    Parameters
    ----------
    samples : dict
        Parameter name -> flat draw column, shape ``(n_chains * n_samples,)``,
        chain-major.
    n_chains : int
        Number of chains the columns were drawn from. A value below two splits
        the single chain into halves.

    Returns
    -------
    tuple of str
        Parameters flagged by the within/between and within/pooled ratios, in the
        order of ``samples``. Empty when every parameter mixes or there are too
        few draws to judge.

    Notes
    -----
    The posterior width is taken as the pooled std of all draws, because a
    ``Posterior`` does not carry its prior. A chain set that is frozen at one
    point in a broad posterior therefore reads as healthy here; only disagreement
    between chains, or a sliver against a width the draws themselves establish,
    is visible.
    """
    flagged = []
    for name, value in samples.items():
        column = np.asarray(value, dtype=float)
        if _is_frozen_moving(column, n_chains):
            flagged.append(name)
    return tuple(flagged)
