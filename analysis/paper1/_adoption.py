#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The one rule deciding whether a grid cell is drawn.

fig05 and fig06 each carried their own copy of this, written as a chain of
per-configuration branches ending in a fall-through that rejected everything
it did not name. Both named I, II and III, so both silently refused to draw
Configurations IV, V and VI however many cells were on disk. Two copies also
means a correction to one leaves the other behind, which is the condition
this module exists to remove.

The bar is `adoption_pass`, the zero-divergence criterion the driver records.
Configuration III is the single exception, and it is measured rather than
assumed: across `results/fits_superseded_oldsuite_20260920`, 0 of 17 of its
cells clear a zero-divergence bar, against 15 of 17, 14 of 18, 16 of 18, 15 of
16 and 15 of 15 for the other five. Its nonparametric continuity SFH does not
reach that bar at this dimensionality, so it is judged on convergence and a
divergence rate instead. Any figure drawing Configuration III must say so in
its caption; the relaxation is not applied to any other configuration.
"""

from __future__ import annotations

from typing import NamedTuple

RELAXED_CONFIGS = frozenset({"III"})
RELAXED_RHAT_MAX = 1.01
RELAXED_DIVERGENCE_RATE = 0.015

_DEFAULT_N_SAMPLES = 600
_DEFAULT_N_CHAINS = 4


class Verdict(NamedTuple):
    """Whether to draw a cell, and why not when it is refused."""

    adopted: bool
    reason: str


def divergence_rate(meta: dict) -> float:
    """Divergent transitions per post-warmup draw, over all chains."""
    divergences = meta.get("divergences") or 0
    # `or` would read a recorded zero as absent and substitute the default,
    # inventing a denominator for a cell that honestly reports no draws.
    n_samples = meta.get("n_samples")
    n_chains = meta.get("n_chains")
    n_samples = _DEFAULT_N_SAMPLES if n_samples is None else n_samples
    n_chains = _DEFAULT_N_CHAINS if n_chains is None else n_chains
    draws = n_samples * n_chains
    if draws <= 0:
        return 0.0
    return divergences / draws


def is_adopted(meta: dict, config: str) -> Verdict:
    """Judge one cell. Every configuration is judged; none falls through."""
    if config in RELAXED_CONFIGS:
        rhat_max = meta.get("rhat_max")
        if rhat_max is None:
            return Verdict(False, "no rhat_max recorded")
        rate = divergence_rate(meta)
        if rhat_max >= RELAXED_RHAT_MAX:
            return Verdict(
                False,
                f"relaxed bar: rhat_max {rhat_max:.4f} >= {RELAXED_RHAT_MAX}",
            )
        if rate > RELAXED_DIVERGENCE_RATE:
            return Verdict(
                False,
                f"relaxed bar: divergence rate {rate:.4f} > {RELAXED_DIVERGENCE_RATE}",
            )
        return Verdict(True, "relaxed bar")

    if meta.get("adoption_pass") is True:
        return Verdict(True, "adoption_pass")
    return Verdict(False, "did not pass the adoption bar")
