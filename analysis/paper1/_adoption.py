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

#: Effective samples below which an adopted cell carries too little
#: information to quote. NOT part of the bar -- the bar is the owner's
#: criterion and is left exactly as it is. This only makes the case
#: visible, because the bar cannot see it by construction.
LOW_ESS = 100.0


class Verdict(NamedTuple):
    """Whether to draw a cell, and why not when it is refused."""

    adopted: bool
    reason: str


def divergence_rate(meta: dict) -> float:
    """Divergent transitions per post-warmup draw, over all chains.

    Returns NaN when the count was never recorded. ``or 0`` read an absent
    count as a clean zero, which is the optimistic reading of a cell nobody
    measured: the divergence half of the relaxed bar then passed vacuously.
    ``is_adopted`` already refuses a cell with no ``rhat_max``; this makes the
    divergence count answer to the same standard instead of defaulting in its
    favor. NaN compares false against every threshold, so a caller that forgets
    to check gets a refusal rather than a pass.
    """
    divergences = meta.get("divergences")
    if divergences is None:
        return float("nan")
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
        if meta.get("divergences") is None:
            return Verdict(False, "no divergence count recorded")
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


def low_ess_note(meta: dict, verdict: Verdict) -> str | None:
    """An adopted cell resting on too few effective samples, or ``None``.

    The bar is zero divergences and ``rhat_max`` under 1.01, and it has no
    effective-sample criterion. That is not an oversight that R-hat covers:
    **``rhat_max`` and ``ess_min`` are extrema over different parameters.**
    In a thirty-six parameter posterior the worst-mixing R-hat can belong to
    one parameter while the worst ESS belongs to another, so a good
    ``rhat_max`` places no bound at all on ``ess_min``. The two diagnostics
    are not redundant and neither substitutes for the other.

    Measured with tengri's own diagnostics on a single parameter, R-hat does
    catch most near-frozen chains -- an AR(1) at rho=0.97 gives rhat 1.052 and
    is refused. But there is a window it misses: rho=0.9 at stationarity gives
    rhat 1.0009, which passes, on an ESS of 33 out of 1200 draws. Across many
    parameters the gap is wider still.

    A cell like that is not wrong, it is uninformative, and folding it into an
    adoption *rate* silently mixes converged fits with near-frozen ones. So it
    is reported rather than reclassified: changing the bar is the owner's call.
    """
    if not verdict.adopted:
        return None
    ess = meta.get("ess_min")
    if ess is None:
        return "adopted with no ess_min recorded, so its information content is unverified"
    if float(ess) < LOW_ESS:
        return (
            f"adopted on ess_min {float(ess):.1f} < {LOW_ESS:.0f}: the bar has no "
            "effective-sample criterion, and rhat_max cannot bound ess_min "
            "because they are extrema over different parameters"
        )
    return None
