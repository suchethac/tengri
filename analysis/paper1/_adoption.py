#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The one rule deciding whether a grid cell is drawn.

fig05 and fig06 each carried their own copy of this, written as a chain of
per-configuration branches ending in a fall-through that rejected everything
it did not name. Both named I, II and III, so both silently refused to draw
Configurations IV, V and VI however many cells were on disk. Two copies also
means a correction to one leaves the other behind, which is the condition
this module exists to remove.

The bar is `adoption_pass`, the criterion the driver records. On this
branch `fit_one` computes it as two legs, `n_divergent == 0 and rhat_max <
1.01` -- not the one leg this line used to claim. The grid session runs a
driver carrying a third, `ess_min >= fit_one.ESS_FLOOR` (100, the owner's
ruling), added in 03ffff576 on `paper1/nss-profile-mass`; that commit is on
neither this branch, nor `paper1/grid-20x6-locked`, nor `main`, so the flag
means different things on either side and a cell's provenance decides which.
Configuration III is the single exception, and it is measured rather than
assumed: across `results/fits_superseded_oldsuite_20260920`, 0 of 17 of its
cells clear a zero-divergence bar, against 15 of 17, 14 of 18, 16 of 18, 15 of
16 and 15 of 15 for the other five. Its nonparametric continuity SFH does not
reach that bar at this dimensionality, so it is judged on convergence and a
divergence rate instead. Any figure drawing Configuration III must say so in
its caption; the relaxation is not applied to any other configuration.
"""

from __future__ import annotations

import re
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

#: Upper bound below which Configuration V's ``peak_gyr`` prior counts as
#: capped. The owner capped it at ``age_at_z(z)``, 5.1 to 5.9 Gyr across this
#: sample, so 6.0 separates a capped cell from the 13.0 the uncapped prior
#: carried without being sensitive to which galaxy it is.
PEAK_GYR_CAP_MAX = 6.0
_PEAK_PARAM = "sfh_lnorm_peak_gyr"


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
        # The owner added an effective-sample leg to the grid's bar
        # (fit_one.ESS_FLOOR = 100) on 2026-09-21. The relaxed bar is computed
        # here rather than read from adoption_pass, so without this it would
        # adopt a Configuration III cell the grid itself refuses. Absent means
        # unverified, consistently with the rhat_max and divergence checks above.
        ess_min = meta.get("ess_min")
        if ess_min is None:
            return Verdict(False, "relaxed bar: no ess_min recorded")
        if float(ess_min) < LOW_ESS:
            return Verdict(False, f"relaxed bar: ess_min {float(ess_min):.1f} < {LOW_ESS:.0f}")
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


def uncapped_peak_note(meta: dict, config: str) -> str | None:
    """A Configuration V cell that predates, or violates, the ``peak_gyr`` cap.

    Row V was rerun after the owner capped ``peak_gyr`` at ``age_at_z(z)``;
    the pre-cap cells were archived rather than deleted, so both kinds exist on
    disk and a figure pointed at the wrong directory would mix two priors under
    one configuration label.

    **A cell with no ``priors`` block fails.** The block was added by the same
    commit that capped the prior, so its absence means the cell predates the
    cap -- exactly the cells that must not be counted. Read the obvious way,
    ``meta["priors"].get("sfh_lnorm_peak_gyr")`` returns ``None`` for those and
    a bound test on ``None`` skips them into the "fine" branch.

    The bound is parsed out of the recorded ``repr``, so an unparseable value
    is reported rather than passed: guessing that a prior is capped because its
    text could not be read is the same failure in a different coat.
    """
    if config != "V":
        return None
    priors = meta.get("priors")
    if not priors:
        return (
            "records no priors block, so it predates the peak_gyr cap "
            "(the block and the cap landed together)"
        )
    recorded = priors.get(_PEAK_PARAM)
    if recorded is None:
        return f"records priors but no {_PEAK_PARAM}, so the cap cannot be verified"
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(recorded))
    if not numbers:
        return f"{_PEAK_PARAM} prior {recorded!r} has no readable bound"
    upper = float(numbers[-1])
    if upper >= PEAK_GYR_CAP_MAX:
        return (
            f"{_PEAK_PARAM} upper bound {upper:g} >= {PEAK_GYR_CAP_MAX:g}: "
            "this cell ran under the uncapped prior"
        )
    return None
