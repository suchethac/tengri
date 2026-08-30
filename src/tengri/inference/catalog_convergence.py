# SPDX-License-Identifier: BSD-3-Clause
"""Per-galaxy health for a catalog fit: converged, refused, or silently frozen.

A catalog throughput number that silently drops galaxies is not a throughput
number, and every way a galaxy can fail to produce a usable posterior is
invisible in the wall clock. This module makes the three failure modes disjoint,
countable and reportable beside the throughput.

Why three buckets and not one pass/fail
---------------------------------------
**Refused** is the loud one. Since #2088 the window-adaptation backends raise
:class:`~tengri.config.exceptions.DeadFitError` when the final warmup window is
>= 90 % divergent, rather than spending the sampling budget on a posterior the
sampler cannot enter. A refusal is a *good* outcome -- it is the sampler saying
so -- but the galaxy still has no posterior, and counting it as anything other
than its own bucket either inflates the failure rate (if lumped with
non-convergence) or hides a missing galaxy (if dropped).

**Silently frozen** is the one this module exists for. #2093: nb05 seed 0
returns *normally* with 1200/1200 sampling draws divergent, split R-hat 1.4e13
and a unique-draw fraction of 0.002. The #2090 guard cannot see it, because that
guard inspects the **warmup** record and this fit's warmup ended healthy; the
draws it never took are where the failure is. So the refusal trigger is
necessary and not sufficient, and a post-hoc check over the returned draws is
the other half.

**Converged** is deliberately not "R-hat passed". Phase 0 measured **73 % of
galaxies clearing max split-R-hat < 1.01 with zero divergences while their worst
ESS was 2.63 of 500 draws** (``bench/reports/2026-08-30_gpu_catalog_throughput.md``).
Split R-hat compares two halves of one chain: two equally badly-mixed halves
agree, and it reads 1.00. R-hat alone is therefore not a convergence count, and
every report from this module carries **min ESS beside it**. Whether a low ESS
should *demote* a galaxy is the caller's call, via ``ess_floor``; the default
reports the number rather than legislating a threshold, because the honest
default is the one that cannot quietly change a published count.

The arithmetic
--------------
Divergence rates go through
``_shared.total_draws``, never through
``n_samples``. ``n_samples`` is recorded per chain while ``n_divergent`` is
summed over every chain, so dividing one by the other over-reports by exactly
the chain count -- #2087 measured 400 % divergences on a 4-chain fit. The
catalog ChEES path is the first one that can run more than one chain per galaxy,
so this stopped being academic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CATALOG_MAX_RHAT",
    "FROZEN_DISTINCT_FRAC",
    "CatalogConvergence",
    "GalaxyHealth",
    "catalog_convergence",
    "galaxy_health",
]

#: Split-R-hat bar a galaxy must clear to be counted converged.
#:
#: 1.01, the bar the notebooks and every ``bench/reports/2026-08-17_*`` report
#: already use. Not 1.1: the reports that established this bar found configs
#: sitting at 1.024 with an ESS of 2.3, and 1.1 would have called them fine.
CATALOG_MAX_RHAT = 1.01

#: Distinct-draw fraction at or below which a galaxy is called frozen.
#:
#: The measured signature (#2093) is **0.002** -- roughly two distinct positions
#: in a thousand draws, a chain that rejected essentially every proposal. A
#: healthy low-ESS chain is nowhere near this: even at ESS 2.6 of 500 draws every
#: accepted proposal is a new position, so the distinct fraction stays near 1.
#: 0.01 sits an order of magnitude above the measured failure and two below the
#: worst healthy case, which is what makes it a *separator* rather than a
#: threshold anyone has to defend to three digits.
FROZEN_DISTINCT_FRAC = 0.01


@dataclass(frozen=True)
class GalaxyHealth:
    """One galaxy's verdict and the numbers behind it.

    Attributes
    ----------
    verdict : {"converged", "unconverged", "frozen", "refused"}
        Exactly one bucket, always. ``"refused"`` is only ever set by a caller
        that caught a :class:`~tengri.config.exceptions.DeadFitError`; it cannot
        be derived from draws, because a refused fit has none.
    max_rhat : float or None
        Worst split R-hat over the free parameters. ``None`` when R-hat could
        not be computed -- which for a frozen chain is itself the evidence
        (``Posterior.rhat`` raises on zero variance by design, #1438).
    max_rhat_param, min_ess_param : str or None
        Which parameter carried those extremes. A catalog-wide "max R-hat 1.9"
        with no name attached cannot be acted on; with one it usually points at
        a single degenerate direction rather than at the sampler.
    min_ess : float or None
        Worst effective sample size over the free parameters, in draws.
        **Reported for every bucket**, including ``"converged"``: see the module
        docstring for the 73 %-at-ESS-2.63 measurement that makes this
        non-optional.
    divergence_rate : float or None
        ``n_divergent / total_draws``, in ``[0, 1]``. A rate of 1.0 is #2093's
        first signature.
    distinct_frac : float or None
        Smallest per-parameter fraction of *distinct* draws. #2093's second
        signature, and the one the divergence rate cannot supply on a chain that
        froze without diverging (#1999).
    reason : str or None
        Free text for a refusal, or the specific signature that produced a
        ``"frozen"`` verdict.
    """

    index: int
    verdict: str
    max_rhat: float | None = None
    max_rhat_param: str | None = None
    min_ess: float | None = None
    min_ess_param: str | None = None
    divergence_rate: float | None = None
    distinct_frac: float | None = None
    n_draws: int = 0
    reason: str | None = None


def _free_sample_columns(posterior) -> dict:
    """The sample columns that belong to parameters the sampler could move.

    ``Fixed`` parameters ride along as constant arrays, so a zero-variance
    column is evidence of a dead sampler only when the parameter was free
    (#2087). ``Posterior.free_names`` is ``None`` for a hand-built posterior
    with no model, in which case every column is judged -- the conservative
    reading, and the one that cannot silently exempt a real failure.
    """
    samples = getattr(posterior, "samples", None) or {}
    free = getattr(posterior, "free_names", None)
    if free is None:
        return {k: np.asarray(v) for k, v in samples.items()}
    return {k: np.asarray(samples[k]) for k in free if k in samples}


def galaxy_health(posterior, *, index: int = 0, ess_floor: float | None = None) -> GalaxyHealth:
    """Classify one galaxy's posterior into exactly one bucket.

    Parameters
    ----------
    posterior : Posterior
        One galaxy's result.
    index : int
        Its position in the catalog, carried through for reporting.
    ess_floor : float, optional
        When given, a galaxy whose ``min_ess`` is below this is *not* counted
        converged even if its R-hat passed. ``None`` (default) reports ESS
        without gating on it -- see the module docstring on why the default does
        not legislate a threshold.

    Returns
    -------
    GalaxyHealth

    Notes
    -----
    The frozen test is deliberately a **disjunction of two independent
    signatures**, either of which is sufficient:

    1. every kept draw diverged (``divergence_rate == 1``), and
    2. a free parameter took essentially no distinct values
       (``distinct_frac <= FROZEN_DISTINCT_FRAC``).

    #2093's fit trips both. #1999's frozen-with-zero-divergences fits trip only
    the second, and a sampler that rejects everything at a step size it cannot
    recover from trips only the first. Requiring both would miss half of each.
    """
    from tengri.inference.backends.mcmc._shared import total_draws

    diag = getattr(posterior, "diagnostics", None) or {}
    columns = _free_sample_columns(posterior)

    n_chains = max(int(diag.get("n_chains", 1) or 1), 1)
    if "n_samples" in diag:
        n_total = total_draws(diag)
    elif columns:
        # No recorded per-chain count: the column length is already the
        # flattened (n_chains * n_samples,) axis, so it IS the total. Divide
        # before ``total_draws`` multiplies it back, or a posterior carrying
        # n_chains but no n_samples reports twice the draws it has (#2087).
        flat = len(next(iter(columns.values())))
        n_total = total_draws({"n_chains": n_chains}, n_samples=flat // n_chains)
    else:
        n_total = 0

    n_div = diag.get("n_divergent")
    div_rate = (int(n_div) / n_total) if (n_div is not None and n_total) else None

    distinct = None
    for col in columns.values():
        arr = np.asarray(col).reshape(-1)
        if arr.size == 0:
            continue
        frac = float(np.unique(arr).size) / float(arr.size)
        distinct = frac if distinct is None else min(distinct, frac)

    max_rhat = max_rhat_param = None
    rhat_failed = False
    try:
        rh = posterior.rhat()
    except Exception:
        # ``Posterior.rhat`` raises on a zero-variance chain by design (#1438).
        # That is a frozen-chain signature, not a missing measurement.
        rhat_failed = True
    else:
        finite = [(float(v), k) for k, v in rh.items() if np.isfinite(v)]
        if finite:
            max_rhat, max_rhat_param = max(finite)

    min_ess = min_ess_param = None
    if columns and n_total >= 4:
        from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

        try:
            ess = effective_sample_size(columns)
        except Exception:
            ess = {}
        for name, rec in ess.items():
            val = rec["ess"] if isinstance(rec, dict) else rec
            if np.isfinite(val) and (min_ess is None or float(val) < min_ess):
                min_ess, min_ess_param = float(val), name

    frozen_reasons = []
    if div_rate is not None and div_rate >= 1.0:
        frozen_reasons.append(f"every kept draw diverged ({div_rate:.0%} of {n_total})")
    if distinct is not None and distinct <= FROZEN_DISTINCT_FRAC:
        frozen_reasons.append(f"distinct-draw fraction {distinct:.4f} <= {FROZEN_DISTINCT_FRAC}")
    if rhat_failed:
        frozen_reasons.append("R-hat undefined: a free parameter has zero variance")

    if frozen_reasons:
        return GalaxyHealth(
            index=index,
            verdict="frozen",
            max_rhat=max_rhat,
            max_rhat_param=max_rhat_param,
            min_ess=min_ess,
            min_ess_param=min_ess_param,
            divergence_rate=div_rate,
            distinct_frac=distinct,
            n_draws=n_total,
            reason="; ".join(frozen_reasons),
        )

    converged = (
        max_rhat is not None
        and max_rhat < CATALOG_MAX_RHAT
        and (div_rate is None or div_rate == 0.0)
        and (ess_floor is None or (min_ess is not None and min_ess >= ess_floor))
    )
    return GalaxyHealth(
        index=index,
        verdict="converged" if converged else "unconverged",
        max_rhat=max_rhat,
        max_rhat_param=max_rhat_param,
        min_ess=min_ess,
        min_ess_param=min_ess_param,
        divergence_rate=div_rate,
        distinct_frac=distinct,
        n_draws=n_total,
    )


@dataclass(frozen=True)
class CatalogConvergence:
    """Catalog-wide health, as four disjoint counts that sum to ``n_galaxies``.

    ``n_converged + n_unconverged + n_frozen + n_refused == n_galaxies``, and
    that identity is the point: a count that does not close has dropped
    galaxies somewhere.
    """

    n_galaxies: int
    n_converged: int
    n_unconverged: int
    n_frozen: int
    n_refused: int
    max_rhat: float | None
    min_ess: float | None
    #: Worst ESS **among the galaxies counted converged**, which is a different
    #: number from ``min_ess`` and the one a converged-rate must be quoted with.
    #: The catalog-wide minimum is set by the tail that already failed R-hat, so
    #: printing it beside a converged rate compares two different populations.
    min_ess_converged: float | None
    divergence_rate: float | None
    per_galaxy: tuple = field(default=(), repr=False)

    @property
    def frac_converged(self) -> float | None:
        return (self.n_converged / self.n_galaxies) if self.n_galaxies else None

    def __repr__(self) -> str:
        ess = "n/a" if self.min_ess_converged is None else f"{self.min_ess_converged:.1f}"
        rhat = "n/a" if self.max_rhat is None else f"{self.max_rhat:.4f}"
        return (
            f"CatalogConvergence({self.n_converged} converged / "
            f"{self.n_unconverged} unconverged / {self.n_frozen} frozen / "
            f"{self.n_refused} refused of {self.n_galaxies}; "
            f"max R-hat {rhat}, min ESS among converged {ess})"
        )

    def summary(self) -> str:
        """A one-paragraph report that never quotes a rate without its ESS."""
        pct = "n/a" if self.frac_converged is None else f"{100 * self.frac_converged:.1f} %"
        ess = "n/a" if self.min_ess_converged is None else f"{self.min_ess_converged:.2f}"
        div = "n/a" if self.divergence_rate is None else f"{self.divergence_rate:.4f}"
        return (
            f"{self.n_converged}/{self.n_galaxies} galaxies converged ({pct}) at "
            f"max split-R-hat < {CATALOG_MAX_RHAT} with zero divergences; the worst "
            f"ESS among them is {ess} draws. Also: {self.n_unconverged} moved and did "
            f"not mix, {self.n_frozen} came back frozen (a posterior R-hat cannot "
            f"fault), {self.n_refused} were refused before sampling. "
            f"Catalog divergence rate {div}."
        )


def catalog_convergence(
    posteriors,
    *,
    refusals=None,
    n_galaxies: int | None = None,
    ess_floor: float | None = None,
    max_galaxies: int | None = None,
) -> CatalogConvergence:
    """Roll per-galaxy verdicts into the four disjoint catalog counts.

    Parameters
    ----------
    posteriors : sequence of Posterior
        The galaxies that produced draws.
    refusals : mapping, optional
        ``{catalog_index: reason}`` for galaxies that raised
        :class:`~tengri.config.exceptions.DeadFitError` and therefore have no
        posterior at all. Counted, never inferred.
    n_galaxies : int, optional
        Catalog size. Defaults to ``len(posteriors) + len(refusals)``. Pass it
        explicitly when a caller knows the catalog was larger than what it
        holds -- a mismatch is exactly the silent drop this function exists to
        make impossible to miss.
    ess_floor : float, optional
        Forwarded to :func:`galaxy_health`.
    max_galaxies : int, optional
        Judge only the first this many posteriors. ESS is an autocorrelation
        estimate per parameter per galaxy and is the expensive part; a caller
        sweeping N = 2048 may want a sample. The returned ``n_galaxies`` then
        reports what was **checked**, so no rate is ever computed over a
        denominator that was not measured.
    """
    refusals = dict(refusals or {})
    posteriors = list(posteriors)
    if max_galaxies is not None:
        posteriors = posteriors[:max_galaxies]
        refusals = {k: v for k, v in refusals.items() if k < max_galaxies}
        n_galaxies = len(posteriors) + len(refusals)
    if n_galaxies is None:
        n_galaxies = len(posteriors) + len(refusals)

    rows = [galaxy_health(p, index=i, ess_floor=ess_floor) for i, p in enumerate(posteriors)]
    rows += [
        GalaxyHealth(index=i, verdict="refused", reason=str(why)) for i, why in refusals.items()
    ]

    counts = {"converged": 0, "unconverged": 0, "frozen": 0, "refused": 0}
    for row in rows:
        counts[row.verdict] += 1

    rhats = [r.max_rhat for r in rows if r.max_rhat is not None and np.isfinite(r.max_rhat)]
    esses = [r.min_ess for r in rows if r.min_ess is not None]
    esses_ok = [r.min_ess for r in rows if r.verdict == "converged" and r.min_ess is not None]
    n_div = sum((r.divergence_rate * r.n_draws) for r in rows if r.divergence_rate is not None)
    n_draws = sum(r.n_draws for r in rows if r.divergence_rate is not None)

    return CatalogConvergence(
        n_galaxies=n_galaxies,
        n_converged=counts["converged"],
        n_unconverged=counts["unconverged"],
        n_frozen=counts["frozen"],
        n_refused=counts["refused"],
        max_rhat=max(rhats) if rhats else None,
        min_ess=min(esses) if esses else None,
        min_ess_converged=min(esses_ok) if esses_ok else None,
        divergence_rate=(n_div / n_draws) if n_draws else None,
        per_galaxy=tuple(rows),
    )
