# SPDX-License-Identifier: BSD-3-Clause
"""Convergence diagnostics of a sampler run, computed once in the shared driver.

Every engine reaches its samplers through :func:`~tengri.inference.sampling.driver.run_sampler`,
so the diagnostics live there rather than in each engine. The estimators are the repo's own
(:func:`~tengri.analysis.diagnostics.autocorrelation.split_rhat` and
:func:`~tengri.analysis.diagnostics.autocorrelation.effective_sample_size`) with the
thresholds of :mod:`tengri.inference.sampler_health`; nothing is estimated anew here.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np

from tengri.analysis.diagnostics.autocorrelation import effective_sample_size, split_rhat
from tengri.inference.sampler_health import ESS_FRACTION_WARN, RHAT_WARN

#: Fraction of steps that move the position below which a Metropolis chain has collapsed.
COLLAPSED_MOVE_FRACTION = 0.05

#: Relative change of the mean ELBO between the last two windows above which the optimizer
#: has not converged.
ELBO_REL_CHANGE_WARN = 0.05

#: Fraction of the ELBO trace each comparison window holds.
ELBO_WINDOW = 0.2

#: Weakest retained ``|log lambda|`` of a low-rank metric above which the rank is read as
#: truncating: a retained direction still departs from the diagonal by this factor, so the
#: ones beyond the rank probably do too.
LOWRANK_TRUNCATION_LOG_LAMBDA = float(np.log(1.5))

#: Most columns whose autocorrelation time is estimated; the rest are not read.
_MAX_ESS_COLUMNS = 2000


def element_names(layout) -> list[str]:
    """Return one name per element of the flat vector, in ``ravel`` order.

    ``layout`` is a :class:`~tengri.inference.sampling.target.LatentLayout` or a pytree
    shaped like the unraveled vector.
    """
    tree = layout.template() if hasattr(layout, "template") else layout
    names = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        base = "/".join(str(getattr(key, "key", key)) for key in path)
        shape = np.shape(leaf)
        if not shape:
            names.append(base)
        else:
            for index in np.ndindex(*shape):
                names.append(f"{base}[{','.join(map(str, index))}]")
    return names


def chain_diagnostics(draws, names, *, exchangeable: bool = False) -> tuple[dict, list[str]]:
    """Return the R-hat, ESS and move fractions of ``draws`` and the problems they show.

    A non-finite draw, R-hat or ESS is a named problem, never a skipped column: a chain that
    turns to NaN has not converged, and a NaN step is not a move.

    Parameters
    ----------
    draws : ndarray, shape (n_chains, n, D)
        Draws in the coordinates the caller wants judged.
    names : sequence of str
        One name per coordinate.
    exchangeable : bool
        True for resampled particle sets, whose autocorrelation ESS is not meaningful: only
        the cross-population R-hat is computed.

    Returns
    -------
    record : dict
        ``rhat_max``, ``rhat_max_param``, ``ess_bulk_min`` and ``ess_bulk_min_param`` (the
        ESS is the sum of the per-chain ESS), ``acceptance_rate`` (the fraction of steps that
        moved the position, pooled) and ``acceptance_rate_min`` (the worst single chain);
        absent keys mean the quantity could not be computed.
    problems : list of str
        One phrase per failure found.
    """
    draws = np.asarray(draws)
    n_chains, n, _dim = draws.shape
    record: dict = {}
    problems: list[str] = []
    if n < 8:
        return record, problems
    finite_columns = np.all(np.isfinite(draws), axis=(0, 1))
    for j in np.flatnonzero(~finite_columns)[:3]:
        bad = int(np.sum(~np.isfinite(draws[..., j])))
        problems.append(f"non-finite draws in '{names[j]}' ({bad} of {n_chains * n})")
    if int(np.sum(~finite_columns)) > 3:
        problems.append(f"non-finite draws in {int(np.sum(~finite_columns)) - 3} further columns")
    usable = np.flatnonzero(finite_columns)
    if n > 1 and not exchangeable:
        both = np.isfinite(draws[:, 1:]) & np.isfinite(draws[:, :-1])
        moved = np.any((np.diff(draws, axis=1) != 0) & both, axis=-1)
        record["acceptance_rate"] = float(moved.mean())
        record["acceptance_rate_min"] = float(moved.mean(axis=1).min())
        if record["acceptance_rate_min"] < COLLAPSED_MOVE_FRACTION:
            problems.append(
                f"a chain moved on {record['acceptance_rate_min']:.1%} of steps "
                f"(below {COLLAPSED_MOVE_FRACTION:.0%}): it has collapsed"
            )
    moving = [j for j in usable if np.ptp(draws[..., j]) > 0]
    if not moving:
        return record, problems

    worst = None
    for j in moving:
        column = draws[..., j]
        if exchangeable:
            if n_chains < 2:
                continue
            value = split_rhat(column)
        elif n_chains >= 2:
            half = n // 2
            value = split_rhat(np.concatenate([column[:, :half], column[:, half : 2 * half]]))
        else:
            value = split_rhat(column[0])
        if not np.isfinite(value):
            problems.append(f"R-hat is not finite for '{names[j]}'")
        elif worst is None or value > worst[0]:
            worst = (float(value), names[j])
    if worst is not None:
        record["rhat_max"], record["rhat_max_param"] = worst
        if worst[0] > RHAT_WARN:
            problems.append(f"R-hat {worst[0]:.3f} for '{worst[1]}' (above {RHAT_WARN})")

    if not exchangeable:
        low = None
        for j in moving[:_MAX_ESS_COLUMNS]:
            total = 0.0
            for chain in draws[..., j]:
                entry = effective_sample_size({names[j]: chain}, exclude_prefixes=())
                total += float(entry[names[j]]["ess"]) if names[j] in entry else 0.0
            if not np.isfinite(total):
                problems.append(f"ESS is not finite for '{names[j]}'")
            elif low is None or total < low[0]:
                low = (total, names[j])
        if low is not None:
            record["ess_bulk_min"], record["ess_bulk_min_param"] = low
            total_draws = n_chains * n
            if low[0] < ESS_FRACTION_WARN * total_draws:
                problems.append(
                    f"ESS {low[0]:.1f} for '{low[1]}' of {total_draws} draws "
                    f"(below {ESS_FRACTION_WARN:.0%})"
                )
    return record, problems


def elbo_diagnostics(elbo_history) -> tuple[dict, list[str]]:
    """Return the ELBO trace's relative change over its last window and any problem."""
    trace = np.asarray(elbo_history, dtype=float)
    trace = trace[np.isfinite(trace)]
    window = int(ELBO_WINDOW * trace.size)
    if window < 2:
        return {}, []
    last = float(trace[-window:].mean())
    before = float(trace[-2 * window : -window].mean())
    change = abs(last - before) / max(abs(last), 1e-12)
    problems = []
    if change > ELBO_REL_CHANGE_WARN:
        problems.append(
            f"the ELBO changed by {change:.1%} between its last two windows "
            f"(above {ELBO_REL_CHANGE_WARN:.0%}): the optimizer has not converged"
        )
    return {"elbo_rel_change": float(change)}, problems


def lowrank_diagnostics(rank, weakest_log_lambda, dim) -> tuple[dict, list[str]]:
    """Return the retained rank against D and a truncation warning."""
    record = {
        "lowrank_rank": int(rank),
        "lowrank_dim": int(dim),
        "lowrank_weakest_log_lambda": float(weakest_log_lambda),
    }
    problems = []
    if rank < dim and weakest_log_lambda > LOWRANK_TRUNCATION_LOG_LAMBDA:
        problems.append(
            f"the low-rank metric retains rank {int(rank)} of D={int(dim)} and its weakest "
            f"retained direction still departs from the diagonal by |log lambda| = "
            f"{weakest_log_lambda:.2f}: directions beyond the rank are probably truncated"
        )
    return record, problems


def warn_once(label: str, problems: list[str], *, stacklevel: int = 3) -> None:
    """Emit one warning naming every problem of the run."""
    if problems:
        warnings.warn(
            f"{label}: " + "; ".join(problems) + ".", RuntimeWarning, stacklevel=stacklevel
        )
