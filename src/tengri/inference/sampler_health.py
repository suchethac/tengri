# SPDX-License-Identifier: BSD-3-Clause
"""Post-run convergence check shared by every MCMC and SMC backend (#3456).

A chain sampler can return a posterior that is far from the target and still
carry R-hat and ESS values that say so, with nothing in the result that makes a
caller read them. :func:`check_sampler_convergence` runs once per fit, at the
single seam where ``Fitter.run`` hands back every backend's result, so no
sampler carries its own copy of the test.

It reuses the diagnostics the package already has: split R-hat from
:mod:`tengri.analysis.diagnostics.autocorrelation` and Sokal's ESS from the same
module. It does not change any draw.

For SMC the bulk-ESS test is replaced by the particle ancestor ESS the sampler
already reports, because a resampled population is exchangeable and its
autocorrelation ESS reads close to the particle count whatever the population
is worth (see ``_smc_ancestor_ess``).
"""

from __future__ import annotations

import warnings

import numpy as np

from tengri.analysis.diagnostics.autocorrelation import split_rhat
from tengri.config.exceptions import warn_measured

#: Split R-hat above this marks a run that has not mixed. Looser than the
#: Vehtari+2021 value of 1.01 on purpose: the check speaks only for clear failures.
RHAT_WARN = 1.1

#: Bulk ESS, or SMC ancestor ESS, below this fraction of the draw (particle)
#: count marks a run whose draws carry little independent information.
ESS_FRACTION_WARN = 0.1

#: Backend key of the tempered SMC sampler, which uses the particle ESS.
_SMC_BACKEND = "mcmc_smc"


def _scalar_columns(posterior) -> dict[str, np.ndarray]:
    """Flat draw columns of the free, moving, scalar parameters."""
    free = posterior.free_names
    names = (
        list(posterior.samples) if free is None else [n for n in free if n in posterior.samples]
    )
    columns = {}
    for name in names:
        arr = np.asarray(posterior.samples[name])
        if arr.ndim != 1 or arr.size < 4 or np.ptp(arr) == 0:
            continue
        columns[name] = arr
    return columns


def _worst_rhat(columns: dict[str, np.ndarray], n_chains: int) -> tuple[float, str] | None:
    """Largest split R-hat over the columns, with its parameter name."""
    worst = None
    for name, arr in columns.items():
        if n_chains >= 2 and arr.size % n_chains == 0:
            value = split_rhat(arr.reshape(n_chains, -1))
        else:
            value = split_rhat(arr)
        if np.isfinite(value) and (worst is None or value > worst[0]):
            worst = (float(value), name)
    return worst


def _worst_ess(columns: dict[str, np.ndarray], n_chains: int) -> tuple[float, str, int] | None:
    """Smallest bulk ESS over the columns, with its name and draw count.

    The ESS of a column is the sum of its per-chain ESS (:func:`summed_chain_ess`), the same
    estimator the sampling driver judges a run by.
    """
    from tengri.inference.sampling.health import summed_chain_ess

    worst = None
    for name, arr in columns.items():
        chains = n_chains if n_chains >= 2 and arr.size % n_chains == 0 else 1
        value = summed_chain_ess(arr.reshape(chains, -1), name)
        if np.isfinite(value) and (worst is None or value < worst[0]):
            worst = (value, name, int(arr.size))
    return worst


def rhat_problem(value: float, name: str) -> str | None:
    """Describe a split R-hat above :data:`RHAT_WARN`, or None when it is within it."""
    if value > RHAT_WARN:
        return f"R-hat {value:.3f} for '{name}' (above {RHAT_WARN})"
    return None


def ess_problem(value: float, name: str, n_draws: int) -> str | None:
    """Describe a bulk ESS below :data:`ESS_FRACTION_WARN` of ``n_draws``, or None."""
    if value < ESS_FRACTION_WARN * n_draws:
        return f"ESS {value:.1f} for '{name}' of {n_draws} draws (below {ESS_FRACTION_WARN:.0%})"
    return None


def smc_ess_problem(diagnostics: dict) -> str | None:
    """Describe the SMC particle-ESS failure, or None when the population is healthy."""
    ancestor = diagnostics.get("min_ancestor_ess")
    n_particles = diagnostics.get("n_particles")
    if ancestor is None or not n_particles:
        return None
    if float(ancestor) < ESS_FRACTION_WARN * float(n_particles):
        return (
            f"particle ESS {float(ancestor):.1f} of {int(n_particles)} particles "
            f"(below {ESS_FRACTION_WARN:.0%})"
        )
    return None


def _problems(posterior, backend: str) -> list[str]:
    """Every convergence failure found in this run, as short phrases."""
    diagnostics = posterior.diagnostics or {}
    n_chains = max(int(diagnostics.get("n_chains", 1)), 1)
    from tengri.inference.sampling.health import nonfinite_problems

    columns = _scalar_columns(posterior)
    found = []
    # A column with a non-finite draw is named and judged by no estimator that would drop it.
    for name in [n for n, arr in columns.items() if not np.all(np.isfinite(arr))]:
        found += nonfinite_problems(columns.pop(name)[None, :, None], [name])[1]

    if columns:
        worst_rhat = _worst_rhat(columns, n_chains)
        if worst_rhat is not None and (problem := rhat_problem(*worst_rhat)):
            found.append(problem)

    if backend == _SMC_BACKEND:
        smc_problem = smc_ess_problem(diagnostics)
        if smc_problem is not None:
            found.append(smc_problem)
    elif columns:
        worst_ess = _worst_ess(columns, n_chains)
        if worst_ess is not None and (problem := ess_problem(*worst_ess)):
            found.append(problem)
    return found


def check_sampler_convergence(posterior, *, backend: str):
    """Warn once when a finished MCMC or SMC run is not supported by its draws.

    Emits at most one :class:`RuntimeWarning` naming the backend and the worst
    parameter, and records the verdict in ``posterior.diagnostics`` under
    ``"convergence_warning"``: the warning text, or ``None`` for a clean run.
    The sampling itself is not touched.

    Parameters
    ----------
    posterior : Posterior
        The result of a sampler run. A result without samples is returned as is.
    backend : str
        Registry key of the backend that produced it, such as ``"mcmc_ghmc"``.

    Returns
    -------
    Posterior
        The same object, with ``diagnostics["convergence_warning"]`` set.
    """
    if posterior.samples is None:
        return posterior

    found = _problems(posterior, backend)
    message = None
    if found:
        message = (
            f"{backend}: the run does not support its posterior summaries: "
            f"{'; '.join(found)}. The draws have not mixed or carry too little "
            "information. Run longer or with more warmup, or use a sampler that mixes "
            "for this model (mcmc_nuts for a low-dimensional target)."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    posterior.diagnostics = {**(posterior.diagnostics or {}), "convergence_warning": message}
    return posterior


def check_student_t_mode(fitter, posterior):
    """Warn once when a Student-t marginal of a finished fit may not have reached its best mode.

    Not an entry of :func:`check_sampler_convergence`: that one reads the draws of a chain
    sampler, while this one applies to every backend (MAP and variational fits too) and needs
    the fitter's data term. The marginal is evaluated once more, eagerly, at the posterior's
    point estimate; no draw is touched.

    A warning is emitted when the weights of the iteratively reweighted mode still move by
    more than ``WEIGHT_CHANGE_WARN``, when the two starting points reach marginals more than
    ``START_GAP_WARN`` nats apart, or when the ratio of nuisance columns to data reaches
    ``RELIABLE_COLUMN_RATIO``. The measured values are recorded in ``posterior.diagnostics`` under
    ``student_t_weight_change``, ``student_t_start_gap`` and ``student_t_column_ratio``.

    Parameters
    ----------
    fitter : Fitter
        The fitter that produced ``posterior``.
    posterior : Posterior
        The result; returned unchanged apart from ``diagnostics``.

    Returns
    -------
    Posterior
        The same object. Without a Student-t marginal nothing is recorded.
    """
    from tengri.inference.loss_functions import build_data_term_for_fitter
    from tengri.observation.linear_marginal import (
        RELIABLE_COLUMN_RATIO,
        START_GAP_WARN,
        WEIGHT_CHANGE_WARN,
    )
    from tengri.observation.noise import uses_student_t

    marginalized = getattr(fitter, "_calibration_marginalize", False) or getattr(
        fitter, "_eline_marginalize", False
    )
    if not (marginalized and uses_student_t(fitter.spec)) or not posterior.params:
        return posterior
    params = {
        name: posterior.params[name] for name in fitter._free_names if name in posterior.params
    }
    params.update(fitter._fixed_values)
    params = fitter.spec.resolve_mirrors(params)
    report = build_data_term_for_fitter(fitter).mode_report(params, fitter._data_args)
    if report is None:
        return posterior

    ratio = report["n_columns"] / max(report["n_data"], 1.0)
    found = []
    if report["weight_change"] > WEIGHT_CHANGE_WARN:
        found.append(f"the weights still change by {report['weight_change']:.1e}")
    if report["start_gap"] > START_GAP_WARN:
        found.append(f"two starting points reach marginals {report['start_gap']:.2f} nat apart")
    if ratio >= RELIABLE_COLUMN_RATIO:
        found.append(
            f"{report['n_columns']} nuisance columns for {report['n_data']:.0f} data "
            f"(ratio {ratio:.2f}, at or above {RELIABLE_COLUMN_RATIO})"
        )
    if found:
        warn_measured(
            "Student-t marginal over the nuisance coefficients: "
            + "; ".join(found)
            + ". Its posterior can have several modes and the reported one may not be the "
            "global one; compare with Gaussian noise or fewer nuisance columns.",
            RuntimeWarning,
            stacklevel=3,
            weight_change=report["weight_change"],
            start_gap=report["start_gap"],
            column_ratio=ratio,
        )
    posterior.diagnostics = {
        **(posterior.diagnostics or {}),
        "student_t_weight_change": report["weight_change"],
        "student_t_start_gap": report["start_gap"],
        "student_t_column_ratio": ratio,
    }
    return posterior
