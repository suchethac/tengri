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

from tengri.analysis.diagnostics.autocorrelation import effective_sample_size, split_rhat

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
    """Largest finite split R-hat over the columns, with its parameter name."""
    worst = None
    for name, arr in columns.items():
        if n_chains >= 2 and arr.size % n_chains == 0:
            value = split_rhat(arr.reshape(n_chains, -1))
        else:
            value = split_rhat(arr)
        if np.isfinite(value) and (worst is None or value > worst[0]):
            worst = (float(value), name)
    return worst


def _worst_ess(columns: dict[str, np.ndarray]) -> tuple[float, str, int] | None:
    """Smallest finite bulk ESS over the columns, with its name and draw count."""
    info = effective_sample_size(columns)
    worst = None
    for name, arr in columns.items():
        value = float(info[name]["ess"])
        if np.isfinite(value) and (worst is None or value < worst[0]):
            worst = (value, name, int(arr.size))
    return worst


def _smc_ess_problem(diagnostics: dict) -> str | None:
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
    columns = _scalar_columns(posterior)
    found = []

    if columns:
        worst_rhat = _worst_rhat(columns, n_chains)
        if worst_rhat is not None and worst_rhat[0] > RHAT_WARN:
            found.append(f"R-hat {worst_rhat[0]:.3f} for '{worst_rhat[1]}' (above {RHAT_WARN})")

    if backend == _SMC_BACKEND:
        smc_problem = _smc_ess_problem(diagnostics)
        if smc_problem is not None:
            found.append(smc_problem)
    elif columns:
        worst_ess = _worst_ess(columns)
        if worst_ess is not None and worst_ess[0] < ESS_FRACTION_WARN * worst_ess[2]:
            found.append(
                f"ESS {worst_ess[0]:.1f} for '{worst_ess[1]}' of {worst_ess[2]} draws "
                f"(below {ESS_FRACTION_WARN:.0%})"
            )
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
