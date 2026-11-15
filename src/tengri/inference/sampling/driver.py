# SPDX-License-Identifier: BSD-3-Clause
"""The one sampler driver: a registry of samplers and the layers every one of them gets.

:func:`run_sampler` runs any registered sampler on a :class:`~.target.FlatTarget`. Two
settings are generic layers rather than per-sampler code:

* ``n_chains``: chain families adapt once and vmap overdispersed chains
  (:mod:`tengri.inference.backends.mcmc.chain_program`); every other sampler runs
  independent chains from overdispersed starts.
* ``metric="dense"``: families with a dense adaptation of their own use it; every other
  sampler runs on a target whitened by a pilot run's covariance
  (:meth:`~.target.FlatTarget.whiten`), a reparametrization of the target that applies to
  any sampler, with the draws mapped back. Where whitening cannot change the result (a MAP
  point, a Laplace Gaussian: both affine-equivariant) the sampler is marked ``invariant`` and
  runs as it is, recorded as such in its diagnostics.

The sampler implementations live in :mod:`tengri.inference.sampling.samplers` and register
here, so adding a sampler is one registration and every engine reaches it.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from tengri.inference.sampling.target import FlatTarget, pilot_whitening

#: Settings every sampler may read; an unknown key is refused by name.
DEFAULTS: dict[str, Any] = {
    "n_warmup": 300,
    "n_burnin": 100,
    "n_samples": 500,
    "n_leapfrog": 10,
    "max_num_doublings": 5,
    "dense_mass_matrix": False,
    "target_accept_rate": None,
    "chain_jitter": None,
    "ghmc_alpha": 0.8,
    "ghmc_delta": 0.65,
    "mclmc_target_accept_rate": 0.65,
    "map_steps": 300,
    "map_learning_rate": 0.05,
    "laplace_grad_tol": 1e-2,
    "nss_n_live": 500,
    "nss_num_delete": 50,
    "nss_num_inner_steps": None,
    "nss_log_evidence_tol": -3.0,
    "nss_max_iterations": 10000,
    "n_ensemble": 32,
    "max_leapfrog_steps": 200,
    "max_rank": 10,
    "smc_n_particles": 512,
    "smc_n_mcmc_steps": 5,
    "smc_target_ess": 0.5,
    "smc_max_temperatures": 200,
    "vi_n_steps": 2000,
    "vi_n_mc_samples": 5,
    "vi_learning_rate": 1e-2,
    "is_n_draws": 50_000,
    "is_proposal_df": 5.0,
    "is_proposal_inflation": 1.5,
    "memory_budget_gb": 8.0,
    "verbose": False,
}

KINDS = ("chain", "ensemble", "variational", "nested", "point")
FIDELITIES = ("exact", "mean_only", "point")
METRIC_MODES = ("layer", "native", "internal", "invariant")


@dataclass(frozen=True)
class SamplerResult:
    """Draws and diagnostics of one sampler run.

    Parameters
    ----------
    draws : ndarray, shape (n_chains, n, D)
        Retained draws, one row of ``n`` per chain.
    divergent : ndarray of bool, shape (n_chains, n) or None
        Divergence flags where the family reports them.
    diagnostics : dict
        Sampler-specific diagnostics.
    """

    draws: Any
    divergent: Any = None
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Sampler:
    """A registered sampler.

    Parameters
    ----------
    name : str
        Driver name.
    kind : {"chain", "ensemble", "variational", "nested", "point"}
        What the sampler is; chain kinds wrap a ``ChainProgram`` and ``Adaptation``.
    methods : tuple of str
        Registry method names (and aliases) that select it.
    run : callable
        ``run(target, key, n_chains, metric, cfg) -> SamplerResult``.
    fidelity : {"exact", "mean_only", "point"}
        What its draws promise on a Gaussian posterior: the whole posterior, only its mean
        (mean-field), or a point estimate.
    metric_mode : {"layer", "native", "internal", "invariant"}
        How ``metric="dense"`` is served: by the pilot-whitening layer, by the family's own
        dense adaptation, by the runner itself (it delegates to another sampler), or not at
        all because the result is invariant under the whitening.
    advisory : str or None
        A warning emitted once per :func:`run_sampler` call, however many internal passes
        (a pilot, then the final run) the layers make.
    vmappable : bool
        True when :func:`run_chain_family`-style traced use under ``vmap`` is supported.
    """

    name: str
    kind: str
    methods: tuple[str, ...]
    run: Callable
    fidelity: str = "exact"
    metric_mode: str = "layer"
    advisory: str | None = None
    vmappable: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"sampler {self.name!r}: kind {self.kind!r} not in {KINDS}")
        if self.metric_mode not in METRIC_MODES:
            raise ValueError(f"sampler {self.name!r}: metric_mode {self.metric_mode!r}")
        if self.fidelity not in FIDELITIES:
            raise ValueError(
                f"sampler {self.name!r}: fidelity {self.fidelity!r} not in {FIDELITIES}"
            )


_REGISTRY: dict[str, Sampler] = {}


def register_sampler(sampler: Sampler) -> Sampler:
    """Add ``sampler`` to the registry and return it."""
    _REGISTRY[sampler.name] = sampler
    return sampler


def _ensure_registered() -> None:
    from tengri.inference.sampling import samplers  # noqa: F401  (registers on import)


def get_sampler(name: str) -> Sampler:
    """Return the sampler registered under a driver name or a method name.

    Raises
    ------
    ValueError
        If ``name`` is neither.
    """
    _ensure_registered()
    if name in _REGISTRY:
        return _REGISTRY[name]
    for sampler in _REGISTRY.values():
        if name in sampler.methods:
            return sampler
    raise ValueError(f"unknown sampler {name!r}; registered: {sorted(_REGISTRY)}")


def sampler_names() -> tuple[str, ...]:
    """Return every registered driver name."""
    _ensure_registered()
    return tuple(_REGISTRY)


def method_map() -> dict[str, str]:
    """Return ``method name -> driver name`` for every registered sampler."""
    _ensure_registered()
    return {m: s.name for s in _REGISTRY.values() for m in s.methods}


def resolve_cfg(cfg: dict) -> dict:
    """Return ``DEFAULTS`` updated with ``cfg``, refusing unknown keys by name."""
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise TypeError(f"unknown sampler setting(s) {unknown}; accepted: {sorted(DEFAULTS)}")
    return {**DEFAULTS, **cfg}


def _usable_rows(result: SamplerResult):
    """Flatten a pilot result to ``(rows, mask)`` of finite, non-divergent draws."""
    draws = result.draws.reshape(-1, result.draws.shape[-1])
    mask = jnp.all(jnp.isfinite(draws), axis=-1)
    if result.divergent is not None:
        mask = mask & ~result.divergent.reshape(-1)
    return draws, mask


def run_sampler(
    name: str,
    target: FlatTarget,
    *,
    key,
    n_chains: int = 1,
    metric: str = "diag",
    vmap_items: bool = False,
    devices=None,
    **cfg,
) -> SamplerResult:
    """Run the sampler ``name`` on ``target`` with the shared ``n_chains`` and metric layers.

    Parameters
    ----------
    name : str
        Driver or method name of a registered sampler.
    target : FlatTarget
        The problem.
    key : PRNGKey
        Random key.
    n_chains : int
        Chains. Chain families adapt once and vmap the chains; others run independent
        chains from overdispersed starts.
    metric : {"diag", "dense"}
        ``"dense"`` uses the family's own dense adaptation where it has one and a pilot
        covariance whitening otherwise. Samplers with no metric record that it was not
        applied.
    vmap_items : bool
        Reserved for engines that map the sampler over items (galaxies); a single flat
        target is one item.
    devices : sequence or None
        Reserved for engines that shard items across devices.
    **cfg
        Sampler settings, see :data:`DEFAULTS`.

    Returns
    -------
    SamplerResult
        ``draws`` of shape ``(n_chains, n, D)`` and the diagnostics, which carry
        ``metric_requested``, ``metric_applied`` and ``metric_fallback``.

    Raises
    ------
    ValueError
        For an unknown sampler, metric or ``n_chains < 1``.
    TypeError
        For a setting no sampler reads.
    """
    if metric not in ("diag", "dense"):
        raise ValueError(f"metric must be 'diag' or 'dense', got {metric!r}")
    n_chains = int(n_chains)
    if n_chains < 1:
        raise ValueError(f"n_chains must be >= 1, got {n_chains}")
    if vmap_items and not isinstance(target, FlatTarget):
        raise ValueError("vmap_items needs a FlatTarget")
    sampler = get_sampler(name)
    full = resolve_cfg(cfg)
    if sampler.advisory:
        warnings.warn(sampler.advisory, UserWarning, stacklevel=2)
    if metric == "diag" or sampler.metric_mode != "layer":
        result = sampler.run(target, key, n_chains, metric, full)
        applied = "diag" if metric == "diag" else sampler.metric_mode
        return _tagged(result, metric, applied, False)
    pilot_key, final_key = jax.random.split(key)
    pilot = sampler.run(target, pilot_key, 1, "diag", full)
    rows, mask = _usable_rows(pilot)
    mu, factor, fallback = pilot_whitening(rows, mask)
    if bool(fallback):
        # A pilot that cannot define a metric (too few usable draws, or a stuck chain) leaves
        # the diagonal run; the fallback is reported, never silent.
        result = sampler.run(target, final_key, n_chains, "diag", full)
        return _tagged(result, metric, "none", True)
    whitened, back = target.whiten(mu, factor)
    result = sampler.run(whitened, final_key, n_chains, "diag", full)
    result = dataclasses.replace(result, draws=back(result.draws))
    return _tagged(result, metric, "pilot", False)


def _tagged(result: SamplerResult, requested: str, applied: str, fallback: bool) -> SamplerResult:
    """Return ``result`` with the metric record added to its diagnostics."""
    record = {
        "metric_requested": requested,
        "metric_applied": applied,
        "metric_fallback": bool(fallback),
    }
    return dataclasses.replace(result, diagnostics={**result.diagnostics, **record})
