# SPDX-License-Identifier: BSD-3-Clause
"""One adapted-chain runner for the MCMC families.

A chain is an adaptation followed by a sampling scan. The adaptation tunes the step
size and the mass matrix from a warmup of ``n_warmup`` steps; the sampling scan then
draws from the tuned kernel. Each MCMC family differs only in its kernel, its
adaptation and the quantities it collects per draw, so a family is described once as
a :class:`ChainProgram` and an :class:`Adaptation`, and the three jitted entries
below run any pair of them.

Seam: ``adapt_chain`` returns the tuned parameters, ``sample_chain`` draws from a
given state with given parameters, and ``run_chain`` fuses both. The runner takes
PRNG keys and never splits them, so the caller owns the key order and a run is
reproduced exactly by the same keys. The log density is passed as a static
two-argument function and ``data_args`` as a traced pytree, bound in one place by
:func:`bind_logdensity`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from tengri.inference.backends.mcmc import _shared


@dataclass(frozen=True)
class NUTSConfig:
    """Static statics of the NUTS family.

    Parameters
    ----------
    max_doublings : int
        Maximum tree doublings per step, for warmup and sampling alike.
    use_dense : bool
        Dense mass matrix when True, diagonal when False.
    target_accept_rate : float
        Target acceptance rate of the step-size adaptation.
    """

    max_doublings: int
    use_dense: bool
    target_accept_rate: float

    def sampling(self) -> NUTSSamplerConfig:
        """Return the sampling-scan config: the tree-depth cap, nothing else."""
        return NUTSSamplerConfig(max_doublings=self.max_doublings)


@dataclass(frozen=True)
class NUTSSamplerConfig:
    """Static statics of the NUTS sampling scan: the only field the sampler reads.

    Keeping the sampling scan's static argument to the tree-depth cap means a change
    in the adaptation's target acceptance rate or mass policy does not recompile it.

    Parameters
    ----------
    max_doublings : int
        Maximum tree doublings per sampling step.
    """

    max_doublings: int


@dataclass(frozen=True)
class HMCConfig:
    """Static statics of the HMC family.

    Parameters
    ----------
    n_leapfrog : int
        Leapfrog integration steps per proposal.
    use_dense : bool
        Dense mass matrix when True, diagonal when False.
    target_accept_rate : float
        Target acceptance rate of the step-size adaptation.
    """

    n_leapfrog: int
    use_dense: bool
    target_accept_rate: float


class Draw(NamedTuple):
    """Per-draw outputs of a sampling scan.

    Parameters
    ----------
    position : ndarray, shape (n_iter, D)
        Position after each step, in unbounded latent space.
    divergent : ndarray of bool, shape (n_iter,) or None
        Divergence flag per step; None for families that do not report one.
    extras : dict
        Family-specific per-step quantities.
    """

    position: Any
    divergent: Any
    extras: dict


@dataclass(frozen=True)
class ChainProgram:
    """The sampler half of a chain: kernel, state constructor, step and collection.

    Parameters
    ----------
    name : str
        Family name.
    kernel : callable
        Zero-argument getter returning the cached BlackJAX kernel.
    init : callable
        ``init(position, log_density)`` returning the initial sampler state.
    step : callable
        ``step(kernel, state, key, log_density, params, cfg)`` returning ``(state, info)``.
    collect : callable
        ``collect(state, info)`` returning a :class:`Draw` for the advanced state.
    starts_from_warmup : bool
        True when sampling continues from the adaptation's final state.
    wall : bool
        True when the log density maps non-finite values to walls.
    """

    name: str
    kernel: Any
    init: Any
    step: Any
    collect: Any
    starts_from_warmup: bool
    wall: bool = False


@dataclass(frozen=True)
class Adaptation:
    """The warmup half of a chain.

    Parameters
    ----------
    name : str
        Adaptation name.
    run : callable
        ``run(warmup_key, init_flat, log_density, n_warmup, cfg, info_fn)`` returning
        ``(state, parameters, info)`` with ``parameters`` a dict holding
        ``step_size`` and ``inverse_mass_matrix``.
    """

    name: str
    run: Any


def bind_logdensity(ld2, data_args, wall: bool = False):
    """Return the one-argument log density of ``position`` with ``data_args`` bound.

    Parameters
    ----------
    ld2 : callable
        ``log_p(position, data_args)``.
    data_args : pytree
        Observed data, traced under jit.
    wall : bool, optional
        Map non-finite values to walls with a zero gradient there.

    Returns
    -------
    callable
        ``log_p(position)``.
    """
    if wall:
        return _shared._walled_logdensity(ld2, data_args)

    def ld_1arg(position):
        return ld2(position, data_args)

    return ld_1arg


def divergence_record(divergent) -> dict:
    """Reduce per-step divergence flags to the diagnostics record.

    Parameters
    ----------
    divergent : ndarray of bool or None
        Divergence flag per step, or None when the family does not report one.

    Returns
    -------
    dict
        Empty for None, else ``n_divergent`` (count) and ``divergent_mask``.
    """
    if divergent is None:
        return {}
    return {"n_divergent": jnp.sum(divergent), "divergent_mask": divergent}


def _adapt_info_fn(keep_info: bool):
    """Return the BlackJAX adaptation-info filter, keeping divergence flags on request."""
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    if keep_info:
        return get_filter_adapt_info_fn(info_keys={"is_divergent", "num_integration_steps"})
    return get_filter_adapt_info_fn()


def _window_nuts(warmup_key, init_flat, ld, n_warmup, cfg, info_fn):
    """Run BlackJAX window adaptation around the NUTS kernel."""
    import blackjax

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        ld,
        is_mass_matrix_diagonal=not cfg.use_dense,
        target_acceptance_rate=cfg.target_accept_rate,
        adaptation_info_fn=info_fn,
        max_num_doublings=cfg.max_doublings,
    )
    (state, parameters), info = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    return state, parameters, info


def _pathfinder_nuts(warmup_key, init_flat, ld, n_warmup, cfg, info_fn):
    """Run BlackJAX Pathfinder adaptation around the NUTS kernel."""
    import blackjax
    from blackjax.adaptation.pathfinder_adaptation import pathfinder_adaptation

    warmup = pathfinder_adaptation(
        blackjax.nuts,
        ld,
        target_acceptance_rate=cfg.target_accept_rate,
        adaptation_info_fn=info_fn,
        max_num_doublings=cfg.max_doublings,
    )
    with _shared._bounded_pathfinder_elbo_draws():
        (state, parameters), info = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    return state, parameters, info


def _nuts_init(position, ld):
    """Return the NUTS state at ``position``."""
    import blackjax

    return blackjax.mcmc.nuts.init(position, ld)


def _nuts_step(kernel, state, key, ld, params, cfg):
    """Advance NUTS one step with the tuned step size and mass matrix."""
    step_size, inv_mass_matrix = params
    return kernel(key, state, ld, step_size, inv_mass_matrix, cfg.max_doublings)


def _nuts_collect(state, info) -> Draw:
    """Collect the position, divergence flag, tree depth, leapfrog count and energy."""
    return Draw(
        position=state.position,
        divergent=info.is_divergent,
        extras={
            "expansions": info.num_trajectory_expansions,
            "n_leapfrog": info.num_integration_steps,
            "energy": info.energy,
        },
    )


#: NUTS sampler. The kernel is the cached BlackJAX NUTS kernel, so it is identity-stable.
NUTS = ChainProgram(
    name="nuts",
    kernel=_shared._get_nuts_kernel,
    init=_nuts_init,
    step=_nuts_step,
    collect=_nuts_collect,
    starts_from_warmup=True,
)

#: Window adaptation: BlackJAX Stan-style windows over the NUTS kernel.
WINDOW_NUTS = Adaptation(name="window_nuts", run=_window_nuts)

#: Pathfinder adaptation: an L-BFGS Gaussian seeds the mass matrix and step size.
PATHFINDER_NUTS = Adaptation(name="pathfinder_nuts", run=_pathfinder_nuts)


def nuts_adaptation(pathfinder: bool) -> Adaptation:
    """Return the NUTS adaptation selected by the Pathfinder flag."""
    return PATHFINDER_NUTS if pathfinder else WINDOW_NUTS


def _hmc_init(position, ld):
    """Return the HMC state at ``position``."""
    import blackjax

    return blackjax.mcmc.hmc.init(position, ld)


def _hmc_step(kernel, state, key, ld, params, cfg):
    """Advance HMC one step with the tuned step size and mass matrix."""
    step_size, inv_mass_matrix = params
    return kernel(key, state, ld, step_size, inv_mass_matrix, cfg.n_leapfrog)


def _hmc_collect(state, info) -> Draw:
    """Collect the position and divergence flag of an HMC step."""
    return Draw(position=state.position, divergent=info.is_divergent, extras={})


#: HMC sampler with a fixed trajectory length. The kernel is the cached BlackJAX HMC kernel.
HMC = ChainProgram(
    name="hmc",
    kernel=_shared._get_hmc_kernel,
    init=_hmc_init,
    step=_hmc_step,
    collect=_hmc_collect,
    starts_from_warmup=True,
)


def _window_hmc_run(n_leapfrog, diag, warmup_key, init_flat, ld, n_warmup, cfg, info_fn):
    """Run BlackJAX window adaptation around the HMC kernel with ``n_leapfrog`` steps."""
    import blackjax

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        ld,
        is_mass_matrix_diagonal=diag,
        target_acceptance_rate=cfg.target_accept_rate,
        num_integration_steps=n_leapfrog,
        adaptation_info_fn=info_fn,
    )
    (state, parameters), info = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
    return state, parameters, info


@functools.cache
def WINDOW_HMC(n_leapfrog: int, diag: bool) -> Adaptation:
    """Return the HMC window adaptation with ``n_leapfrog`` steps and a diagonal or dense metric.

    Parameters
    ----------
    n_leapfrog : int
        Leapfrog steps per proposal during adaptation.
    diag : bool
        True for a diagonal mass matrix, False for a dense one.

    Returns
    -------
    Adaptation
        Cached per argument pair, so the same pair is the same static object under jit.
    """
    name = f"window_hmc_{n_leapfrog}_{'diag' if diag else 'dense'}"
    return Adaptation(name=name, run=functools.partial(_window_hmc_run, n_leapfrog, diag))


def _adapt_body(
    program, adaptation, init_flat, warmup_key, ld2, data_args, n_warmup, cfg, keep_info
):
    """Adapt from ``init_flat`` and return the final state, parameters and adaptation info."""
    ld = bind_logdensity(ld2, data_args, wall=program.wall)
    state, parameters, info = adaptation.run(
        warmup_key, init_flat, ld, n_warmup, cfg, _adapt_info_fn(keep_info)
    )
    params = (parameters["step_size"], parameters["inverse_mass_matrix"])
    return state, params, info


def _sample_body(program, state, chain_keys, ld, params, cfg):
    """Scan the sampler over ``chain_keys`` from ``state`` and return the stacked draws."""
    kernel = program.kernel()

    def _step(s, k):
        s, info = program.step(kernel, s, k, ld, params, cfg)
        return s, program.collect(s, info)

    _, draws = jax.lax.scan(_step, state, chain_keys)
    return draws


@functools.partial(jax.jit, static_argnums=(0, 1, 4, 6, 7))
def adapt_chain(
    program, adaptation, init_flat, warmup_key, ld2, data_args, n_warmup, cfg
) -> tuple[Any, Any, Any]:
    """Adapt the chain and return ``(params, warmup_divergent, warmup_n_grad)``.

    Parameters
    ----------
    program : ChainProgram (static)
        Sampler family.
    adaptation : Adaptation (static)
        Warmup strategy.
    init_flat : ndarray, shape (D,)
        Start position in unbounded latent space.
    warmup_key : PRNGKey
        Key of the adaptation.
    ld2 : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree
        Observed data, traced.
    n_warmup : int (static)
        Warmup steps.
    cfg : NUTSConfig (static)
        Family statics.

    Returns
    -------
    params : tuple
        ``(step_size, inverse_mass_matrix)``.
    warmup_divergent : ndarray of bool, shape (n_warmup,)
        Divergence flag per warmup step.
    warmup_n_grad : scalar int
        Leapfrog steps spent in the warmup.
    """
    _, params, info = _adapt_body(
        program, adaptation, init_flat, warmup_key, ld2, data_args, n_warmup, cfg, True
    )
    warmup_divergent = jnp.asarray(info.info.is_divergent)
    warmup_n_grad = jnp.sum(jnp.asarray(info.info.num_integration_steps))
    return params, warmup_divergent, warmup_n_grad


@functools.partial(jax.jit, static_argnums=(0, 3, 6))
def sample_chain(program, state, chain_keys, ld2, data_args, params, cfg) -> Draw:
    """Sample from ``state`` with fixed adaptation and return the stacked draws.

    Parameters
    ----------
    program : ChainProgram (static)
        Sampler family.
    state : sampler state
        Initial state, for example from ``program.init``.
    chain_keys : ndarray, shape (n_iter, 2)
        One key per step, used as given.
    ld2 : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree
        Observed data, traced.
    params : tuple
        ``(step_size, inverse_mass_matrix)`` from :func:`adapt_chain`.
    cfg : NUTSSamplerConfig (static)
        Sampling statics, only the fields the sampler reads.

    Returns
    -------
    Draw
        Positions, divergence flags and extras, each with leading axis ``n_iter``.
    """
    ld = bind_logdensity(ld2, data_args, wall=program.wall)
    return _sample_body(program, state, chain_keys, ld, params, cfg)


@functools.partial(jax.jit, static_argnums=(0, 1, 5, 7, 8))
def run_chain(
    program, adaptation, init_flat, warmup_key, chain_keys, ld2, data_args, n_warmup, cfg
) -> tuple[Draw, tuple]:
    """Adapt, then sample, in one program; return ``(draws, params)``.

    Parameters
    ----------
    program : ChainProgram (static)
        Sampler family.
    adaptation : Adaptation (static)
        Warmup strategy.
    init_flat : ndarray, shape (D,)
        Start position in unbounded latent space.
    warmup_key : PRNGKey
        Key of the adaptation.
    chain_keys : ndarray, shape (n_iter, 2)
        One key per sampling step, used as given.
    ld2 : callable (static)
        ``log_p(position, data_args)``.
    data_args : pytree
        Observed data, traced.
    n_warmup : int (static)
        Warmup steps.
    cfg : NUTSConfig (static)
        Family statics.

    Returns
    -------
    draws : Draw
        Sampling outputs with leading axis ``n_iter``.
    params : tuple
        ``(step_size, inverse_mass_matrix)`` tuned by the adaptation.
    """
    ld = bind_logdensity(ld2, data_args, wall=program.wall)
    state, params, _ = _adapt_body(
        program, adaptation, init_flat, warmup_key, ld2, data_args, n_warmup, cfg, False
    )
    if not program.starts_from_warmup:
        state = program.init(init_flat, ld)
    return _sample_body(program, state, chain_keys, ld, params, cfg), params
