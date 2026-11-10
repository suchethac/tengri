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


@dataclass(frozen=True)
class FirstOrderConfig:
    """Static statics of the first-order family (Barker and MALA).

    Parameters
    ----------
    proposal : str
        ``"barker"`` or ``"mala"``.
    target_accept_rate : float
        Target acceptance rate of the dual-averaging step-size adaptation.
    """

    proposal: str
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
        ``kernel(cfg)`` returning the BlackJAX kernel for the statics; cached per family.
    init : callable
        ``init(position, log_density, key, cfg)`` returning the initial sampler state;
        families that take no key or no statics ignore them.
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
        ``step_size`` and ``inverse_mass_matrix``, and ``alpha`` and ``delta`` when the
        family tunes them.
    ensemble : bool
        True when ``state`` is a batch of warmed states, one per sampling chain, that
        the sampling chains start from (MEADS); False when it is one state.
    """

    name: str
    run: Any
    ensemble: bool = False


class CompilePlan(NamedTuple):
    """The statics a precompilation pass warms for one registered family.

    Parameters
    ----------
    kwargs : dict
        Resolved runner kwargs, passed to ``BackendEntry.adaptation_for``.
    adapt_cfg : object
        Statics of the warmup program (and of the fused program when ``fused``).
    sample_cfg : object or None
        Statics of the sampling program; None for families that read none.
    fused : bool
        True to warm one fused ``run_chain`` program instead of ``adapt_chain``
        followed by ``sample_chain``.
    """

    kwargs: dict
    adapt_cfg: Any
    sample_cfg: Any
    fused: bool = False


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


def _nuts_init(position, ld, key=None, cfg=None):
    """Return the NUTS state at ``position``; the key is unused."""
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
    kernel=lambda cfg: _shared._get_nuts_kernel(),
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


def _hmc_init(position, ld, key=None, cfg=None):
    """Return the HMC state at ``position``; the key is unused."""
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
    kernel=lambda cfg: _shared._get_hmc_kernel(),
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


def _dynamic_hmc_init(position, ld, key, cfg=None):
    """Return the dynamic HMC state at ``position``, seeded from ``key``."""
    import blackjax

    return blackjax.mcmc.dynamic_hmc.init(position, ld, key)


def _dynamic_hmc_step(kernel, state, key, ld, params, cfg):
    """Advance dynamic HMC one step; the trajectory length is drawn per step, ``cfg`` unused."""
    step_size, inv_mass_matrix = params
    return kernel(key, state, ld, step_size, inv_mass_matrix)


def _dynamic_hmc_collect(state, info) -> Draw:
    """Collect the position and divergence flag of a dynamic HMC step."""
    return Draw(position=state.position, divergent=info.is_divergent, extras={})


#: Integration steps used to tune dynamic HMC, not to sample with. ``dynamic_hmc.init``
#: needs a random generator that window adaptation cannot supply, so the step size and
#: mass matrix come from plain HMC window adaptation at this fixed trajectory length.
DYNAMIC_HMC_TUNING_LEAPFROG_STEPS = 10

#: Dynamic HMC sampler: the trajectory length is drawn each step. Starts from a fresh
#: state at ``init_flat`` built with the caller's init key, not from the warmup state.
DYNAMIC_HMC = ChainProgram(
    name="dynamic_hmc",
    kernel=lambda cfg: _shared._get_dynamic_hmc_kernel(),
    init=_dynamic_hmc_init,
    step=_dynamic_hmc_step,
    collect=_dynamic_hmc_collect,
    starts_from_warmup=False,
)


@dataclass(frozen=True)
class GHMCConfig:
    """Static statics of the GHMC family.

    Parameters
    ----------
    alpha : float or None
        Momentum persistence. A float is a static argument of the kernel; None reads the
        traced value from the tuned parameters.
    delta : float or None
        Slice-translation scaling. Same convention as ``alpha``.
    use_dense : bool
        Must be False: GHMC's momentum generator takes a diagonal inverse scale.
    target_accept_rate : float or None
        Target acceptance rate of the window adaptation; unused by MEADS.
    n_ensemble : int
        MEADS ensemble width; unused by the window adaptation.
    n_folds : int
        MEADS folds K.
    jitter_scale : float
        Gaussian dispersion of the MEADS ensemble around the seed position.
    low_rank_rank : int or None
        MEADS-LRD rank; None adapts the diagonal metric.
    low_rank_window_fraction : float
        Fraction of warmup over which the low-rank covariance accumulates.
    """

    alpha: float | None
    delta: float | None
    use_dense: bool = False
    target_accept_rate: float | None = None
    n_ensemble: int = 0
    n_folds: int = 4
    jitter_scale: float = 0.0
    low_rank_rank: int | None = None
    low_rank_window_fraction: float = 0.5

    def __post_init__(self):
        """Refuse a dense metric: GHMC's momentum generator is diagonal."""
        if self.use_dense:
            raise ValueError("GHMC refuses use_dense=True: its momentum generator is diagonal")


def _ghmc_init(position, ld, key, cfg=None):
    """Return the GHMC state at ``position``, with momentum drawn from ``key``."""
    import blackjax

    # Keyword args: blackjax reordered ghmc.init's (rng_key, logdensity_fn) between 1.3 and 1.6.
    return blackjax.mcmc.ghmc.init(position=position, logdensity_fn=ld, rng_key=key)


def _ghmc_step(kernel, state, key, ld, params, cfg):
    """Advance GHMC one step; alpha and delta come from ``cfg`` when pinned, else ``params``."""
    step_size, momentum_inv_scale = params[0], params[1]
    alpha = params[2] if cfg.alpha is None else cfg.alpha
    delta = params[3] if cfg.delta is None else cfg.delta
    return kernel(key, state, ld, step_size, momentum_inv_scale, alpha, delta)


def _ghmc_collect(state, info) -> Draw:
    """Collect the position and divergence flag of a GHMC step."""
    return Draw(position=state.position, divergent=info.is_divergent, extras={})


#: Generalized HMC sampler. The kernel is the cached BlackJAX GHMC kernel.
GHMC = ChainProgram(
    name="ghmc",
    kernel=lambda cfg: _shared._get_ghmc_kernel(),
    init=_ghmc_init,
    step=_ghmc_step,
    collect=_ghmc_collect,
    starts_from_warmup=False,
)

#: Integration steps of the HMC window that tunes GHMC's diagonal metric.
GHMC_TUNING_LEAPFROG_STEPS = 10

#: Window adaptation for GHMC: diagonal HMC window adaptation, since GHMC's momentum
#: generator takes a diagonal metric. Sampling re-initializes at ``init_flat``.
WINDOW_GHMC = WINDOW_HMC(GHMC_TUNING_LEAPFROG_STEPS, True)


def _meads_run(warmup_key, init_flat, ld, n_warmup, cfg, info_fn):
    """Run MEADS over a jittered ensemble; return the warmed ensemble, its parameters and info.

    The key is split into the ensemble-dispersion key and the MEADS adaptation key here,
    so the runner never splits keys itself.
    """
    import blackjax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    jitter_key, adapt_key = jax.random.split(warmup_key)
    ensemble = init_flat[None, :] + cfg.jitter_scale * jax.random.normal(
        jitter_key, (cfg.n_ensemble, init_flat.shape[0]), dtype=init_flat.dtype
    )
    # An empty info filter: the adapted values come back in ``parameters``, and keeping
    # every ensemble state per warmup step would cost n_warmup * n_ensemble * D floats.
    warmup = blackjax.meads_adaptation(
        ld,
        num_chains=cfg.n_ensemble,
        num_folds=cfg.n_folds,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        low_rank_rank=cfg.low_rank_rank,
        low_rank_window_fraction=cfg.low_rank_window_fraction,
    )
    (last_states, parameters), info = warmup.run(adapt_key, ensemble, num_steps=n_warmup)
    alpha = parameters["alpha"] if cfg.alpha is None else jnp.asarray(cfg.alpha)
    delta = parameters["delta"] if cfg.delta is None else jnp.asarray(cfg.delta)
    tuned = {
        "step_size": parameters["step_size"],
        "inverse_mass_matrix": parameters["momentum_inverse_scale"],
        "alpha": alpha,
        "delta": delta,
    }
    return last_states, tuned, info


#: MEADS adaptation (Hoffman and Sountsov 2022): the ensemble's warmed states become the
#: sampling chains' starting states, and alpha and delta are tuned from the ensemble.
MEADS = Adaptation(name="meads", run=_meads_run, ensemble=True)


def _first_order_init_state(position, ld, key, cfg):
    """Return the Barker or MALA state at ``position``; the key is unused."""
    return _shared._first_order_init(cfg.proposal, position, ld)


def _first_order_step(kernel, state, key, ld, params, cfg):
    """Advance a first-order sampler one step with the tuned step size; the mass is unused."""
    step_size, _ = params
    return kernel(key, state, ld, step_size)


def _first_order_collect(state, info) -> Draw:
    """Collect the position and Metropolis acceptance probability of a first-order step."""
    return Draw(position=state.position, divergent=None, extras={"accept": info.acceptance_rate})


#: Barker or MALA sampler, selected by ``cfg.proposal``. Metropolis-corrected, so a rejected
#: step is counted as acceptance rather than divergence: ``divergent`` is None.
FIRST_ORDER = ChainProgram(
    name="first_order",
    kernel=lambda cfg: _shared._get_first_order_kernel(cfg.proposal),
    init=_first_order_init_state,
    step=_first_order_step,
    collect=_first_order_collect,
    starts_from_warmup=True,
)


def _dual_averaging_run(warmup_key, init_flat, ld, n_warmup, cfg, info_fn):
    """Run dual averaging on the step size from ``init_flat``; return the warmed state.

    The warmup key is split into ``n_warmup`` step keys here, so the runner never splits
    keys itself. The mass is the identity, which is why ``inverse_mass_matrix`` is None.
    """
    from blackjax.adaptation.step_size import dual_averaging_adaptation

    kernel = _shared._get_first_order_kernel(cfg.proposal)
    da_init, da_update, da_final = dual_averaging_adaptation(cfg.target_accept_rate)
    state = _first_order_init_state(init_flat, ld, None, cfg)
    warmup_keys = jax.random.split(warmup_key, n_warmup)

    def _warmup_step(carry, k):
        """Advance one kernel step and feed its acceptance to dual averaging."""
        s, da = carry
        step_size = jnp.exp(da.log_step_size)
        s, info = kernel(k, s, ld, step_size)
        da = da_update(da, info.acceptance_rate)
        return (s, da), info.acceptance_rate

    (state, da_state), _ = jax.lax.scan(
        _warmup_step, (state, da_init(_shared.FIRST_ORDER_INITIAL_STEP_SIZE)), warmup_keys
    )
    parameters = {"step_size": da_final(da_state), "inverse_mass_matrix": None}
    return state, parameters, None


#: Dual-averaging step-size adaptation for the first-order family. Adapts the step size
#: only, with the identity mass matrix, and reports no adaptation info.
DUAL_AVERAGING = Adaptation(name="dual_averaging", run=_dual_averaging_run)


def _adapt_body(
    program, adaptation, init_flat, warmup_key, ld2, data_args, n_warmup, cfg, keep_info
):
    """Adapt from ``init_flat`` and return the final state, parameters and adaptation info."""
    ld = bind_logdensity(ld2, data_args, wall=program.wall)
    state, parameters, info = adaptation.run(
        warmup_key, init_flat, ld, n_warmup, cfg, _adapt_info_fn(keep_info)
    )
    params = (parameters["step_size"], parameters["inverse_mass_matrix"])
    if "alpha" in parameters:
        params += (parameters["alpha"], parameters["delta"])
    return state, params, info


def _sample_body(program, state, chain_keys, ld, params, cfg):
    """Scan the sampler over ``chain_keys`` from ``state`` and return the stacked draws."""
    kernel = program.kernel(cfg)

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
    if info is None:
        return params, None, None
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


@functools.partial(jax.jit, static_argnums=(0, 1, 6, 8, 9))
def run_chain(
    program,
    adaptation,
    init_flat,
    warmup_key,
    init_key,
    chain_keys,
    ld2,
    data_args,
    n_warmup,
    cfg,
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
    init_key : PRNGKey or None
        Key of the sampler state, read only by families that start from a fresh state
        (``starts_from_warmup=False``); None otherwise.
    chain_keys : ndarray, shape (n_iter, 2) or (n_chains, n_iter, 2)
        One key per sampling step, used as given. The three-dimensional form, for an
        ensemble adaptation, runs one chain per leading index from the first ``n_chains``
        warmed states.
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
        Sampling outputs with leading axis ``n_iter``, or ``(n_chains, n_iter)`` for the
        ensemble form.
    params : tuple
        ``(step_size, inverse_mass_matrix)`` tuned by the adaptation.
    """
    if adaptation.ensemble and program is not GHMC:
        raise ValueError(
            f"the {adaptation.name} ensemble adaptation runs only with GHMC, not {program.name}"
        )
    ld = bind_logdensity(ld2, data_args, wall=program.wall)
    state, params, _ = _adapt_body(
        program, adaptation, init_flat, warmup_key, ld2, data_args, n_warmup, cfg, False
    )
    if adaptation.ensemble:
        n_chains = chain_keys.shape[0]
        states = jax.tree.map(lambda leaf: leaf[:n_chains], state)
        draws = jax.vmap(lambda s, k: _sample_body(program, s, k, ld, params, cfg))(
            states, chain_keys
        )
        return draws, params
    if not program.starts_from_warmup:
        state = program.init(init_flat, ld, init_key, cfg)
    return _sample_body(program, state, chain_keys, ld, params, cfg), params
