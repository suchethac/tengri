# SPDX-License-Identifier: BSD-3-Clause
"""Chain families: the ``ChainProgram`` + ``Adaptation`` + statics of every registered chain.

One resolver for the statics (``chain_statics``) and one traced-safe runner
(``run_chain_family``) serve the flat driver and the vmapped catalog engine alike, so a
setting read here is read by every engine.
"""

from __future__ import annotations

import jax

#: Chain family -> the registry entry that owns its program and adaptation.
FAMILY_ENTRY: dict[str, str] = {
    "nuts": "mcmc_nuts",
    "nuts_pathfinder": "mcmc_nuts",
    "hmc": "mcmc_hmc",
    "dynamic_hmc": "mcmc_dynamic_hmc",
    "ghmc": "mcmc_ghmc",
    "barker": "mcmc_barker",
    "mala": "mcmc_mala",
}

#: Families whose adaptation has a dense metric of its own.
DENSE_FAMILIES = frozenset({"nuts", "nuts_pathfinder", "hmc", "dynamic_hmc"})


def resolve_accept(family: str, cfg: dict) -> float:
    """Return the dual-averaging target of ``family``: the user's value or the family default."""
    if cfg["target_accept_rate"] is not None:
        return float(cfg["target_accept_rate"])
    if family in ("barker", "mala"):
        from tengri.inference.backends.mcmc._shared import FIRST_ORDER_TARGET_ACCEPT_RATE

        return float(FIRST_ORDER_TARGET_ACCEPT_RATE)
    return 0.8


def chain_statics(family: str, cfg: dict, *, dense: bool = False):
    """Return the chain program, window adaptation and statics of one family.

    Parameters
    ----------
    family : str
        Key of :data:`FAMILY_ENTRY`.
    cfg : dict
        Resolved sampler settings (:data:`~tengri.inference.sampling.driver.DEFAULTS`).
    dense : bool
        Use the family's own dense metric (read only by :data:`DENSE_FAMILIES`).

    Returns
    -------
    program : ChainProgram
    adaptation : Adaptation
    statics : object
        The family's frozen config.
    """
    from tengri.inference._backend_registry import get_backend
    from tengri.inference.backends.mcmc.chain_program import (
        DYNAMIC_HMC,
        DYNAMIC_HMC_TUNING_LEAPFROG_STEPS,
        FIRST_ORDER,
        GHMC,
        HMC,
        NUTS,
        FirstOrderConfig,
        GHMCConfig,
        HMCConfig,
        NUTSConfig,
    )

    if family not in FAMILY_ENTRY:
        raise ValueError(f"no chain family {family!r}; families: {sorted(FAMILY_ENTRY)}")
    use_dense = bool(dense) and family in DENSE_FAMILIES
    accept = resolve_accept(family, cfg)
    if family in ("nuts", "nuts_pathfinder"):
        program = NUTS
        statics = NUTSConfig(
            max_doublings=cfg["max_num_doublings"], use_dense=use_dense, target_accept_rate=accept
        )
    elif family == "hmc":
        program = HMC
        statics = HMCConfig(
            n_leapfrog=cfg["n_leapfrog"], use_dense=use_dense, target_accept_rate=accept
        )
    elif family == "dynamic_hmc":
        program = DYNAMIC_HMC
        statics = HMCConfig(
            n_leapfrog=DYNAMIC_HMC_TUNING_LEAPFROG_STEPS,
            use_dense=use_dense,
            target_accept_rate=accept,
        )
    elif family == "ghmc":
        program = GHMC
        statics = GHMCConfig(
            alpha=cfg["ghmc_alpha"], delta=cfg["ghmc_delta"], target_accept_rate=accept
        )
    else:
        program = FIRST_ORDER
        statics = FirstOrderConfig(proposal=family, target_accept_rate=accept)
    entry = get_backend(FAMILY_ENTRY[family])
    adaptation = entry.adaptation_for(
        {
            "dense_mass_matrix": use_dense,
            "n_leapfrog_steps": cfg["n_leapfrog"],
            "pathfinder_warmstart": family == "nuts_pathfinder",
        },
        path="compile",
    )
    if adaptation is None:
        raise ValueError(f"registry entry {entry.name!r} has no compile adaptation rule")
    return program, adaptation, statics


def run_chain_family(
    family: str,
    ld2,
    data_args,
    init,
    warmup_key,
    chain_key,
    *,
    n_chains: int,
    n_chain: int,
    n_warmup: int,
    cfg: dict,
    dense: bool = False,
    chain_jitter: float | None = None,
):
    """Adapt once and sample ``n_chains`` chains of ``n_chain`` steps; traced-safe.

    Parameters
    ----------
    family : str
        Key of :data:`FAMILY_ENTRY`.
    ld2 : callable
        ``log_p(position, data_args)``, a stable-identity function.
    data_args : pytree
        Traced data.
    init : ndarray, shape (D,)
        Start position.
    warmup_key, chain_key : PRNGKey
        Keys of the adaptation and of the sampling streams.
    n_chains, n_chain, n_warmup : int
        Static counts. ``n_chain`` includes burn-in.
    cfg : dict
        Resolved sampler settings.
    dense : bool
        Use the family's own dense metric where it has one.
    chain_jitter : float or None
        Start dispersion of the extra chains; ``None`` takes the shared default.

    Returns
    -------
    draws : Draw
        Leading axis ``n_chain`` for one chain, ``(n_chains, n_chain)`` otherwise.
    params : tuple
        ``(step_size, inverse_mass_matrix)`` of the adaptation.
    """
    from tengri.inference.backends.mcmc.chain_program import run_chain
    from tengri.inference.sampling.chains import PRIOR_SCALE, UNADAPTED_START_SCALE

    program, adaptation, statics = chain_statics(family, cfg, dense=dense)
    if n_chains == 1:
        chain_keys = jax.random.split(chain_key, n_chain)
        jitter = None
    else:
        # Adaptation runs once; each chain gets its own key stream and its own start.
        chain_keys = jax.random.split(chain_key, n_chains * n_chain).reshape(n_chains, n_chain, 2)
        default = UNADAPTED_START_SCALE if family in ("barker", "mala") else PRIOR_SCALE
        jitter = float(default if chain_jitter is None else chain_jitter)
    init_key = None if program.starts_from_warmup else jax.random.fold_in(warmup_key, 1)
    return run_chain(
        program,
        adaptation,
        init,
        warmup_key,
        init_key,
        chain_keys,
        ld2,
        data_args,
        n_warmup,
        statics,
        jitter,
    )
