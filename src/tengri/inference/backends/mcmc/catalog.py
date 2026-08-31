# SPDX-License-Identifier: BSD-3-Clause
"""Vectorized per-galaxy MCMC engine for :class:`~tengri.inference.catalog_fitter.CatalogFitter`.

Catalog fitting wants *sampled* per-galaxy posteriors, not a single point
estimate or a variational approximation: each galaxy is an independent,
low-dimensional target that is cheap to sample properly. This module builds a
single ``run_one(init_flat, key, data, noise)`` callable that runs BlackJAX
window adaptation **plus** sampling for one galaxy entirely inside JIT, so
``jax.lax.map(run_one, ..., batch_size=K)`` executes K galaxies' NUTS/HMC
chains in parallel on the accelerator while the compiled program stays O(1) in
the catalog size N.

The forward model (log-posterior) and its large shared inputs (SSP grid,
template data, fixed values, threaded through ``data_args["_jit_inputs"]``)
are identical across galaxies and are captured once; only each galaxy's
observed ``data`` / ``noise`` vary per ``lax.map`` step. This mirrors the
native-VI catalog engine in :mod:`tengri.inference.backends.vi.native`.

Unlike :func:`tengri.inference.backends.mcmc.nuts.run_nuts`, warmup is run
**per galaxy** (each galaxy adapts its own step size and mass matrix), which is
the statistically correct choice for a catalog of galaxies with different SEDs
and therefore different posterior geometries.

Three adaptation conventions exist in this codebase, and they must not be
conflated
--------------------------------------------------------------------------
1. **Per-galaxy adaptation inside the vmap** — what this module does, for all
   three samplers it carries. Each lane adapts to its own posterior.
2. **One adaptation on the first galaxy, reused for all** —
   ``Fitter._fit_batch_vmap_mcmc``. Cheaper, and *wrong for a catalog*: every
   galaxy's step size would then be a function of whichever galaxy happened to
   be first, so its posterior would depend on batch composition.
3. **Adaptation across a chain ensemble** — ChEES (and MEADS). This is a third
   axis, not a replacement for the first: the ensemble is
   **chains-within-galaxy**, so convention 1 still holds and the ensemble sits
   *inside* each lane. See
   ``_shared._resolve_chees_ensemble`` for
   why an ensemble spanning galaxies is refused.

Convention 1 is a real lock-step hazard for NUTS and is **not** one for ChEES,
and the difference is structural rather than a matter of degree. NUTS's window
adaptation scans a kernel whose per-step cost is a data-dependent
``lax.while_loop`` tree doubling, so a vmapped batch pays the deepest tree any
lane asks for, at every step, throughout warmup. ChEES's adaptation is a
fixed-length scan of ``n_warmup`` steps, each bounded statically by
``max_leapfrog_steps``; lanes may differ in their adapted ``L``, so the batch
still runs to the widest lane, but that width is a compile-time constant rather
than ``2**depth``. ``bench/reports/2026-08-31_catalog_batched_samplers.md``
measures both.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.inference.backends.mcmc._shared import (
    _CHEES_JITTER_SCALE,
    _CHEES_LEARNING_RATE,
    DEFAULT_MAX_NUM_DOUBLINGS,
    _chees_scan,
    _get_flat_logdensity,
    _hmc_full_scan,
    _nuts_full_scan,
    _resolve_chees_ensemble,
)
from tengri.inference.likelihoods.gaussian import inv_noise_std

_SAMPLERS = ("nuts", "hmc", "chees")

#: Adaptation-ensemble width for a *catalog* ChEES fit.
#:
#: Deliberately below the single-fit
#: ``_shared._CHEES_DEFAULT_ENSEMBLE`` of 32,
#: and the reason is arithmetic rather than taste. The ensemble is
#: chains-within-galaxy, so it is an *inner* axis under the galaxy vmap: a
#: catalog cell at ``forward_chunk_size=K`` carries ``K * n_ensemble`` live
#: chains through every adaptation step. At K = 128 an ensemble of 32 is 4096
#: concurrent chains for adaptation alone, which is a VRAM question before it is
#: a statistics one. 8 is the smallest power of two comfortably above
#: ``_shared._CHEES_MIN_ENSEMBLE`` (4), the
#: floor below which the cross-chain centered positions ChEES differentiates
#: collapse toward zero and the trajectory length stops adapting.
#:
#: This is a **default, not a cap**: pass ``n_ensemble=`` to ``run`` to raise it,
#: and expect the VRAM to scale with ``K * n_ensemble``.
CATALOG_CHEES_ENSEMBLE = 8

#: ADAM steps for the per-galaxy MAP warm start. 300 matches the single-fit
#: default in :func:`~tengri.inference.backends.map_dispatch.run_map`.
DEFAULT_MAP_INIT_STEPS = 300

#: Learning rate for that warm start, in the standardized (N(0,1)-prior) space
#: the flat vector lives in, so one value suits every parameter.
DEFAULT_MAP_INIT_LR = 0.05


def _make_substitute(template_data_args, thread_redshift: bool, thread_line_fluxes: bool):
    """Build the per-galaxy ``data_args`` substitution used by every catalog kernel.

    Substituting this galaxy's data into the shared template is what keeps the
    log-posterior receiving exactly the pytree it was built for: the shared
    ``_jit_inputs`` (SSP grid, templates) stay captured and only the observation
    varies per ``lax.map`` step. It lives in one place so the MAP warm start and
    the sampler cannot disagree about what a galaxy's data is.

    Parameters
    ----------
    template_data_args : dict
        The shared ``data_args`` pytree the flat log-posterior was built for.
    thread_redshift, thread_line_fluxes : bool
        Build-time Python bools, so the branches resolve during tracing: the
        per-galaxy ``redshift`` (#1337) and line-flux arrays (#1480) only reach
        ``data_args`` for catalogs that actually supply them.

    Returns
    -------
    callable
        ``(data, noise, presence, redshift, line_flux_obs, line_flux_err) -> dict``.

    Notes
    -----
    **JIT/vmap-compatible.**
    """

    def substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err):
        data_args = dict(template_data_args)
        data_args["data"] = data
        data_args["noise"] = noise
        data_args["sqrt_noise_inv"] = inv_noise_std(noise)
        # Per-galaxy presence masks (0/1) enable heterogeneous catalogs.
        data_args["presence"] = presence
        if thread_redshift:
            data_args["redshift"] = redshift
        if thread_line_fluxes:
            data_args["line_flux_obs"] = line_flux_obs
            data_args["line_flux_err"] = line_flux_err
        return data_args

    return substitute


def build_catalog_map_init(
    fitter,
    *,
    n_steps: int = DEFAULT_MAP_INIT_STEPS,
    learning_rate: float = DEFAULT_MAP_INIT_LR,
    thread_redshift: bool = False,
    thread_line_fluxes: bool = False,
):
    """Build a vmap-safe per-galaxy MAP warm start for the catalog samplers.

    The single-galaxy samplers start their warmup from a MAP estimate
    (``_maybe_map_init`` in :mod:`~tengri.inference.backends.mcmc.hmc`); the
    catalog path started every galaxy from ``0.1 * N(0, 1)`` about the prior
    center instead. That asymmetry is #1529's failure shape: "killed six of
    eight NUTS fits with R-hat up to 10.74 and zero divergences", and it is
    measurable here: on an identical model, galaxy and settings, the catalog
    path returned split R-hat 1.47 where the single-fit path returned 1.04.

    Each galaxy gets its own independent ADAM descent on its own posterior, so
    the warm start vectorizes exactly like the sampler that follows it.

    Parameters
    ----------
    fitter : Fitter
        Template fitter for the shared model, as for
        :func:`build_catalog_mcmc_engine`.
    n_steps : int, optional
        ADAM steps per galaxy.
    learning_rate : float, optional
        ADAM learning rate, in the standardized flat space.
    thread_redshift, thread_line_fluxes : bool, optional
        As for :func:`build_catalog_mcmc_engine`.

    Returns
    -------
    callable
        ``map_init_one(init_flat, data, noise, presence, redshift,
        line_flux_obs, line_flux_err) -> ndarray, shape (n_dim,)``, the warm
        start for that galaxy.

    Notes
    -----
    **JIT/vmap-compatible.** A galaxy whose descent leaves the finite domain
    keeps its original random start: the optimizer is a convenience, and it must
    never turn a usable initial point into a NaN one.
    """
    import optax

    init_params = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    log_posterior_flat_2arg, _unravel_fn, _init_flat, template_data_args = _get_flat_logdensity(
        fitter, init_params
    )
    substitute = _make_substitute(template_data_args, thread_redshift, thread_line_fluxes)
    optimizer = optax.adam(learning_rate)

    def map_init_one(init_flat, data, noise, presence, redshift, line_flux_obs, line_flux_err):
        data_args = substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err)

        def objective(x):
            return -log_posterior_flat_2arg(x, data_args)

        def step(carry, _):
            x, opt_state = carry
            grads = jax.grad(objective)(x)
            # A non-finite gradient must not poison the position: hold still for
            # that step rather than propagating NaN into the warm start.
            grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(x, updates), opt_state), None

        (x_final, _), _ = jax.lax.scan(
            step, (init_flat, optimizer.init(init_flat)), None, length=n_steps
        )
        keep = jnp.all(jnp.isfinite(x_final))
        return jnp.where(keep, x_final, init_flat)

    return map_init_one


def build_catalog_mcmc_engine(
    fitter,
    sampler: str,
    *,
    n_warmup: int,
    n_burnin: int,
    n_samples: int,
    max_num_doublings: int = DEFAULT_MAX_NUM_DOUBLINGS,
    n_leapfrog: int = 10,
    target_accept_rate: float = 0.85,
    use_dense: bool = False,
    thread_redshift: bool = False,
    thread_line_fluxes: bool = False,
    n_chains: int = 1,
    n_ensemble: int | str = CATALOG_CHEES_ENSEMBLE,
    ensemble_jitter: float = _CHEES_JITTER_SCALE,
    chain_jitter: float | None = None,
    max_leapfrog_steps: int = 200,
    chees_learning_rate: float = _CHEES_LEARNING_RATE,
    mass_matrix_estimation=None,
):
    """Build a vmap-safe per-galaxy NUTS/HMC/ChEES sampling callable.

    Parameters
    ----------
    fitter : Fitter
        A template :class:`~tengri.inference.fitter.Fitter` for the shared
        model. Only its structure is used, its log-posterior and the shared
        ``_jit_inputs`` are captured; per-galaxy ``data``/``noise`` are supplied
        at call time so the compiled program is reused across galaxies.
    sampler : {"nuts", "hmc", "chees"}
        Which BlackJAX sampler to vectorize.
    n_warmup : int
        Adaptation steps, run per galaxy. Window adaptation for ``"nuts"`` /
        ``"hmc"``; ChEES adaptation over this galaxy's own chain ensemble for
        ``"chees"``.
    n_burnin : int
        Post-warmup samples discarded (sliced inside the traced call, static).
    n_samples : int
        Posterior samples kept per galaxy.
    max_num_doublings : int, default DEFAULT_MAX_NUM_DOUBLINGS (10)
        NUTS tree depth cap (ignored for HMC). Shares the single-fit
        default and its rationale; see ``DEFAULT_MAX_NUM_DOUBLINGS`` in
        ``tengri.inference.backends.mcmc._shared``.
    n_leapfrog : int, default 10
        HMC leapfrog steps per proposal (ignored for NUTS).
    target_accept_rate : float, default 0.85
        Dual-averaging target acceptance rate.
    use_dense : bool, default False
        Dense vs diagonal mass matrix. Catalog sampling defaults to **diagonal**:
        each galaxy is low-D and the width parallelism is over galaxies, so a
        diagonal mass keeps the vmap flat and avoids the dense-mass warmup memory
        spike documented in :func:`tengri.inference.backends.mcmc.nuts.run_nuts`.
    thread_line_fluxes : bool, default False
        Whether to thread per-galaxy emission-line fluxes through the engine.
        When False, line fluxes are not used and the compiled program is
        unchanged. When True, per-galaxy line_flux_obs and line_flux_err arrays
        are substituted into data_args.
    n_chains : int, default 1
        Sampling chains **per galaxy**, concatenated into that galaxy's draws.
        ``"nuts"`` and ``"hmc"`` accept only 1 here; ChEES accepts more, which is
        what lets a catalog fit carry a split R-hat over genuinely separate
        chains rather than over two halves of one.
    n_ensemble : int or "auto", default :data:`CATALOG_CHEES_ENSEMBLE`
        ChEES adaptation-ensemble width, **within each galaxy**. Ignored by
        ``"nuts"`` / ``"hmc"``.
    ensemble_jitter : float
        Dispersion of that ensemble around the galaxy's seed position.
    chain_jitter : float or None, default None
        ``None`` seeds the sampling chains from the ensemble's warmed final
        states. A float seeds them independently and overdispersed, which is
        what makes a per-galaxy split R-hat a real test rather than a
        consistency check.
    max_leapfrog_steps : int, default 200
        Hard cap on ChEES's adapted trajectory length. **This is the parameter
        that keeps a vmapped ChEES batch bounded**: lanes with different adapted
        ``L`` batch to the widest one, and this is how wide that can get.
    chees_learning_rate : float
        Adam step on ChEES's ``log`` trajectory length.
    mass_matrix_estimation : None or "diagonal"
        ``None`` (default) leaves ChEES's metric at the identity. See
        :func:`~tengri.inference.backends.mcmc.chees.run_chees`.

    Returns
    -------
    run_one : callable
        ``(init_flat, key, data, noise, presence, redshift, line_flux_obs,
        line_flux_err) -> (positions, divergent)`` with ``positions`` shape
        ``(n_chains * n_samples, D)`` and ``divergent`` shape
        ``(n_chains * n_samples,)``. When
        ``thread_line_fluxes=False``, line_flux_obs and line_flux_err are ignored.
        Safe to wrap with ``jax.vmap`` / ``jax.lax.map``.
    unravel_fn : callable
        ``1D ndarray -> pytree`` for turning flat positions back into parameter
        dicts.

    Notes
    -----
    **JIT/vmap-compatible.** The returned ``run_one`` closes over the static
    log-posterior and the shared ``data_args`` template (including the big
    ``_jit_inputs`` arrays); each call substitutes the galaxy's ``data``/``noise``
    and its derived ``sqrt_noise_inv``. The underlying scan cores
    (``_nuts_full_scan`` / ``_hmc_full_scan``) are pre-JIT'd with ``data_args``
    traced, so different galaxies never trigger recompilation.
    """
    if sampler not in _SAMPLERS:
        raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}")

    n_chains = max(1, int(n_chains))
    if sampler != "chees" and n_chains != 1:
        raise ValueError(
            f"sampler={sampler!r} runs exactly one chain per galaxy in the catalog "
            f"path, got n_chains={n_chains}. Window adaptation is per galaxy here, so "
            "a second chain would have to re-run it; only the ChEES path adapts once "
            "over an ensemble and can then sample several chains from it. Use "
            "method='mcmc_chees' for a per-galaxy multi-chain R-hat."
        )
    ensemble_size = _resolve_chees_ensemble(n_ensemble, n_chains) if sampler == "chees" else 0

    init_params = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    log_posterior_flat_2arg, unravel_fn, _init_flat, template_data_args = _get_flat_logdensity(
        fitter, init_params
    )
    n_chain = n_burnin + n_samples

    substitute = _make_substitute(template_data_args, thread_redshift, thread_line_fluxes)

    def run_one(init_flat, gal_key, data, noise, presence, redshift, line_flux_obs, line_flux_err):
        data_args = substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err)

        warmup_key, chain_key = jax.random.split(gal_key)
        chain_keys = jax.random.split(chain_key, n_chain)

        if sampler == "chees":
            # This galaxy's OWN ChEES adaptation, over an ensemble of
            # ``ensemble_size`` chains that all target THIS galaxy's posterior.
            # Convention 1 + convention 3 of the module docstring: per-galaxy
            # adaptation, ensemble on the inner axis. Nothing here is shared with
            # any other lane, so a galaxy's draws do not depend on which galaxies
            # it was batched with -- the property that would be lost if the
            # ensemble were reused as the galaxy axis.
            #
            # ``chain_keys`` is reshaped to the (n_chains, n_iter, 2) layout
            # ``_chees_scan`` wants. Splitting per chain here rather than inside
            # keeps the key stream identical in shape to the NUTS/HMC lanes.
            ck = jax.random.split(chain_key, n_chains * n_chain).reshape(n_chains, n_chain, 2)
            positions, divergent, _step_size, _inv_mass, _n_leapfrog = _chees_scan(
                init_flat,
                warmup_key,
                ck,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                ensemble_size,
                n_chains,
                n_chain,
                float(ensemble_jitter),
                1.0,  # jitter_amount: BlackJAX's full Halton jitter
                target_accept_rate,
                max_leapfrog_steps,
                float(chees_learning_rate),
                mass_matrix_estimation,
                None if chain_jitter is None else float(chain_jitter),
            )
            # Burn-in is per chain (the ``_vmap_chains`` contract), so it is
            # sliced BEFORE the chains are flattened together. Slicing after
            # would drop the head of chain 0 and nothing from the rest.
            positions = positions[:, n_burnin:]
            divergent = divergent[:, n_burnin:]
            return positions.reshape(-1, positions.shape[-1]), divergent.reshape(-1)

        if sampler == "nuts":
            positions, divergent, _expansions, _step_size, _inv_mass = _nuts_full_scan(
                init_flat,
                warmup_key,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                max_num_doublings,
                use_dense,
                target_accept_rate,
                False,  # use_pathfinder_warmup
            )
        else:  # "hmc"
            positions, divergent, _step_size, _inv_mass = _hmc_full_scan(
                init_flat,
                warmup_key,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                n_leapfrog,
                use_dense,
                target_accept_rate,
            )

        # Discard per-galaxy burn-in (n_burnin is static → uniform shape).
        return positions[n_burnin:], divergent[n_burnin:]

    return run_one, unravel_fn
