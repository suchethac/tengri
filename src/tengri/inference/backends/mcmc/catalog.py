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
template data, fixed values — threaded through ``data_args["_jit_inputs"]``)
are identical across galaxies and are captured once; only each galaxy's
observed ``data`` / ``noise`` vary per ``lax.map`` step. This mirrors the
native-VI catalog engine in :mod:`tengri.inference.backends.vi.native`.

Unlike :func:`tengri.inference.backends.mcmc.nuts.run_nuts`, warmup is run
**per galaxy** (each galaxy adapts its own step size and mass matrix), which is
the statistically correct choice for a catalog of galaxies with different SEDs
and therefore different posterior geometries.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.inference.backends.mcmc._shared import (
    DEFAULT_MAX_NUM_DOUBLINGS,
    _get_flat_logdensity,
    _hmc_full_scan,
    _nuts_full_scan,
)
from tengri.inference.likelihoods.gaussian import inv_noise_std

_SAMPLERS = ("nuts", "hmc")

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
    center instead. That asymmetry is #1529's failure shape — "killed six of
    eight NUTS fits with R-hat up to 10.74 and zero divergences" — and it is
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
        line_flux_obs, line_flux_err) -> ndarray, shape (n_dim,)`` — the warm
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
):
    """Build a vmap-safe per-galaxy NUTS/HMC sampling callable.

    Parameters
    ----------
    fitter : Fitter
        A template :class:`~tengri.inference.fitter.Fitter` for the shared
        model. Only its structure is used — its log-posterior and the shared
        ``_jit_inputs`` are captured; per-galaxy ``data``/``noise`` are supplied
        at call time so the compiled program is reused across galaxies.
    sampler : {"nuts", "hmc"}
        Which BlackJAX sampler to vectorize.
    n_warmup : int
        Window-adaptation steps, run per galaxy.
    n_burnin : int
        Post-warmup samples discarded (sliced inside the traced call — static).
    n_samples : int
        Posterior samples kept per galaxy.
    max_num_doublings : int, default DEFAULT_MAX_NUM_DOUBLINGS (10)
        NUTS tree depth cap (ignored for HMC). Shares the single-fit
        default and its rationale — see ``DEFAULT_MAX_NUM_DOUBLINGS`` in
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

    Returns
    -------
    run_one : callable
        ``(init_flat, key, data, noise, presence, redshift, line_flux_obs,
        line_flux_err) -> (positions, divergent)`` with ``positions`` shape
        ``(n_samples, D)`` and ``divergent`` shape ``(n_samples,)``. When
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
