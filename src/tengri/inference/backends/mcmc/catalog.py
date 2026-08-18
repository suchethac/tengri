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

from tengri.inference.backends.mcmc._shared import (
    DEFAULT_MAX_NUM_DOUBLINGS,
    _get_flat_logdensity,
    _hmc_full_scan,
    _nuts_full_scan,
)
from tengri.inference.likelihoods.gaussian import inv_noise_std

_SAMPLERS = ("nuts", "hmc")


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
        default and its rationale — see
        :data:`~tengri.inference.backends.mcmc._shared.DEFAULT_MAX_NUM_DOUBLINGS`.
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

    def run_one(init_flat, gal_key, data, noise, presence, redshift, line_flux_obs, line_flux_err):
        # Substitute this galaxy's data into the shared data_args template so the
        # log-posterior receives exactly the pytree it was built for — the shared
        # _jit_inputs (SSP grid, templates) stay captured, only data/noise/presence vary.
        # Per-galaxy presence masks (0/1) enable heterogeneous catalogs (missing bands).
        data_args = dict(template_data_args)
        data_args["data"] = data
        data_args["noise"] = noise
        data_args["sqrt_noise_inv"] = inv_noise_std(noise)
        data_args["presence"] = presence
        # Per-galaxy redshift override (#1337 phase 2). ``thread_redshift`` is a
        # build-time Python bool, so the branch resolves during tracing: the
        # ``redshift`` arg never reaches ``data_args`` for free / shared-redshift
        # catalogs, and only a per-galaxy Fixed-z catalog threads it.
        if thread_redshift:
            data_args["redshift"] = redshift
        # Per-galaxy line fluxes (#1480). ``thread_line_fluxes`` is a build-time
        # Python bool, so the branch resolves during tracing: line flux arrays only
        # reach data_args when a model carries line fluxes and the catalog supplies them.
        if thread_line_fluxes:
            data_args["line_flux_obs"] = line_flux_obs
            data_args["line_flux_err"] = line_flux_err

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
