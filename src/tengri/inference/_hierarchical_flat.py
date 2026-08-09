# SPDX-License-Identifier: BSD-3-Clause
"""The hierarchical model as a flat vector — one seam, every sampler.

``PopulationFitter`` used to accept 8 of the 20 registered backends. The other
12 raised ``ValueError``, not because hierarchical inference is incompatible
with them but because the only flat-vector formulation lived *inside*
``_run_raytrace`` as a set of closures, reachable by exactly one sampler.

This seam drives the NUTS / HMC / dynamic-HMC / GHMC / MAP / pathfinder
family. Every other registered name is either driven by ``PopulationFitter``'s
own runners or refused with a stated reason — see :data:`FLAT_UNSUPPORTED`. A
name is driven here only when the driver runs the algorithm the name promises;
a stand-in (the first draft ran plain HMC under five distinct-algorithm names)
is silent substitution, not support. ``nss`` is the founding entry of the
refused set:
the prior transform it would need is exact and is provided here; what is
missing is a real nested sampler on top of it, and a blind rejection stand-in
returns biased samples rather than an approximation (#1429).

This module lifts that formulation out. Once the problem is
``(init_flat, log_likelihood, log_prior, prior_transform)`` on an unconstrained
vector, every sampler in the registry is a plain function call.

The prior is the reason this is clean
-------------------------------------
The hierarchical parameterization is already **iid standard normal**. Every
bounded quantity is stored as an unconstrained latent and mapped through
:func:`~tengri.utils.transforms.to_bounded`, which is the Gaussian CDF, so an
N(0,1) latent yields a genuine Uniform(lo, hi) physical prior. The log posterior
is therefore separable by construction:

.. math::

    \\log p(u \\mid d) = \\underbrace{-\\tfrac{1}{2}\\chi^2(u)}_{\\log L}
                        \\underbrace{-\\tfrac{1}{2}\\lVert u \\rVert^2}_{\\log \\pi}

Two consequences that do the work here:

* Gradient samplers (NUTS, HMC, MCLMC, …) take ``log_prob = log_L + log_pi``
  directly — no bijector stack, no Jacobian bookkeeping.
* Nested sampling needs a unit-cube → prior map, and for an iid N(0,1) prior
  that is exactly the probit :func:`jax.scipy.special.ndtri`, applied
  elementwise. No prior-transform machinery had to be invented; it was implied
  by the parameterization all along.

Units
-----
``psd_sigma`` is dimensionless; ``psd_tau_myr`` is [Myr]. Both are stored
unconstrained here (``psd_sigma_u``, ``psd_tau_u``) and only become physical
inside the likelihood, via ``to_bounded``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.utils.transforms import to_bounded, to_unbounded

__all__ = ["FLAT_SAMPLERS", "FlatProblem", "build_flat_problem", "run_flat_sampler"]

#: Registry methods reachable through the flat seam, and how each is driven.
#:
#: Derived membership, never a hand-written prose list: ``PopulationFitter``
#: builds its supported-method set from this mapping, so wiring a sampler here
#: is the single edit that makes it available hierarchically. The literal that
#: preceded this named two backends and went stale the moment a third gained the
#: same capability (#1394).
#:
#: A name may appear here only when its driver runs the algorithm the name
#: promises. The first draft mapped five distinct-algorithm names (ESS, dynamic
#: HMC, GHMC, MCLMC, adjusted MCLMC) onto the plain static-leapfrog ``"hmc"``
#: driver and ``laplace`` onto the bare ``"map"`` point estimate — the result's
#: diagnostics recorded the requested name while a different algorithm ran.
#: Dynamic HMC and GHMC have since gained their real ``_shared.py`` full-scan
#: drivers and rejoined; the rest live in :data:`FLAT_UNSUPPORTED` until their
#: real drivers are wired.
FLAT_SAMPLERS: dict[str, str] = {
    "mcmc_nuts": "nuts",
    "mcmc_hmc": "hmc",
    "mcmc_dynamic_hmc": "dynamic_hmc",
    "mcmc_ghmc": "ghmc",
    # Pinned to NUTS, unlike the single-galaxy auto-pick (NUTS for low-D,
    # raytrace above D~20): hierarchical D grows with the catalog, and at that
    # D raytrace degenerates and raises DegenerateChainError by design
    # (#1530). Auto-picking it would make the generic name a guaranteed
    # failure.
    "mcmc": "nuts",
    "map": "map",
    "pathfinder": "nuts_pathfinder",
}

#: Registered backends that this seam deliberately does NOT drive, and why.
#: Listed so ``PopulationFitter`` can raise a specific error instead of a generic
#: "unknown method", and so the reason survives longer than a commit message.
FLAT_UNSUPPORTED: dict[str, str] = {
    "nss": (
        "nested sampling needs constrained exploration WITHIN the likelihood "
        "contour (slice/MCMC steps from a surviving live point). A blind "
        "rejection sampler drawn from the prior is not an acceptable stand-in: "
        "measured on a 2-galaxy D=18 problem it exhausted its attempt budget "
        "and terminated at iteration 147 of a requested 200, returning a "
        "silently truncated -- and therefore biased -- sample set. The prior "
        "transform this seam provides is exact and correct; what is missing is "
        "the sampler on top of it. See #1429."
    ),
    "mcmc_ess": (
        "elliptical slice sampling has no driver at this seam yet. The flat "
        "problem is tailor-made for it -- the prior is exactly iid N(0,1), "
        "which is the ellipse ESS needs (see backends/mcmc/elliptical_slice.py "
        "for the single-galaxy driver to adapt, honoring the data_args "
        "compile-reuse contract). Running the static-leapfrog HMC driver under "
        "this name instead would be silent substitution. Use mcmc_nuts or "
        "mcmc_hmc hierarchically, or run ESS per-galaxy through Fitter."
    ),
    "mcmc_mclmc": (
        "microcanonical Langevin MC needs its own (L, step size) adaptation, "
        "not the HMC window adaptation this seam runs. _mclmc_sample_scan in "
        "backends/mcmc/_shared.py provides the sampling half but not the "
        "tuning. Use mcmc_nuts or mcmc_hmc hierarchically."
    ),
    "mcmc_adjusted_mclmc": (
        "adjusted microcanonical Langevin MC needs its own (L, step size) "
        "adaptation, not the HMC window adaptation this seam runs. "
        "_adjusted_mclmc_sample_scan in backends/mcmc/_shared.py provides the "
        "sampling half but not the tuning. Use mcmc_nuts or mcmc_hmc "
        "hierarchically."
    ),
    "laplace": (
        "a Laplace approximation is MAP plus a Gaussian covariance from the "
        "curvature at a verified mode; this seam's map driver computes no "
        "covariance, so driving laplace with it would silently drop the error "
        "bars that distinguish laplace from map. Use ``map`` for the "
        "hierarchical point estimate, or run laplace per-galaxy through "
        "Fitter."
    ),
}


@dataclass(frozen=True)
class FlatProblem:
    """A hierarchical population fit expressed on one unconstrained vector.

    Attributes
    ----------
    init_flat : ndarray, shape (D,)
        MAP-initialized starting point in unconstrained space.
    unravel : callable
        ``ndarray (D,) -> pytree``, the inverse of the flattening.
    n_dim : int
        D — shared hyperparameters plus every galaxy's free parameters (plus the
        stochastic field latents when the SFH is stochastic).
    log_likelihood : callable
        ``ndarray (D,) -> scalar``. Gaussian data term only, no prior.
    log_prior : callable
        ``ndarray (D,) -> scalar``. Unnormalized iid standard normal.
    log_prob : callable
        ``ndarray (D,) -> scalar``. Their sum — the log posterior.
    prior_transform : callable
        ``ndarray (D,) in [0,1]^D -> ndarray (D,)``. Unit cube to N(0,1) latent,
        for nested sampling. Exact, because the prior is iid standard normal.
    extract_shared : callable
        ``ndarray (D,) -> ndarray (2,)`` giving physical
        ``(psd_sigma, psd_tau_myr)``.

    Notes
    -----
    JIT/grad/vmap-compatible: every callable is pure and closes only over arrays
    and Python constants. ``log_likelihood`` runs the forward model under
    ``jax.lax.map`` so the compiled graph stays O(1) in the number of galaxies.
    """

    init_flat: Any
    unravel: Callable
    n_dim: int
    log_likelihood: Callable
    log_prior: Callable
    log_prob: Callable
    prior_transform: Callable
    extract_shared: Callable
    #: ``(all_data, all_noise)`` — the observed vectors, kept OUT of the closure.
    data_args: Any = ()
    #: ``(flat, data_args) -> scalar``. The form samplers must use: with the data
    #: supplied as a traced argument, one compiled program serves every catalog.
    log_prob_with_data: Callable | None = None


def build_flat_problem(fitter, *, key, memory_mode="low", verbose=False, map_steps=80):
    """Flatten a :class:`PopulationFitter` into a sampler-agnostic problem.

    Parameters
    ----------
    fitter : PopulationFitter
        The hierarchical fit to flatten. Read-only — nothing is mutated.
    key : PRNGKey
        Used for the per-galaxy MAP initialization.
    memory_mode : {"low", "high"}
        ``"low"`` wraps the per-galaxy forward in :func:`jax.checkpoint`,
        trading recomputation for activation memory. This matters: the graph
        holds every galaxy's forward pass at once.
    verbose : bool
        Print initialization progress.
    map_steps : int
        Gradient steps for the per-galaxy MAP initialization.

    Returns
    -------
    FlatProblem

    Notes
    -----
    This is now the ONLY definition of the hierarchical posterior.
    ``_run_raytrace`` used to build its own ``init``, ``ravel_pytree`` and
    ``log_prob`` inline — ~135 lines textually equivalent to this function but
    structurally independent, so nothing prevented the two from drifting into
    sampling different distributions while every docstring claimed otherwise.
    It calls this builder instead, verified bit-for-bit: raytrace on a fixed key
    returned ``sigma=1.667, tau=154.89`` both before and after the change.

    So "every sampler targets the same posterior" is a structural property here,
    not a claim requiring anyone's diligence.
    """
    from tengri import Fitter
    from tengri.inference.backends.map_dispatch import build_vectorized_map_solver

    n_gal = fitter.n_galaxies
    spec = fitter._spec
    stochastic = spec.stochastic
    n_grid = spec.n_grid
    free_names = fitter._free_names
    bounds = {name: spec.get_distribution(name).bounds for name in free_names}
    fixed_values = spec.get_fixed_values()
    sigma_lo, sigma_hi = fitter.psd_sigma_bounds
    tau_lo, tau_hi = fitter.psd_tau_bounds
    data_type = fitter.data_type

    sigma_mid = 0.5 * (sigma_lo + sigma_hi)
    tau_mid = 0.5 * (tau_lo + tau_hi)

    # One model instance, shared by init and likelihood. PSD values are
    # overridden per-call inside the likelihood, so the ones baked in here are
    # only used for the MAP warm start.
    model = fitter.model_factory(psd_sigma=sigma_mid, psd_tau_myr=tau_mid)

    keys = jax.random.split(key, n_gal + 2)

    if verbose:
        print("  Initializing per-galaxy params via vectorized MAP...")

    _template_gal = fitter.galaxies[0]
    _template_fitter = Fitter(
        model,
        _template_gal["flux_obs"],
        _template_gal["noise"],
        data_type=data_type,
    )
    map_solve_one = build_vectorized_map_solver(
        _template_fitter, n_steps=map_steps, learning_rate=0.05
    )

    all_flux_init = jnp.stack([jnp.asarray(g["flux_obs"]) for g in fitter.galaxies])
    all_noise_init = jnp.stack([jnp.asarray(g["noise"]) for g in fitter.galaxies])

    all_init_unbounded = jax.lax.map(
        lambda args: map_solve_one(args[0], args[1], args[2]),
        (all_flux_init, all_noise_init, keys[:n_gal]),
        batch_size=1,
    )

    if verbose:
        print("  MAP initialization complete")

    init = {
        "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
        "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
        "gal": {name: all_init_unbounded.get(name, jnp.zeros(n_gal)) for name in free_names},
    }
    if stochastic:
        init["gal_xi"] = all_init_unbounded.get("psd_xi", jnp.zeros((n_gal, n_grid)))

    init_flat, unravel_fn = ravel_pytree(init)
    n_dim = int(init_flat.shape[0])

    all_data_arr = jnp.concatenate([jnp.asarray(g["flux_obs"]) for g in fitter.galaxies])
    all_noise_arr = jnp.concatenate([jnp.asarray(g["noise"]) for g in fitter.galaxies])

    def _predict(params):
        if data_type == "photometry":
            return model.predict_photometry(params)
        return model.predict_spectrum(params)

    def log_likelihood_with_data(flat_params, data_args):
        """Gaussian data term, -chi^2/2, with the data supplied as an ARGUMENT.

        Taking ``(all_data, all_noise)`` as a traced argument rather than
        capturing them is what lets one compiled NUTS program serve different
        galaxies. Capturing them bakes them in as constants, so every new
        catalog is a new program — measured at one full recompile per fit.
        """
        all_data, all_noise = data_args
        p = unravel_fn(flat_params)
        psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
        psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

        def forward_one(ub_scalars, xi):
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(ub_scalars[name], lo, hi)
            for name, val in fixed_values.items():
                if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                    params[name] = val
            params["sfh_field_psd_sigma"] = psd_sigma
            params["sfh_field_psd_tau_myr"] = psd_tau
            if stochastic:
                params["sfh_field_xi"] = xi
            params = spec.resolve_mirrors(params)
            return _predict(params)

        # lax.map keeps the compiled graph O(1) in N_gal.
        fwd = jax.checkpoint(forward_one) if memory_mode == "low" else forward_one
        if stochastic:
            predictions = jax.lax.map(lambda args: fwd(args[0], args[1]), (p["gal"], p["gal_xi"]))
        else:
            predictions = jax.lax.map(lambda ub: fwd(ub, None), p["gal"])

        chi2 = jnp.sum(((all_data - predictions.reshape(-1)) / all_noise) ** 2)
        return -0.5 * chi2

    _data_args = (all_data_arr, all_noise_arr)

    def log_likelihood(flat_params):
        """Convenience alias with this fit's data bound. NOT for samplers."""
        return log_likelihood_with_data(flat_params, _data_args)

    def log_prior(flat_params):
        """Unnormalized iid standard normal on every latent."""
        p = unravel_fn(flat_params)
        penalty = p["psd_sigma_u"] ** 2 + p["psd_tau_u"] ** 2
        for name in free_names:
            penalty += jnp.sum(p["gal"][name] ** 2)
        if stochastic:
            penalty += jnp.sum(p["gal_xi"] ** 2)
        return -0.5 * penalty

    def log_prob(flat_params):
        return log_likelihood(flat_params) + log_prior(flat_params)

    def log_prob_with_data(flat_params, data_args):
        return log_likelihood_with_data(flat_params, data_args) + log_prior(flat_params)

    def prior_transform(cube):
        """Unit cube -> N(0,1) latent. Exact for an iid standard normal prior.

        Clipped off the open interval's endpoints because ``ndtri(0)`` and
        ``ndtri(1)`` are -inf/+inf, which would poison a live point rather than
        merely sit at the boundary.
        """
        eps = jnp.finfo(jnp.asarray(cube).dtype).tiny
        return jax.scipy.special.ndtri(jnp.clip(cube, eps, 1.0 - jnp.finfo(float).eps))

    def extract_shared(flat_params):
        p = unravel_fn(flat_params)
        return jnp.array(
            [
                to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi),
                to_bounded(p["psd_tau_u"], tau_lo, tau_hi),
            ]
        )

    return FlatProblem(
        init_flat=init_flat,
        unravel=unravel_fn,
        n_dim=n_dim,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        log_prob=log_prob,
        prior_transform=prior_transform,
        extract_shared=extract_shared,
        data_args=_data_args,
        log_prob_with_data=log_prob_with_data,
    )


def run_flat_sampler(
    fitter,
    method,
    *,
    key,
    n_warmup=300,
    n_burnin=100,
    n_samples=500,
    n_leapfrog=10,
    max_num_doublings=5,
    dense_mass_matrix=False,
    target_accept_rate=0.8,
    memory_mode="low",
    map_steps=300,
    map_learning_rate=0.05,
    ghmc_alpha=0.8,
    ghmc_delta=0.65,
    allow_unvalidated=False,
    verbose=True,
    **_ignored,
):
    """Run any flat-seam-reachable sampler on a hierarchical population fit.

    Parameters
    ----------
    fitter : PopulationFitter
        The hierarchical problem.
    method : str
        Canonical registry name; must be a key of :data:`FLAT_SAMPLERS`.
    key : PRNGKey
    n_warmup, n_burnin, n_samples : int
        Window adaptation / discarded / retained chain lengths (MCMC drivers).
    n_leapfrog : int
        Leapfrog steps per HMC proposal.
    max_num_doublings : int
        NUTS tree depth cap; the trajectory may reach ``2**max_num_doublings``
        leapfrog steps, each a full ``grad(log_prob)`` over every galaxy.

        Defaults to **5** here against ``run_nuts``'s 10, because this is where
        the memory is. Measured on a 2-galaxy D=18 problem, marginal peak RSS
        over the built problem was 2.55 GB at 3, 2.58 GB at 5, then 3.95 GB at
        7 and 3.96 GB at 10 — a ~1.4 GB step between 5 and 7 that then
        saturates. The physics is not what costs: the SSP grid is 64 MB, the
        whole model plus one forward is 0.7 GB, and a single ``log_prob`` adds
        0.02 GB. Nearly all of the rest is the compiled NUTS trajectory buffer,
        sized for the deepest tree the sampler is allowed to build.

        Raise it if trajectories are hitting the cap (check ``divergent`` and
        the step size); the cost is roughly the step above.
    dense_mass_matrix : bool
        Dense vs diagonal metric. Defaults **False** here, unlike the
        single-galaxy path: hierarchical D grows with the number of galaxies, and
        a dense matrix is O(D^2) in both memory and adaptation cost.
    target_accept_rate : float
        Dual-averaging target.
    memory_mode : {"low", "high"}
        Passed to :func:`build_flat_problem`.
    map_steps, map_learning_rate : int, float
        Gradient-ascent settings for the ``map`` driver.
    ghmc_alpha : float
        GHMC momentum persistence, in [0, 1] [dimensionless]. Same default as
        the single-galaxy ``run_ghmc``. The GHMC driver always uses a diagonal
        mass matrix (momentum-generator constraint), regardless of
        ``dense_mass_matrix``.
    ghmc_delta : float
        GHMC proposal step-size scaling [dimensionless]. Same default as the
        single-galaxy ``run_ghmc``.
    allow_unvalidated : bool
        Opt in to ``tier="broken"`` backends, exactly as ``Fitter.run`` does.
        Required for ``pathfinder`` and ``mcmc_ghmc``, the tier="broken" names
        this seam drives — reachable, but not safe by default.
    verbose : bool

    Returns
    -------
    PopulationPosterior

    Notes
    -----
    Every driver here samples the **same** posterior object built by
    :func:`build_flat_problem`, which is also what ray tracing uses. That is the
    point of the seam: two samplers disagreeing is then a sampler bug, not a
    difference in what was being fitted.

    ``dense_mass_matrix`` defaults to False and NUTS at large D still draws the
    shared high-dimension advisory — see
    :func:`tengri.inference._dimension_guard.warn_if_nuts_high_dim`.
    """
    import time

    from tengri.inference._backend_registry import check_usable, get_backend
    from tengri.inference._dimension_guard import warn_if_nuts_high_dim
    from tengri.inference.hierarchical import PopulationPosterior

    driver = FLAT_SAMPLERS.get(method)
    if driver is None:
        raise ValueError(
            f"method={method!r} is not reachable through the hierarchical flat "
            f"seam. Reachable: {', '.join(sorted(FLAT_SAMPLERS))}."
        )

    # Reachable is not the same as unguarded. The single-galaxy path refuses
    # tier="broken" backends unless the caller opts in, and opening the
    # hierarchical seam must not quietly become a way around that: measured on a
    # 2-galaxy, D=18 problem, `pathfinder` OOM-kills the process outright
    # (SIGKILL, exit 137), which is precisely what its tier records. So the same
    # gate applies here — the method stays available, the caller just has to say
    # out loud that they accept it.
    with contextlib.suppress(KeyError):  # a FLAT_SAMPLERS key with no registry entry
        check_usable(get_backend(method), allow_unvalidated=allow_unvalidated)

    prob = build_flat_problem(fitter, key=key, memory_mode=memory_mode, verbose=verbose)

    # D is only knowable once the problem is flattened, which is why the
    # hierarchical call site lives here rather than in PopulationFitter.run.
    warn_if_nuts_high_dim(method, prob.n_dim, surface="PopulationFitter.run")

    if verbose:
        print(
            f"Hierarchical {method}: {fitter.n_galaxies} galaxies, {prob.n_dim} total parameters"
        )

    key_run = jax.random.fold_in(key, 1)
    t0 = time.time()

    # The sampler receives `_ld2` (module-level, stable identity) plus the data
    # as a TRACED argument. Both halves are required for compile reuse, and the
    # obvious spelling defeats it twice over. Measured with jax.log_compiles:
    #
    #   fresh `lambda p,_: prob.log_prob(p)`, data in closure   1 compile EVERY fit
    #   stable module fn + data through data_args               0 compiles after the first
    #
    # `logdensity_fn_2arg` is a STATIC argument, so a new lambda object per call
    # is a new compile key even when the code is identical; and capturing the
    # data bakes it in as a constant, making each catalog its own program.
    # `data_args` must hold ARRAYS ONLY — it is traced. The callable goes in the
    # static slot, and is built once per problem rather than once per call.
    ld2 = prob.log_prob_with_data
    data_args = prob.data_args

    if driver in ("nuts", "nuts_pathfinder", "hmc", "dynamic_hmc", "ghmc"):
        from tengri.inference.backends.mcmc._shared import (
            _dynamic_hmc_full_scan,
            _ghmc_full_scan,
            _hmc_full_scan,
            _nuts_full_scan,
        )

        n_chain = n_burnin + n_samples
        wkey, ckey = jax.random.split(key_run)
        chain_keys = jax.random.split(ckey, n_chain)
        if driver == "hmc":
            positions, divergent, step_size, _imm = _hmc_full_scan(
                prob.init_flat,
                wkey,
                chain_keys,
                ld2,
                data_args,
                n_warmup,
                n_leapfrog,
                dense_mass_matrix,
                target_accept_rate,
            )
        elif driver == "dynamic_hmc":
            positions, divergent, step_size, _imm = _dynamic_hmc_full_scan(
                prob.init_flat,
                wkey,
                jax.random.fold_in(wkey, 1),
                chain_keys,
                ld2,
                data_args,
                n_warmup,
                dense_mass_matrix,
                target_accept_rate,
            )
        elif driver == "ghmc":
            positions, divergent, step_size, _imm = _ghmc_full_scan(
                prob.init_flat,
                wkey,
                jax.random.fold_in(wkey, 1),
                chain_keys,
                ld2,
                data_args,
                n_warmup,
                target_accept_rate,
                ghmc_alpha,
                ghmc_delta,
            )
        else:
            positions, divergent, step_size, _imm = _nuts_full_scan(
                prob.init_flat,
                wkey,
                chain_keys,
                ld2,
                data_args,
                n_warmup,
                max_num_doublings,
                dense_mass_matrix,
                target_accept_rate,
                driver == "nuts_pathfinder",
            )
        chain = positions[n_burnin:]
        extra = {
            "step_size": float(step_size),
            "divergent": int(jnp.sum(divergent[n_burnin:])),
            "n_warmup": n_warmup,
        }

    elif driver == "map":
        import optax

        opt = optax.adam(map_learning_rate)
        state = opt.init(prob.init_flat)

        def _step(carry, _):
            pos, st = carry
            g = jax.grad(prob.log_prob)(pos)
            updates, st = opt.update(jax.tree.map(lambda x: -x, g), st, pos)
            return (optax.apply_updates(pos, updates), st), None

        (best, _), _ = jax.lax.scan(_step, (prob.init_flat, state), None, length=map_steps)
        # A point estimate is a length-1 "chain", so downstream extraction is shared.
        chain = best[None, :]
        extra = {"n_steps": map_steps, "log_prob": float(prob.log_prob(best))}

    else:  # pragma: no cover - FLAT_SAMPLERS admits no other driver
        raise ValueError(f"no driver for {method!r}")

    wall_time = time.time() - t0
    shared_arr = jax.vmap(prob.extract_shared)(chain)
    shared_samples = {"psd_sigma": shared_arr[:, 0], "psd_tau_myr": shared_arr[:, 1]}
    shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

    if verbose:
        print(
            f"  Complete in {wall_time:.1f}s. "
            f"sigma_PSD = {shared_params['psd_sigma']:.2f}, "
            f"tau_PSD = {shared_params['psd_tau_myr']:.1f} Myr"
        )

    return PopulationPosterior(
        shared_samples=shared_samples,
        shared_params=shared_params,
        method=f"Hierarchical {method}",
        wall_time_s=wall_time,
        diagnostics={
            "n_galaxies": fitter.n_galaxies,
            "n_samples": int(chain.shape[0]),
            "D_total": prob.n_dim,
            "method": method,
            **extra,
        },
    )
