# SPDX-License-Identifier: BSD-3-Clause
"""The hierarchical model as a flat vector — one seam, every sampler.

``PopulationFitter`` used to accept 8 of the 20 registered backends. The other
12 raised ``ValueError``, not because hierarchical inference is incompatible
with them but because the only flat-vector formulation lived *inside*
``_run_raytrace`` as a set of closures, reachable by exactly one sampler.

This seam drives the NUTS / HMC / dynamic-HMC / GHMC / elliptical-slice /
MCLMC / adjusted-MCLMC / MAP / Laplace / pathfinder / nested-slice family.
Every other registered name is either driven by ``PopulationFitter``'s
own runners or refused with a stated reason — see :data:`FLAT_UNSUPPORTED`. A
name is driven here only when the driver runs the algorithm the name promises;
a stand-in (the first draft ran plain HMC under five distinct-algorithm names)
is silent substitution, not support. ``nss`` was the founding entry of the
refused set — the prior transform it needs is exact and was provided here from
the start; what was missing was a real nested sampler on top of it, and a
blind rejection stand-in returns biased samples rather than an approximation.
The in-tree Nested Slice Sampler now fills that gap (#1429), and the refused
set is empty.

This module lifts that formulation out. Once the problem is
``(init_flat, log_likelihood, log_prior, prior_transform)`` on an unconstrained
vector, every sampler in the registry is a plain function call.

The prior is the reason this is clean
-------------------------------------
The hierarchical parameterization is already **iid standard normal**. Every
free parameter is stored as an unconstrained latent and mapped through its
distribution's own ``unstandardize`` pushforward — the classes' single source
of truth, shared with ``sample`` and the single-galaxy unbounded machinery —
so an N(0,1) latent yields the DECLARED physical prior exactly: Uniform via
the Gaussian-CDF box map, and every other class via its quantile map (#1651;
the pushforward-vs-``log_prob`` agreement is pinned class-by-class in the
regression suite). The log posterior is therefore separable by construction:

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

from tengri.inference.likelihoods.gaussian import standardized_residual
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
#: Dynamic HMC, GHMC and elliptical slice have since gained their real
#: ``_shared.py`` full-scan drivers and rejoined; the MCLMC pair followed with
#: their blackjax ``(adjusted_)mclmc_find_L_and_step_size`` tuning. The rest
#: live in :data:`FLAT_UNSUPPORTED` until their real drivers are wired.
FLAT_SAMPLERS: dict[str, str] = {
    "mcmc_nuts": "nuts",
    "mcmc_hmc": "hmc",
    "mcmc_dynamic_hmc": "dynamic_hmc",
    "mcmc_ghmc": "ghmc",
    # The flat prior is exactly the iid N(0,1) the ESS ellipse assumes, so
    # the sampler's one structural requirement holds by construction here.
    "mcmc_ess": "ess",
    "mcmc_mclmc": "mclmc",
    "mcmc_adjusted_mclmc": "adjusted_mclmc",
    # Pinned to NUTS, unlike the single-galaxy auto-pick (NUTS for low-D,
    # raytrace above D~20): hierarchical D grows with the catalog, and at that
    # D raytrace degenerates and raises DegenerateChainError by design
    # (#1530). Auto-picking it would make the generic name a guaranteed
    # failure.
    "mcmc": "nuts",
    "map": "map",
    "laplace": "laplace",
    "pathfinder": "nuts_pathfinder",
    # The founding refusal, resolved by wiring rather than by lowering the
    # bar: the in-tree Nested Slice Sampler (Yallup+2026 — constrained HRSS
    # exploration WITHIN the likelihood contour, the same implementation the
    # single-galaxy backend runs) on the standardized problem, with live
    # points drawn from the exact iid N(0,1) prior. The blind-rejection
    # stand-in this replaces exhausted its attempt budget at iteration
    # 147/200 on a 2-galaxy D=18 problem and returned silently truncated —
    # therefore biased — samples (#1429).
    "nss": "nss",
}

#: Registered backends that this seam deliberately does NOT drive, and why.
#: Listed so ``PopulationFitter`` can raise a specific error instead of a generic
#: "unknown method", and so the reason survives longer than a commit message.
FLAT_UNSUPPORTED: dict[str, str] = {
    "hmc_is": (
        "importance-sampled evidence needs a proposal that covers the posterior; "
        "hierarchical D grows with the catalog, and a single Student-t fitted to "
        "the flat chain misses mass long before that (ESS collapses). Run "
        "mcmc_hmc for the hierarchical posterior and per-galaxy hmc_is/nss for "
        "evidences."
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
    #: ``(flat, data_args) -> scalar``. The LIKELIHOOD alone in the same
    #: data-as-argument form — for samplers that handle the prior themselves
    #: (elliptical slice encodes the exact N(0,1) prior in its ellipse; handing
    #: it ``log_prob_with_data`` would double-count the prior).
    log_likelihood_with_data: Callable | None = None


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
    physical = _physical_map(spec, free_names)
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

    # #1671 made operational, at the population surface: `model` above came
    # through the fit factory, which resolves the precompute LUT (#1641), so
    # price its forward bias against the whole catalog's SNR once. The exact
    # reference is rebuilt from the RAW factory at the SAME psd arguments —
    # pairing models that differ in anything but the LUT would measure
    # physics, not approximation. Advisory: any failure degrades to silence.
    _raw_factory = getattr(fitter, "_raw_model_factory", None)
    if _raw_factory is not None:
        from tengri.inference.fitter import _warn_if_lut_bias_amplified

        with contextlib.suppress(Exception):
            _raw_model = _raw_factory(psd_sigma=sigma_mid, psd_tau_myr=tau_mid)
            if model is not _raw_model:
                _warn_if_lut_bias_amplified(
                    _raw_model,
                    model,
                    all_data_arr,
                    all_noise_arr,
                    data_type,
                    surface="PopulationFitter",
                )

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
                # The declared prior's own N(0,1) pushforward — Uniform's is
                # bit-identical to the old to_bounded box map; every other
                # distribution becomes EXACT instead of silently Uniform
                # (#1651). Same convention as the per-galaxy MAP init and the
                # single-galaxy unbounded machinery.
                params[name] = physical[name](ub_scalars[name])
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

        chi2 = jnp.sum(standardized_residual(all_data, predictions.reshape(-1), all_noise) ** 2)
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
        log_likelihood_with_data=log_likelihood_with_data,
    )


def _physical_map(spec, free_names):
    """The per-parameter N(0,1) -> physical pushforwards for the seam.

    Each free parameter's map IS its distribution's own ``unstandardize`` —
    the classes' declared single source of truth, already used by ``sample``
    and by the single-galaxy unbounded machinery — so the seam's standardized
    space realizes the DECLARED prior exactly for every distribution
    (Uniform's pushforward is bit-identical to the old ``to_bounded`` box
    map; every other class becomes exact instead of being silently replaced
    by Uniform-over-bounds, the wrong-prior bug the hardening pass refused
    and #1651 specced away). The pushforward-vs-``log_prob`` agreement is
    pinned class-by-class in the regression suite.

    A distribution-like object without ``unstandardize`` cannot be
    standardized and is refused by name rather than mapped wrongly.
    """
    from tengri.parameters.priors import Uniform

    physical = {}
    for name in free_names:
        dist = spec.get_distribution(name)
        if isinstance(dist, Uniform):
            # Uniform keeps the EXACT previous graph, not merely the exact
            # math: Uniform.unstandardize is bitwise-equal to to_bounded
            # pointwise, but to_bounded is @jax.jit-decorated, so calling it
            # embeds a nested-jit boundary in the traced likelihood while the
            # bound method inlines — different XLA fusion, one-ULP erf-chain
            # differences, and a measurably different HMC chain (step_size
            # 0.06797 vs the historical 0.06732 on the reference fixture).
            # Routing Uniform through to_bounded keeps all-Uniform fits
            # bit-identical to every published result.
            lo, hi = dist.bounds
            physical[name] = lambda u, lo=lo, hi=hi: to_bounded(u, lo, hi)
            continue
        unstd = getattr(dist, "unstandardize", None)
        if not callable(unstd):
            raise NotImplementedError(
                f"hierarchical free parameter {name!r} declares a "
                f"{type(dist).__name__} prior with no 'unstandardize' "
                f"pushforward, so it cannot enter the seam's standardized "
                f"N(0,1) space. Implement unstandardize/standardize on the "
                f"distribution (see tengri.parameters.priors for the "
                f"contract), or fit this parameter per-galaxy through "
                f"Fitter."
            )
        physical[name] = unstd
    return physical


def _require_moving_chain(chain, method):
    """Refuse a retained chain whose draws are all identical.

    #1530's lesson generalized past its raytrace origin: MAP-echo draws look
    like a plausible answer. ``_require_finite_tuning`` catches the MCLMC
    starved-tuner *cause*; this is the effect-side net for every MCMC driver
    — whatever produced it, a chain that never moved is not a posterior.
    """
    import numpy as np

    if not bool(np.all(np.asarray(chain[1:]) == np.asarray(chain[0]))):
        return
    from tengri.inference.hierarchical import DegenerateChainError

    raise DegenerateChainError(
        f"{method}: every retained draw is identical to the first — the "
        f"chain never moved, so this is the initialization echoed "
        f"{int(chain.shape[0])} times, not a posterior. Check the sampler's "
        f"tuning diagnostics (step size, acceptance, warmup length) rather "
        f"than using these draws."
    )


def _adam_map_ascent(prob, map_steps, map_learning_rate):
    """Adam gradient ascent on ``prob.log_prob`` — shared by map and laplace.

    One implementation so the two drivers cannot drift: laplace IS this
    ascent plus a verified-mode covariance.
    """
    import optax

    opt = optax.adam(map_learning_rate)
    state = opt.init(prob.init_flat)

    def _step(carry, _):
        pos, st = carry
        g = jax.grad(prob.log_prob)(pos)
        updates, st = opt.update(jax.tree.map(lambda x: -x, g), st, pos)
        return (optax.apply_updates(pos, updates), st), None

    (best, _), _ = jax.lax.scan(_step, (prob.init_flat, state), None, length=map_steps)
    return best


def _newton_polish(prob, mode, *, tol, max_iters=12):
    """Damped-Newton ascent to a Laplace-grade mode, from Adam's endpoint.

    First-order alone cannot deliver a mode at hierarchical D: measured on
    the D=516 reference fixture, Adam's ``max |grad log_prob|`` was 1.13e3
    after 300 steps and still 84.6 after 8000. Newton converges
    quadratically inside the basin and reuses exactly the machinery the
    Laplace covariance needs anyway (one dense Hessian per iteration, ~D
    gradient-equivalents each).

    Levenberg-Marquardt damping: the step solves ``(H + lambda*I) dx = g``
    with ``lambda`` adapted — far from the mode the negative Hessian is
    typically indefinite (measured: a pure-Newton polish from Adam(300)'s
    endpoint stalled at |grad|=228 on the D=516 fixture), and a large
    ``lambda`` degrades the step gracefully toward plain gradient ascent,
    while near the mode ``lambda`` shrinks and Newton's quadratic finish
    takes over. A step is accepted only when it increases ``log_prob`` and
    stays finite; persistent failure exits, and the downstream guards then
    refuse the unconverged point rather than silently sampling it.
    """
    import numpy as np

    grad_fn = jax.grad(prob.log_prob)
    lam = 1e-3
    for _ in range(max_iters):
        g = grad_fn(mode)
        if float(np.max(np.abs(np.asarray(g)))) <= tol:
            break
        neg_hess = -jax.hessian(prob.log_prob)(mode)
        lp0 = float(prob.log_prob(mode))
        eye = jnp.eye(prob.n_dim if hasattr(prob, "n_dim") else g.shape[0])
        accepted = False
        for _ in range(12):
            step = jnp.linalg.solve(neg_hess + lam * eye, g)
            trial = mode + step
            lp = float(prob.log_prob(trial))
            if np.isfinite(lp) and lp > lp0 and bool(np.all(np.isfinite(np.asarray(trial)))):
                mode = trial
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
    return mode


def _require_converged_mode(grad_at_mode, method, map_steps, *, tol):
    """Refuse a Laplace expansion about a point that is not a mode (#1537).

    The single-galaxy laplace expanded about non-modes with no grad=0 check
    and returned plausible wrong answers — curvature measured off a mode is
    a covariance for the wrong distribution. The gradient's infinity norm at
    the reached point must be small; otherwise raise naming the knob.
    """
    import numpy as np

    gmax = float(np.max(np.abs(np.asarray(grad_at_mode))))
    if gmax <= tol:
        return
    raise RuntimeError(
        f"{method}: the point reached after map_steps={map_steps} is not a "
        f"mode — max |grad log_prob| = {gmax:.3g} exceeds tol={tol:.3g}, and "
        f"a Laplace covariance measured off a mode is a plausible wrong "
        f"answer (#1537). Raise map_steps (or adjust map_learning_rate) "
        f"until the ascent converges, or loosen laplace_grad_tol if this "
        f"tolerance is genuinely too strict for your problem's scale."
    )


def _require_psd_curvature(chol, method):
    """Refuse a non-negative-definite curvature at the expansion point.

    ``jnp.linalg.cholesky`` returns NaNs, not an exception, when the negative
    Hessian is not positive definite — which means the reached point is a
    saddle or worse, not a maximum. Sampling from those NaNs would return a
    posterior of NaNs (or garbage that passes a finite-check downstream);
    refuse by name instead.
    """
    import numpy as np

    if bool(np.all(np.isfinite(np.asarray(chol)))):
        return
    raise RuntimeError(
        f"{method}: the negative Hessian at the reached point is not "
        f"positive definite (its Cholesky factor carries non-finite "
        f"entries), so the point is a saddle rather than a mode and no "
        f"Gaussian covariance exists there. Raise map_steps so the ascent "
        f"reaches an actual maximum (#1537)."
    )


def _require_finite_tuning(L, step_size, method, n_warmup):
    """Refuse a non-finite (L, step size) tuning instead of sampling with it.

    A starved MCLMC tuner produces NaN parameters, and the chain then never
    moves — measured on the 2-galaxy D=516 fixture at ``n_warmup=60``:
    ``L=nan``, ``step_size=nan``, 60 post-tuning draws all equal to the init
    point. That output LOOKS like a plausible populated posterior, which is
    the #1530 failure mode; #1569 made raytrace's version of it a loud
    ``DegenerateChainError``, and this is the MCLMC family's version. The
    fraction-based tuning phases starve below a few hundred steps; the same
    fixture tunes finite at ``n_warmup=500``.
    """
    import numpy as np

    if np.isfinite(float(L)) and np.isfinite(float(step_size)):
        return
    from tengri.inference.hierarchical import DegenerateChainError

    raise DegenerateChainError(
        f"{method}: the (L, step size) tuner returned non-finite values "
        f"(L={float(L)!r}, step_size={float(step_size)!r}) after "
        f"n_warmup={n_warmup} tuning steps, and a chain run with them never "
        f"moves — every draw would be a copy of the initialization, which "
        f"looks like a posterior and is not one. The fraction-based tuning "
        f"phases starve at short warmup; raise n_warmup to a few hundred "
        f"(the single-galaxy default is 500; n_warmup=500 tunes this family "
        f"finite on the reference 2-galaxy problem)."
    )


def _require_nondegenerate_live_set(n_live, n_dim, method):
    """Refuse an NSS live set that cannot span the parameter space.

    HRSS draws its slice directions from the live points' empirical
    covariance, and ``n_live`` points give that matrix rank at most
    ``n_live - 1``. With ``n_live <= D`` every direction lies in a proper
    subspace and the orthogonal complement is NEVER explored — the returned
    samples are confined to a hyperplane while passing every finite-check
    downstream. That is silent bias, not slowness, so it is refused by name
    rather than priced as a warning.
    """
    if n_live > n_dim:
        return
    raise ValueError(
        f"{method}: nss_n_live={n_live} live points cannot span a D={n_dim} "
        f"parameter space — the HRSS direction covariance has rank at most "
        f"n_live-1, so slice directions never leave a proper subspace and "
        f"the samples are silently biased. Raise nss_n_live above D (cost "
        f"grows with both), or use mcmc_nuts / mcmc_hmc, which need no live "
        f"set. Stochastic-field hierarchies put every field latent into D "
        f"(n_grid per galaxy); a coarser n_grid at build time shrinks D."
    )


def _require_converged_evidence(n_iter, max_iterations, remaining, tol, method):
    """Refuse a nested run that hit its iteration cap before the evidence converged.

    Terminating while ``log(Z_live/Z)`` still exceeds tolerance means the
    live set still holds unintegrated posterior mass; resampling then returns
    a silently truncated — therefore biased — sample set, which is precisely
    the failure that kept this name refused (the blind-rejection stand-in
    terminated at iteration 147 of a requested 200 and handed back a
    plausible-looking answer, #1429). Same state, different road, same loud
    refusal.
    """
    if remaining < tol:
        return
    raise RuntimeError(
        f"{method}: nested sampling hit nss_max_iterations={max_iterations} "
        f"(n_iter={n_iter}) while the unintegrated evidence fraction "
        f"log(Z_live/Z) = {remaining:.2f} still exceeds "
        f"nss_log_evidence_tol={tol}. The live set still holds posterior "
        f"mass, so resampling now would return the silently truncated, "
        f"biased sample set that kept this name refused (#1429). Raise "
        f"nss_max_iterations, or raise nss_num_delete to retire more points "
        f"per iteration."
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
    mclmc_target_accept_rate=0.65,
    laplace_grad_tol=1e-2,
    nss_n_live=500,
    nss_num_delete=50,
    nss_num_inner_steps=None,
    nss_log_evidence_tol=-3.0,
    nss_max_iterations=10000,
    allow_unvalidated=False,
    verbose=True,
    **unknown,
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
        The ``ess`` driver has no warmup — its exact-prior ellipse needs no
        tuning, so ``n_warmup`` and the HMC-family knobs below are ignored
        there; only ``n_burnin`` / ``n_samples`` apply. The MCLMC family is
        the mirror image: ``n_warmup`` sets the (L, step size) tuning length,
        which consumes the transient, so ``n_burnin`` is ignored there. The
        ``nss`` driver ignores both — nested sampling has no warmup or
        burn-in; ``n_samples`` is the number of equal-weight posterior draws
        resampled from the dead-point history.
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
        Gradient-ascent settings for the ``map`` and ``laplace`` drivers
        (one shared ascent — laplace IS map plus a verified-mode covariance).
    laplace_grad_tol : float
        Convergence tolerance on ``max |grad log_prob|`` at the point the
        ``laplace`` ascent reaches [dimensionless in the standardized latent
        space]. Curvature measured off a mode is a covariance for the wrong
        distribution (#1537), so exceeding this raises rather than returning
        plausible wrong error bars.
    ghmc_alpha : float
        GHMC momentum persistence, in [0, 1] [dimensionless]. Same default as
        the single-galaxy ``run_ghmc``. The GHMC driver always uses a diagonal
        mass matrix (momentum-generator constraint), regardless of
        ``dense_mass_matrix``.
    ghmc_delta : float
        GHMC proposal step-size scaling [dimensionless]. Same default as the
        single-galaxy ``run_ghmc``.
    mclmc_target_accept_rate : float
        Metropolis acceptance target for the ``adjusted_mclmc`` driver's
        tuner [dimensionless]. Same default (0.65) as the single-galaxy
        ``run_adjusted_mclmc`` — deliberately NOT the HMC-family
        ``target_accept_rate``, whose 0.8 default tunes a different
        proposal mechanism. Unused by the unadjusted ``mclmc`` driver,
        which has no accept/reject step.
    nss_n_live : int
        Live points for the ``nss`` driver. Must exceed the problem's D —
        the HRSS direction covariance is estimated from the live set and is
        singular otherwise (refused loudly). The default (500, matching the
        single-galaxy ``run_nss``) therefore refuses stochastic-field
        hierarchies at their usual D by construction; those want
        ``mcmc_nuts`` anyway.
    nss_num_delete : int
        Live points retired and replaced per NS iteration (``nss`` driver).
    nss_num_inner_steps : int or None
        HRSS walk length per replacement (``nss`` driver). ``None`` (the
        default) uses D, matching the single-galaxy ``run_nss``.
    nss_log_evidence_tol : float
        Terminate when ``log(Z_live / Z_accumulated)`` falls below this
        (``nss`` driver) [dimensionless]. Hitting ``nss_max_iterations``
        first raises — the live set still holds posterior mass, and
        resampling then is the silently-truncated bias of #1429.
    nss_max_iterations : int
        Safety cap on NS iterations (``nss`` driver).
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
    ``tengri.inference._dimension_guard.warn_if_nuts_high_dim``.
    """
    import inspect
    import time

    from tengri.inference._backend_registry import check_usable, get_backend
    from tengri.inference._dimension_guard import warn_if_nuts_high_dim
    from tengri.inference.hierarchical import PopulationPosterior

    if unknown:
        # The previous spelling was ``**_ignored`` — a silent kwarg sink. A
        # typo'd fit option (or ``init_from``, which the hierarchical surface
        # documents as unsupported) vanished while the fit ran with defaults
        # and the caller believed otherwise (#1378's bug class).
        accepted = sorted(
            p
            for p in inspect.signature(run_flat_sampler).parameters
            if p not in ("fitter", "method", "unknown")
        )
        raise TypeError(
            f"run_flat_sampler() got unexpected keyword argument(s) "
            f"{sorted(unknown)}. Accepted fit options for the hierarchical "
            f"flat seam: {accepted}. Note that some options are per-family "
            f"(the ess driver ignores warmup knobs; the MCLMC family ignores "
            f"n_burnin) and init_from is not supported hierarchically — "
            f"per-galaxy initialization is automatic via MAP."
        )

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

    elif driver == "ess":
        from tengri.inference.backends.mcmc._shared import _ess_full_scan

        n_chain = n_burnin + n_samples
        chain_keys = jax.random.split(key_run, n_chain)
        positions, subiters = _ess_full_scan(
            prob.init_flat,
            chain_keys,
            prob.log_likelihood_with_data,
            data_args,
        )
        chain = positions[n_burnin:]
        # No step size, no divergences: the exact-prior ellipse accepts by
        # construction. Subiterations are ESS's only knob-free diagnostic.
        extra = {
            "mean_subiter": float(jnp.mean(subiters[n_burnin:])),
            "n_burnin": n_burnin,
        }

    elif driver in ("mclmc", "adjusted_mclmc"):
        import blackjax

        from tengri.inference.backends.mcmc._shared import (
            _adjusted_mclmc_sample_scan,
            _mclmc_sample_scan,
        )

        # blackjax's (adjusted_)mclmc_find_L_and_step_size takes a ONE-argument
        # logdensity, so the data must close over here — one compile per
        # catalog for this family, exactly as the single-galaxy backends
        # behave. The (ld2, data_args) reuse contract is not reachable through
        # the tuner API without reimplementing the tuner.
        def _ld_1arg(pos):
            return ld2(pos, data_args)

        # No burn-in phase: the (L, step size) tuning consumes the transient,
        # so the chain is n_samples long and n_burnin is ignored.
        tune_key, init_key, ckey = jax.random.split(key_run, 3)
        chain_keys = jax.random.split(ckey, n_samples)
        if driver == "mclmc":
            kernel = blackjax.mcmc.mclmc.build_kernel(
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            )
            state = blackjax.mcmc.mclmc.init(prob.init_flat, _ld_1arg, init_key)
            state, params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                logdensity_fn=_ld_1arg,
                num_steps=n_warmup,
                state=state,
                rng_key=tune_key,
                diagonal_preconditioning=True,
            )
            _require_finite_tuning(params.L, params.step_size, method, n_warmup)
            _, chain = _mclmc_sample_scan(
                state,
                chain_keys,
                kernel,
                params.L,
                params.step_size,
                _ld_1arg,
                params.inverse_mass_matrix,
            )
            extra = {
                "L": float(params.L),
                "step_size": float(params.step_size),
                "n_warmup": n_warmup,
            }
        else:
            kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            )
            state = blackjax.mcmc.adjusted_mclmc.init(prob.init_flat, _ld_1arg)
            state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                logdensity_fn=_ld_1arg,
                num_steps=n_warmup,
                state=state,
                rng_key=tune_key,
                target=mclmc_target_accept_rate,
                diagonal_preconditioning=True,
            )
            _require_finite_tuning(params.L, params.step_size, method, n_warmup)
            n_integration_steps = jnp.ceil(params.L / params.step_size).astype(int)
            _, (chain, divergent) = _adjusted_mclmc_sample_scan(
                state,
                chain_keys,
                kernel,
                params.step_size,
                n_integration_steps,
                _ld_1arg,
                params.inverse_mass_matrix,
            )
            extra = {
                "L": float(params.L),
                "step_size": float(params.step_size),
                "n_integration_steps": int(n_integration_steps),
                "divergent": int(jnp.sum(divergent)),
                "n_warmup": n_warmup,
            }

    elif driver == "nss":
        # The real Nested Slice Sampler (Yallup+2026) on the standardized
        # problem — the sampler whose absence kept this name refused (#1429).
        # The prior needs no machinery at all here: live points are DRAWN
        # from the exact iid N(0,1) prior, and the HRSS walk explores within
        # the rising likelihood contour, so the sampler receives the
        # LIKELIHOOD alone (handing it log_prob would double-count the prior,
        # exactly as for ESS). Same loop shape as the single-galaxy
        # ``run_nss``; same memory-motivated max_shrinkage=20 (100 compiled a
        # 20+ GB vmap(while_loop) graph there).
        from tengri.inference.backends.nested.nss import as_top_level_api
        from tengri.inference.backends.nested.utils import (
            ess as ns_ess,
            finalize as ns_finalize,
            sample as ns_sample,
        )

        _require_nondegenerate_live_set(nss_n_live, prob.n_dim, method)
        num_inner = prob.n_dim if nss_num_inner_steps is None else nss_num_inner_steps

        # NSS's native compile-once mode: a 2-arg likelihood plus data_args
        # carried in the sampler state, so the data stays a traced value —
        # the seam's (ld2, data_args) contract through the sampler's own
        # mechanism rather than a closure.
        algo = as_top_level_api(
            prob.log_prior,
            prob.log_likelihood_with_data,
            num_inner,
            num_delete=nss_num_delete,
            max_steps=10,
            max_shrinkage=20,
            data_args=data_args,
        )
        init_key, loop_key = jax.random.split(key_run)
        live = algo.init(
            jax.random.normal(init_key, (nss_n_live, prob.n_dim)), data_args=data_args
        )
        step = jax.jit(algo.step)

        dead = []
        n_iter = 0
        while True:
            loop_key, sk = jax.random.split(loop_key)
            live, dead_info = step(sk, live)
            # Keep the dead particles, drop update_info immediately — it is
            # the replacement step's MCMC internals, 3-4x larger than the
            # particles and unused by sample/ess (same choice as run_nss).
            dead.append(dead_info._replace(update_info=None))
            n_iter += 1
            remaining = float(live.integrator.logZ_live - live.integrator.logZ)
            if verbose and n_iter % 10 == 0:
                log_z_est = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
                print(f"  NSS iter {n_iter}: log Z = {log_z_est:.2f}, remaining = {remaining:.2f}")
            if remaining < nss_log_evidence_tol or n_iter >= nss_max_iterations:
                break

        _require_converged_evidence(
            n_iter, nss_max_iterations, remaining, nss_log_evidence_tol, method
        )
        log_z = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
        ns_run = ns_finalize(live, dead, update_info=False)
        sample_key, ess_key = jax.random.split(jax.random.fold_in(key_run, 2))
        chain = ns_sample(sample_key, ns_run, n_samples).position
        extra = {
            "log_evidence": log_z,
            "ess": float(ns_ess(ess_key, ns_run)),
            "n_live": nss_n_live,
            "num_delete": nss_num_delete,
            "num_inner_steps": num_inner,
            "n_iterations": n_iter,
            "n_dead": n_iter * nss_num_delete,
        }

    elif driver == "map":
        best = _adam_map_ascent(prob, map_steps, map_learning_rate)
        # A point estimate is a length-1 "chain", so downstream extraction is shared.
        chain = best[None, :]
        extra = {"n_steps": map_steps, "log_prob": float(prob.log_prob(best))}

    elif driver == "laplace":
        # MAP + a Gaussian covariance from the curvature AT A VERIFIED MODE —
        # the two things whose absence kept this name refused: returning the
        # bare point estimate under 'laplace' silently dropped the error bars
        # (the map driver exists for that), and curvature off a mode is a
        # plausible wrong answer (#1537: the single-galaxy laplace expanded
        # about non-modes with no grad=0 check).
        mode = _adam_map_ascent(prob, map_steps, map_learning_rate)
        # Adam alone plateaus at hierarchical D (measured: |grad| 84.6 after
        # 8000 steps on the D=516 fixture); the Newton polish reaches the
        # actual mode with the same Hessian machinery the covariance needs.
        mode = _newton_polish(prob, mode, tol=laplace_grad_tol, max_iters=40)
        grad_at_mode = jax.grad(prob.log_prob)(mode)
        _require_converged_mode(grad_at_mode, method, map_steps, tol=laplace_grad_tol)

        # Dense negative Hessian: D forward-over-reverse passes; at the
        # reference D=516 this is ~D gradient-equivalents, comparable to one
        # more MAP ascent. cov = (-H)^{-1} sampled via -H = L L^T,
        # draw = mode + solve(L^T, eps) so cov(draw) = (L L^T)^{-1} exactly.
        neg_hess = -jax.hessian(prob.log_prob)(mode)
        chol = jnp.linalg.cholesky(neg_hess)
        _require_psd_curvature(chol, method)
        eps = jax.random.normal(key_run, (n_samples, prob.n_dim))
        chain = mode[None, :] + jax.scipy.linalg.solve_triangular(chol.T, eps.T, lower=False).T
        extra = {
            "n_steps": map_steps,
            "log_prob": float(prob.log_prob(mode)),
            "grad_inf_norm": float(jnp.max(jnp.abs(grad_at_mode))),
            "n_samples": n_samples,
        }

    else:  # pragma: no cover - FLAT_SAMPLERS admits no other driver
        raise ValueError(f"no driver for {method!r}")

    if driver != "map":
        # Effect-side net for every MCMC driver: whatever produced it, a chain
        # that never moved is the initialization echoed n_samples times, not a
        # posterior (#1530's failure mode, generalized past raytrace).
        _require_moving_chain(chain, method)

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
