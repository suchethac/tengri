# SPDX-License-Identifier: BSD-3-Clause
"""The hierarchical model as a flat vector, one seam, every sampler.

``PopulationFitter`` used to accept 8 of the 20 registered backends. The other
12 raised ``ValueError``, not because hierarchical inference is incompatible
with them but because the only flat-vector formulation lived *inside*
``_run_raytrace`` as a set of closures, reachable by exactly one sampler.

This seam drives every sampler of the shared driver (:mod:`tengri.inference.sampling`)
on the flat problem built here: the chain families, ChEES, tempered SMC, low-rank HMC,
elliptical slice, MCLMC, importance-sampled HMC, nested slice sampling, MAP, Laplace and
the Gaussian variational families. The method set, :data:`FLAT_SAMPLERS`, is derived from
the sampler registry, and there is no refusal table: a cost concern is a warning with the
computed footprint. A name is driven only when its sampler runs the algorithm the name
promises; a stand-in (the first draft ran plain HMC under five distinct-algorithm names)
is silent substitution, not support. Names not in the set (the NIFTy VI methods, ray
tracing) are driven by ``PopulationFitter``'s own runners.

This module lifts that formulation out. Once the problem is
``(init_flat, log_likelihood, log_prior, prior_transform)`` on an unconstrained
vector, every sampler in the registry is a plain function call.

The prior is the reason this is clean
-------------------------------------
The hierarchical parameterization is already **iid standard normal**. Every
free parameter is stored as an unconstrained latent and mapped through its
distribution's own ``unstandardize`` pushforward, the classes' single source
of truth, shared with ``sample`` and the single-galaxy unbounded machinery,
so an N(0,1) latent yields the DECLARED physical prior exactly: Uniform via
the Gaussian-CDF box map, and every other class via its quantile map (#1651;
the pushforward-vs-``log_prob`` agreement is pinned class-by-class in the
regression suite). The log posterior is therefore separable by construction:

.. math::

    \\log p(u \\mid d) = \\underbrace{-\\tfrac{1}{2}\\chi^2(u)}_{\\log L}
                        \\underbrace{-\\tfrac{1}{2}\\lVert u \\rVert^2}_{\\log \\pi}

Two consequences that do the work here:

* Gradient samplers (NUTS, HMC, MCLMC, …) take ``log_prob = log_L + log_pi``
  directly, no bijector stack, no Jacobian bookkeeping.
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

from tengri.inference.likelihoods.gaussian import inv_noise_std
from tengri.inference.sampling.driver import method_map, run_sampler
from tengri.inference.sampling.guards import (  # noqa: F401  (re-exported guard names)
    _require_converged_mode,
    _require_finite_tuning,
    _require_moving_chain,
    _require_nondegenerate_live_set,
    _require_psd_curvature,
)
from tengri.inference.sampling.target import FlatTarget, LatentLayout
from tengri.utils.transforms import to_bounded, to_unbounded

__all__ = ["FLAT_SAMPLERS", "FlatProblem", "build_flat_problem", "run_flat_sampler"]

#: Registry methods reachable through the flat seam, mapped to the shared driver's sampler.
#:
#: Derived from the sampler registry of :mod:`tengri.inference.sampling`, never written by
#: hand: ``PopulationFitter`` builds its supported-method set from this mapping, so
#: registering a sampler there is the single edit that makes it available hierarchically.
#: A name appears only when its sampler runs the algorithm the name promises (a stand-in
#: reported under another name is silent substitution, the seam's founding bug). Every
#: registered backend with a driver is reachable; a cost concern (memory of a dense metric,
#: an ensemble, a full-rank factor) is a warning with the computed footprint, not a refusal.
FLAT_SAMPLERS: dict[str, str] = method_map()


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
        D, shared hyperparameters plus every galaxy's free parameters (plus the
        stochastic field latents when the SFH is stochastic).
    log_likelihood : callable
        ``ndarray (D,) -> scalar``. Gaussian data term only, no prior.
    log_prior : callable
        ``ndarray (D,) -> scalar``. Unnormalized iid standard normal.
    log_prob : callable
        ``ndarray (D,) -> scalar``. Their sum, the log posterior.
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
    #: ``{"per_galaxy": {...}, "shared": {...}}``: the observed arrays, one row per galaxy
    #: (``data``, ``noise``, ``sqrt_noise_inv``, and ``presence`` / ``data_mask`` when a record
    #: carries them), and the catalog-wide term arrays. Arrays only, kept OUT of the closure.
    data_args: Any = ()
    #: ``(flat, data_args) -> scalar``. The form samplers must use: with the data
    #: supplied as a traced argument, one compiled program serves every catalog.
    log_prob_with_data: Callable | None = None
    #: ``(flat, data_args) -> scalar``. The LIKELIHOOD alone in the same
    #: data-as-argument form, for samplers that handle the prior themselves
    #: (elliptical slice encodes the exact N(0,1) prior in its ellipse; handing
    #: it ``log_prob_with_data`` would double-count the prior).
    log_likelihood_with_data: Callable | None = None
    #: ``sfh={'field_centering': a}`` of the fit [dimensionless]. Below 1 the
    #: GP-field latents carry a partially centered prior that ``prior_transform``
    #: and the nested sampler's live points do not realize (#3161).
    field_centering: float = 1.0
    #: How ``init_flat`` is assembled from the population-shared, per-galaxy and
    #: stochastic-SFH latent blocks.
    layout: LatentLayout | None = None


#: Optional per-band record keys and the value an absent key takes. An absent key means every
#: band is detected: ``presence`` 1 and ``data_mask`` 0, as ``Fitter`` reads them.
_RECORD_NEUTRAL = {"presence": 1.0, "data_mask": 0}
#: The values each optional record key may take (``presence`` exact 0/1, ``data_mask`` trinary).
_RECORD_ALLOWED = {"presence": (0.0, 1.0), "data_mask": (-1, 0, 1)}


def _stack_record_key(galaxies, key):
    """Stack one optional record key over the galaxies, neutral where a galaxy omits it."""
    from tengri.config.exceptions import ConfigError

    rows = []
    for gal in galaxies:
        shape = jnp.shape(gal["flux_obs"])
        value = jnp.asarray(gal[key]) if key in gal else jnp.full(shape, _RECORD_NEUTRAL[key])
        if value.shape != shape:
            raise ConfigError(
                f"record key {key!r} has shape {value.shape}, flux_obs has shape {shape}; "
                "the key is one value per band."
            )
        if not bool(jnp.all(jnp.isin(value, jnp.asarray(_RECORD_ALLOWED[key])))):
            raise ConfigError(f"record key {key!r} takes values in {_RECORD_ALLOWED[key]} only.")
        rows.append(value)
    return jnp.stack(rows)


def _per_galaxy_channels(galaxies):
    """Stack the galaxy records into ``(N, n)`` arrays, including the optional record keys.

    Bands whose ``presence`` is 0 get the placeholder values of ``Fitter``, so a non-finite
    flux there cannot poison the sum.
    """
    from tengri.inference.fitter import _neutralize_absent

    data = [jnp.asarray(gal["flux_obs"]) for gal in galaxies]
    noise = [jnp.asarray(gal["noise"]) for gal in galaxies]
    channels = {
        key: _stack_record_key(galaxies, key)
        for key in _RECORD_NEUTRAL
        if any(key in gal for gal in galaxies)
    }
    if "presence" in channels:
        cleaned = [
            _neutralize_absent(d, n, p)
            for d, n, p in zip(data, noise, channels["presence"], strict=True)
        ]
        data = [d for d, _ in cleaned]
        noise = [n for _, n in cleaned]
    stacked = {"data": jnp.stack(data), "noise": jnp.stack(noise)}
    stacked["sqrt_noise_inv"] = jax.vmap(inv_noise_std)(stacked["noise"])
    return {**stacked, **channels}


def build_flat_problem(fitter, *, key, memory_mode="low", verbose=False, map_steps=80):
    """Flatten a :class:`PopulationFitter` into a sampler-agnostic problem.

    Parameters
    ----------
    fitter : PopulationFitter
        The hierarchical fit to flatten. Read-only, nothing is mutated.
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
    ``log_prob`` inline, ~135 lines textually equivalent to this function but
    structurally independent, so nothing prevented the two from drifting into
    sampling different distributions while every docstring claimed otherwise.
    It calls this builder instead, verified bit-for-bit: raytrace on a fixed key
    returned ``sigma=1.667, tau=154.89`` both before and after the change.

    So "every sampler targets the same posterior" is a structural property here,
    not a claim requiring anyone's diligence.
    """
    from tengri import Fitter
    from tengri.components.stellar.sfh.young_fraction import young_fraction_prior_for_spec
    from tengri.config.exceptions import ConfigError
    from tengri.inference.backends.map_dispatch import build_vectorized_map_solver

    n_gal = fitter.n_galaxies
    spec = fitter._spec
    if young_fraction_prior_for_spec(spec) is not None:
        raise ConfigError(
            "the young-fraction prior term is not evaluated on the hierarchical flat seam "
            "(log_prior there is the iid N(0, 1) of every latent); the Student-t ratio prior "
            "would be dropped. Use coordinates='ratios' there."
        )
    stochastic = spec.stochastic
    field_centering = float(getattr(spec, "field_centering", 1.0))
    from tengri.inference.loss_functions import standardized_neg_log_prior

    n_grid = spec.n_grid
    free_names = fitter._free_names
    physical = _physical_map(spec, free_names)
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

    per_galaxy = _per_galaxy_channels(fitter.galaxies)
    _template_channels = {k: v[0] for k, v in per_galaxy.items() if k in _RECORD_NEUTRAL}
    _template_fitter = Fitter(
        model,
        per_galaxy["data"][0],
        per_galaxy["noise"][0],
        data_type=data_type,
        **_template_channels,
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

    layout = LatentLayout.hierarchical(
        tuple(free_names),
        n_gal,
        n_grid if stochastic else 0,
        sigma_bounds=(sigma_lo, sigma_hi),
        tau_bounds=(tau_lo, tau_hi),
    )
    init = {
        "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
        "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
        "gal": {name: all_init_unbounded.get(name, jnp.zeros(n_gal)) for name in free_names},
    }
    if stochastic:
        init["gal_xi"] = all_init_unbounded.get("psd_xi", jnp.zeros((n_gal, n_grid)))

    init_flat = layout.ravel(init)
    unravel_fn = layout.unravel
    n_dim = int(init_flat.shape[0])

    all_data_arr = jnp.concatenate([jnp.asarray(g["flux_obs"]) for g in fitter.galaxies])
    all_noise_arr = jnp.concatenate([jnp.asarray(g["noise"]) for g in fitter.galaxies])

    # #1671 made operational, at the population surface: `model` above came
    # through the fit factory, which resolves the precompute LUT (#1641), so
    # price its forward bias against the whole catalog's SNR once. The exact
    # reference is rebuilt from the RAW factory at the SAME psd arguments,
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

    from tengri.config.exceptions import ConfigError, ParameterError
    from tengri.inference.data_term import DataTermOptions, build_data_args, build_data_term
    from tengri.observation.noise import has_noise_model

    mask_stack = per_galaxy.get("data_mask")
    if mask_stack is not None and has_noise_model(spec) and bool(jnp.any(mask_stack != 0)):
        raise ParameterError(
            "PopulationFitter(data_mask=...) flags upper or lower limits, and the hierarchical "
            "flat seam cannot score them with a free noise model (noise_frac_cal / noise_dof): "
            "the band's sigma would depend on the parameters while the censored term assumes a "
            "fixed one. Fix the noise parameters, or drop the limits."
        )

    presence_row = per_galaxy["presence"][0] if "presence" in per_galaxy else None
    mask_row = mask_stack[0] if mask_stack is not None else None
    data_args0 = build_data_args(
        model,
        per_galaxy["data"][0],
        per_galaxy["noise"][0],
        data_mask=mask_row,
        presence=presence_row,
    )
    if stochastic and "_jit_inputs" not in data_args0:
        raise ConfigError(
            "the hierarchical flat seam needs the threaded forward (_jit_inputs) for a stochastic "
            "SFH: without it the eager prediction path filters to model.spec.free_params and "
            "drops sfh_field_xi, so every per-galaxy latent gets a zero gradient (#2296 round 3). "
            "Refused by name rather than fitted silently."
        )

    likelihood = _template_fitter._maybe_build_default_likelihood()
    term = build_data_term(
        model,
        data_args=data_args0,
        options=DataTermOptions(data_type),
        likelihood=likelihood,
    )
    shared_args = {k: v for k, v in data_args0.items() if k not in per_galaxy}
    _data_args = {"per_galaxy": per_galaxy, "shared": shared_args}

    def log_likelihood_with_data(flat_params, data_args):
        """Gaussian data term, log p(d | theta), with the data supplied as an ARGUMENT.

        ``data_args`` holds the per-galaxy rows (``per_galaxy``) and the catalog-wide
        arrays (``shared``). Each galaxy's row is scored by the one data term, mapped with
        ``lax.map`` under ``jax.checkpoint``. Taking the arrays as traced arguments lets one
        compiled program serve different catalogs.
        """
        p = unravel_fn(flat_params)
        psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
        psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

        def forward_one(ub_scalars, xi, row):
            params = {}
            for name in free_names:
                # The declared prior's own N(0,1) pushforward, Uniform's is
                # bit-identical to the old to_bounded box map; every other
                # distribution becomes EXACT instead of silently Uniform
                # (#1651). Same convention as the per-galaxy MAP init and the
                # single-galaxy unbounded machinery.
                params[name] = physical[name](ub_scalars[name])
            # The two PSD names are declared FREE on ``model.spec`` (the
            # population-shared parameters this fit varies), so setting them to
            # this step's shared value is a plain free-key write. The threaded
            # forward self-merges every other Fixed value (#2296), and
            # ``sfh_field_xi`` is a runtime latent that is neither free nor Fixed
            # on the spec, so it is passed through unfiltered.
            params["sfh_field_psd_sigma"] = psd_sigma
            params["sfh_field_psd_tau_myr"] = psd_tau
            if stochastic:
                params["sfh_field_xi"] = xi
            params = spec.resolve_mirrors(params)
            return term.log_prob(params, {**data_args["shared"], **row})

        fwd = jax.checkpoint(forward_one) if memory_mode == "low" else forward_one
        rows = data_args["per_galaxy"]
        if stochastic:
            loglik = jax.lax.map(
                lambda args: fwd(args[0], args[1], args[2]), (p["gal"], p["gal_xi"], rows)
            )
        else:
            loglik = jax.lax.map(lambda args: fwd(args[0], None, args[1]), (p["gal"], rows))
        return jnp.sum(loglik)

    def log_likelihood(flat_params):
        """Convenience alias with this fit's data bound. NOT for samplers."""
        return log_likelihood_with_data(flat_params, _data_args)

    def log_prior(flat_params):
        """Unnormalized log prior of the latents (#3161).

        The population PSD latents are iid N(0, 1). At ``field_centering = a < 1``
        the per-galaxy GP-field latents take the partially centered prior, through
        the same ``standardized_neg_log_prior`` the standardized objective calls.
        The field latents are flattened so the normalizer counts every element.
        """
        p = unravel_fn(flat_params)
        psd_penalty = p["psd_sigma_u"] ** 2 + p["psd_tau_u"] ** 2
        if field_centering == 1.0:
            penalty = psd_penalty
            for name in free_names:
                penalty += jnp.sum(p["gal"][name] ** 2)
            if stochastic:
                penalty += jnp.sum(p["gal_xi"] ** 2)
            return -0.5 * penalty
        latents = dict(p["gal"])
        if stochastic:
            latents["psd_xi"] = jnp.ravel(p["gal_xi"])
        neg_log_prior = standardized_neg_log_prior(
            latents,
            free_names,
            stochastic=stochastic,
            centering=field_centering,
            psd_sigma_dex=to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi),
        )
        return -0.5 * psd_penalty - neg_log_prior

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
        field_centering=field_centering,
        layout=layout,
    )


def _physical_map(spec, free_names):
    """The per-parameter N(0,1) -> physical pushforwards for the seam.

    Each free parameter's map IS its distribution's own ``unstandardize``,
    the classes' declared single source of truth, already used by ``sample``
    and by the single-galaxy unbounded machinery, so the seam's standardized
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
            # bound method inlines, different XLA fusion, one-ULP erf-chain
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
    target_accept_rate=None,
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
    n_chains=1,
    metric="diag",
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
        Canonical registry name; must be a key of :data:`FLAT_SAMPLERS`, which the sampler
        registry of :mod:`tengri.inference.sampling` defines.
    key : PRNGKey
    n_warmup, n_burnin, n_samples : int
        Window adaptation / discarded / retained chain lengths (MCMC drivers).
        The ``ess`` driver has no warmup, its exact-prior ellipse needs no
        tuning, so ``n_warmup`` and the HMC-family knobs below are ignored
        there; only ``n_burnin`` / ``n_samples`` apply. The MCLMC family is
        the mirror image: ``n_warmup`` sets the (L, step size) tuning length,
        which consumes the transient, so ``n_burnin`` is ignored there. The
        ``nss`` driver ignores both, nested sampling has no warmup or
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
        7 and 3.96 GB at 10, a ~1.4 GB step between 5 and 7 that then
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
        (one shared ascent, laplace IS map plus a verified-mode covariance).
    laplace_grad_tol : float
        Convergence tolerance on ``max |grad log_prob|`` at the point the
        ``laplace`` ascent reaches [dimensionless in the standardized latent
        space]. Curvature measured off a mode is a covariance for the wrong
        distribution (#1537), so exceeding this raises rather than returning
        plausible wrong error bars.
    ghmc_alpha : float
        GHMC momentum persistence, in [0, 1] [dimensionless]. Hand-set here, and
        no longer shared with the single-galaxy ``run_ghmc``: that path adapts
        both this and ``ghmc_delta`` from a MEADS ensemble and defaults them to
        ``None`` (see ``bench/reports/2026-08-30_ghmc_meads_adaptation.md``).
        This driver still window-adapts, so it still needs a number. The GHMC
        driver always uses a diagonal mass matrix (momentum-generator
        constraint), regardless of ``dense_mass_matrix``.
    ghmc_delta : float
        GHMC proposal step-size scaling [dimensionless]. Hand-set here; see
        ``ghmc_alpha``.
    mclmc_target_accept_rate : float
        Metropolis acceptance target for the ``adjusted_mclmc`` driver's
        tuner [dimensionless]. Same default (0.65) as the single-galaxy
        ``run_adjusted_mclmc``, deliberately NOT the HMC-family
        ``target_accept_rate``, whose 0.8 default tunes a different
        proposal mechanism. Unused by the unadjusted ``mclmc`` driver,
        which has no accept/reject step.
    nss_n_live : int
        Live points for the ``nss`` driver. Must exceed the problem's D,
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
        first raises, the live set still holds posterior mass, and
        resampling then is the silently-truncated bias of #1429.
    nss_max_iterations : int
        Safety cap on NS iterations (``nss`` driver).
    n_chains : int
        Chains. Chain families adapt once and vmap overdispersed chains; every other sampler
        runs independent chains from overdispersed starts. The draws of all chains are
        pooled in the posterior.
    metric : {"diag", "dense"}
        ``"dense"`` uses the family's own dense adaptation where it has one and a pilot
        covariance whitening otherwise (see :func:`tengri.inference.sampling.run_sampler`).
        A sampler that carries no metric records that the request was not applied in
        ``diagnostics["metric_applied"]``.
    **unknown
        Further settings of the shared driver (``n_ensemble``, ``max_rank``, ``smc_*``,
        ``vi_*``, ``is_*``, ``memory_budget_gb``, ``chain_jitter``; see
        :data:`tengri.inference.sampling.driver.DEFAULTS`). Anything else raises.
    allow_unvalidated : bool
        Opt in to ``tier="broken"`` backends, exactly as ``Fitter.run`` does.
        Required for ``pathfinder`` and ``mcmc_ghmc``, the tier="broken" names
        this seam drives, reachable, but not safe by default.
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
    shared high-dimension advisory, see
    ``tengri.inference._dimension_guard.warn_if_nuts_high_dim``.
    """
    import inspect
    import time

    from tengri.inference._backend_registry import check_usable, get_backend
    from tengri.inference._dimension_guard import warn_if_nuts_high_dim
    from tengri.inference.hierarchical import PopulationPosterior
    from tengri.inference.sampling.driver import DEFAULTS

    unrecognized = sorted(set(unknown) - set(DEFAULTS))
    if unrecognized:
        # The previous spelling was ``**_ignored``, a silent kwarg sink. A typo'd fit
        # option (or ``init_from``, which the hierarchical surface documents as
        # unsupported) vanished while the fit ran with defaults and the caller believed
        # otherwise (#1378's bug class).
        accepted = sorted(
            {
                p
                for p in inspect.signature(run_flat_sampler).parameters
                if p not in ("fitter", "method", "unknown")
            }
            | set(DEFAULTS)
        )
        raise TypeError(
            f"run_flat_sampler() got unexpected keyword argument(s) "
            f"{unrecognized}. Accepted fit options for the hierarchical "
            f"flat seam: {accepted}. Note that some options are per-family "
            f"(the ess driver ignores warmup knobs; the MCLMC family ignores "
            f"n_burnin) and init_from is not supported hierarchically, "
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
    # gate applies here, the method stays available, the caller just has to say
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

    # The sampler receives the log density as a module-level function plus the data as a
    # TRACED argument. Both halves are required for compile reuse: a fresh closure per call
    # is a new static compile key, and capturing the data bakes it in as a constant.
    target = FlatTarget(
        logdensity=prob.log_prob_with_data,
        layout=prob.layout if prob.layout is not None else LatentLayout.vector(prob.n_dim),
        init=prob.init_flat,
        item_args=prob.data_args,
        log_likelihood=prob.log_likelihood_with_data,
        prior_is_standard=float(prob.field_centering) == 1.0,
    )
    settings = {
        "n_warmup": n_warmup,
        "n_burnin": n_burnin,
        "n_samples": n_samples,
        "n_leapfrog": n_leapfrog,
        "max_num_doublings": max_num_doublings,
        "dense_mass_matrix": dense_mass_matrix,
        "target_accept_rate": target_accept_rate,
        "map_steps": map_steps,
        "map_learning_rate": map_learning_rate,
        "ghmc_alpha": ghmc_alpha,
        "ghmc_delta": ghmc_delta,
        "mclmc_target_accept_rate": mclmc_target_accept_rate,
        "laplace_grad_tol": laplace_grad_tol,
        "nss_n_live": nss_n_live,
        "nss_num_delete": nss_num_delete,
        "nss_num_inner_steps": nss_num_inner_steps,
        "nss_log_evidence_tol": nss_log_evidence_tol,
        "nss_max_iterations": nss_max_iterations,
        "verbose": verbose,
        **unknown,
    }
    t0 = time.time()
    result = run_sampler(
        driver,
        target,
        key=jax.random.fold_in(key, 1),
        n_chains=n_chains,
        metric=metric,
        **settings,
    )
    chain = result.draws.reshape(-1, prob.n_dim)
    extra = result.diagnostics

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
            "n_chains": int(result.draws.shape[0]),
            **extra,
        },
    )
