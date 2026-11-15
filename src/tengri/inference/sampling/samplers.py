# SPDX-License-Identifier: BSD-3-Clause
"""The registered samplers of the shared driver.

Each runner takes a :class:`~tengri.inference.sampling.target.FlatTarget`, a key, ``n_chains``,
the requested metric and the resolved settings, and returns a
:class:`~tengri.inference.sampling.driver.SamplerResult` of shape ``(n_chains, n, D)``. The
chain families run through ``chain_program.run_chain``; the others are the implementations
the hierarchical flat seam already ran (elliptical slice, MCLMC, nested slice, MAP,
Laplace), the single-galaxy backends' scans (ChEES, tempered SMC, low-rank HMC, Gaussian
VI), and importance-sampled HMC over the target.
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference.sampling import guards
from tengri.inference.sampling.chains import (
    OVERDISPERSION,
    PRIOR_SCALE,
    UNADAPTED_START_SCALE,
    jittered_starts,
    start_scale,
)
from tengri.inference.sampling.driver import (
    Sampler,
    SamplerResult,
    register_sampler,
    run_sampler,
)
from tengri.inference.sampling.families import DENSE_FAMILIES, resolve_accept, run_chain_family
from tengri.inference.sampling.target import FlatTarget

_BYTES = 8


def _jitter(cfg: dict, target: FlatTarget | None = None) -> float | None:
    """The isotropic start scale of ``target``, or ``None`` for the family's own default.

    One rule for every sampler (see :func:`~.chains.start_scale`): ``chain_jitter`` wins; a
    pilot-whitened target, whose posterior has unit scale, gets ``OVERDISPERSION``; anything
    else ``None``, which the chain families resolve to the prior's scale when their adaptation
    estimates a metric and to ``UNADAPTED_START_SCALE`` when it does not, and the
    samplers without an adapted metric (:func:`_unadapted`) to ``UNADAPTED_START_SCALE``.
    """
    if cfg["chain_jitter"] is not None:
        return float(cfg["chain_jitter"])
    if target is not None and target.extras.get("whitened"):
        return OVERDISPERSION
    return None


def _unadapted(cfg: dict, target: FlatTarget) -> float:
    """The start scale of a sampler with no adapted metric."""
    scale = _jitter(cfg, target)
    return UNADAPTED_START_SCALE if scale is None else scale


def _bound(target: FlatTarget):
    """Return ``logp(x)`` with the target's item arguments bound."""
    ld, args = target.logdensity, target.item_args
    return lambda x: ld(x, args)


def _gaussian_logprior(x, args):
    """Unnormalized Gaussian log prior; ``args = (item_args, (mean, precision_factor))``."""
    mean, precision = args[1]
    return -0.5 * jnp.sum((precision @ (x - mean)) ** 2)


@functools.cache
def _likelihood_of_pair(log_likelihood):
    """Return the likelihood reading the item arguments out of ``(item_args, prior)``."""

    def likelihood(x, args):
        return log_likelihood(x, args[0])

    return likelihood


def _gaussian_prior_or_refuse(target: FlatTarget, label: str):
    """Return ``(mean, precision_factor)`` of the target's Gaussian prior or refuse by name."""
    if target.prior_is_standard or target.prior is not None:
        return target.gaussian_prior()
    raise ValueError(
        f"{label}: this sampler draws from the target's Gaussian prior, which is not this "
        "target's (a partially centered GP-field prior, sfh={'field_centering': a < 1}). "
        "Use mcmc_nuts, mcmc_hmc or map there, or set field_centering=1."
    )


def _independent(single, target: FlatTarget, key, n_chains: int, cfg: dict) -> SamplerResult:
    """Run ``single`` once per chain from overdispersed starts and stack the chains.

    This is the ``n_chains`` layer for samplers without a shared adaptation to vmap: every
    chain is a complete, independent run, so a split R-hat over them is a real test.
    """
    if n_chains == 1:
        return single(target, key)
    keys = jax.random.split(key, n_chains)
    starts = jittered_starts(
        target.init, key, n_chains, _jitter(cfg, target) or UNADAPTED_START_SCALE
    )
    runs = [
        single(FlatTarget(**{**_fields(target), "init": starts[c]}), keys[c])
        for c in range(n_chains)
    ]
    div = [r.divergent for r in runs]
    return SamplerResult(
        draws=jnp.concatenate([r.draws for r in runs], axis=0),
        divergent=None if any(d is None for d in div) else jnp.concatenate(div, axis=0),
        diagnostics={**runs[0].diagnostics, "n_chains": n_chains},
    )


def _fields(target: FlatTarget) -> dict:
    return {
        "logdensity": target.logdensity,
        "layout": target.layout,
        "init": target.init,
        "item_args": target.item_args,
        "log_likelihood": target.log_likelihood,
        "prior_is_standard": target.prior_is_standard,
        "prior": target.prior,
        "extras": target.extras,
    }


def _warn_footprint(label: str, nbytes: float, cfg: dict) -> None:
    """Warn, never refuse, when an estimated footprint exceeds ``memory_budget_gb``."""
    gb = nbytes / 1e9
    if gb > cfg["memory_budget_gb"]:
        warnings.warn(
            f"{label}: estimated memory footprint {gb:.1f} GB exceeds memory_budget_gb="
            f"{cfg['memory_budget_gb']:.1f}; running as requested.",
            UserWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------------------
# Chain families
# --------------------------------------------------------------------------------------


def _chain_runner(family: str, *, force_dense: bool = False):
    def run(target, key, n_chains, metric, cfg):
        from tengri.inference.backends.mcmc.chain_program import divergence_record

        dense = force_dense or bool(cfg["dense_mass_matrix"]) or metric == "dense"
        if dense and family in DENSE_FAMILIES:
            _warn_footprint(f"{family} dense metric", 4 * _BYTES * target.n_dim**2, cfg)
        n_burn, n_chain = cfg["n_burnin"], cfg["n_burnin"] + cfg["n_samples"]
        wkey, ckey = jax.random.split(key)
        draws, (step_size, _inverse_scale) = run_chain_family(
            family,
            target.logdensity,
            target.item_args,
            target.init,
            wkey,
            ckey,
            n_chains=n_chains,
            n_chain=n_chain,
            n_warmup=cfg["n_warmup"],
            cfg=cfg,
            dense=dense,
            chain_jitter=_jitter(cfg, target),
        )
        position, divergent = draws.position, draws.divergent
        if n_chains == 1:
            position = position[None]
            divergent = None if divergent is None else divergent[None]
        position = position[:, n_burn:]
        divergent = None if divergent is None else divergent[:, n_burn:]
        record = divergence_record(divergent)
        diagnostics = {"step_size": float(jnp.mean(step_size)), "n_warmup": cfg["n_warmup"]}
        if record:
            diagnostics["divergent"] = int(record["n_divergent"])
            diagnostics["divergent_mask"] = np.asarray(record["divergent_mask"]).reshape(-1)
        return SamplerResult(position, divergent, diagnostics)

    return run


# --------------------------------------------------------------------------------------
# Elliptical slice
# --------------------------------------------------------------------------------------


def _run_ess(target, key, n_chains, metric, cfg):
    from tengri.inference.backends.mcmc._shared import _ess_full_scan

    def single(tgt, k):
        n_chain = cfg["n_burnin"] + cfg["n_samples"]
        # The flat prior is exactly the iid N(0, 1) the ESS ellipse assumes, so the
        # sampler's one structural requirement holds by construction.
        mean, precision = _gaussian_prior_or_refuse(tgt, "mcmc_ess")
        positions, subiters = _ess_full_scan(
            tgt.init,
            jax.random.split(k, n_chain),
            tgt.likelihood(),
            tgt.item_args,
            mean,
            jnp.linalg.inv(precision.T @ precision),
        )
        return SamplerResult(
            positions[None, cfg["n_burnin"] :],
            None,
            {
                "mean_subiter": float(jnp.mean(subiters[cfg["n_burnin"] :])),
                "n_burnin": cfg["n_burnin"],
            },
        )

    return _independent(single, target, key, n_chains, cfg)


# --------------------------------------------------------------------------------------
# MCLMC family
# --------------------------------------------------------------------------------------


def _mclmc_runner(adjusted: bool):
    def run(target, key, n_chains, metric, cfg):
        from tengri.inference.backends.mcmc._shared import (
            _adjusted_mclmc_sample_scan,
            _check_blackjax_floor,
            _mclmc_sample_scan,
        )

        _check_blackjax_floor()
        import blackjax
        from blackjax.mcmc import integrators

        label = "adjusted_mclmc" if adjusted else "mclmc"
        bound = _bound(target)
        # One kernel object for every chain: it is a static argument of the sampling scan.
        family = blackjax.mcmc.adjusted_mclmc if adjusted else blackjax.mcmc.mclmc
        kernel = family.build_kernel(integrator=integrators.isokinetic_mclachlan)

        def single(tgt, k):
            # The target of every chain shares its density and data, so one bound function is
            # one static jit key: the sampling scan compiles once, not once per chain.
            ld = bound
            # No burn-in phase: the (L, step size) tuning consumes the transient, so the
            # chain is n_samples long and n_burnin is ignored.
            tune_key, init_key, ckey = jax.random.split(k, 3)
            chain_keys = jax.random.split(ckey, cfg["n_samples"])
            if not adjusted:
                state = blackjax.mcmc.mclmc.init(tgt.init, ld, init_key)
                state, params, _ = blackjax.mclmc_find_L_and_step_size(
                    mclmc_kernel=kernel,
                    logdensity_fn=ld,
                    num_steps=cfg["n_warmup"],
                    state=state,
                    rng_key=tune_key,
                    diagonal_preconditioning=True,
                )
                guards._require_finite_tuning(params.L, params.step_size, label, cfg["n_warmup"])
                _, chain = _mclmc_sample_scan(
                    state,
                    chain_keys,
                    kernel,
                    params.L,
                    params.step_size,
                    ld,
                    params.inverse_mass_matrix,
                )
                extra = {"L": float(params.L), "step_size": float(params.step_size)}
                return SamplerResult(chain[None], None, {**extra, "n_warmup": cfg["n_warmup"]})
            state = blackjax.mcmc.adjusted_mclmc.init(tgt.init, ld)
            state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                logdensity_fn=ld,
                num_steps=cfg["n_warmup"],
                state=state,
                rng_key=tune_key,
                target=cfg["mclmc_target_accept_rate"],
                diagonal_preconditioning=True,
            )
            guards._require_finite_tuning(params.L, params.step_size, label, cfg["n_warmup"])
            n_int = jnp.ceil(params.L / params.step_size).astype(int)
            _, (chain, divergent) = _adjusted_mclmc_sample_scan(
                state, chain_keys, kernel, params.step_size, n_int, ld, params.inverse_mass_matrix
            )
            return SamplerResult(
                chain[None],
                divergent[None],
                {
                    "L": float(params.L),
                    "step_size": float(params.step_size),
                    "n_integration_steps": int(n_int),
                    "divergent": int(jnp.sum(divergent)),
                    "n_warmup": cfg["n_warmup"],
                },
            )

        return _independent(single, target, key, n_chains, cfg)

    return run


# --------------------------------------------------------------------------------------
# Nested slice sampling
# --------------------------------------------------------------------------------------


def _run_nss(target, key, n_chains, metric, cfg):
    from tengri.inference.backends.evidence import _require_converged_evidence
    from tengri.inference.backends.nested.nss import as_top_level_api
    from tengri.inference.backends.nested.utils import (
        ess as ns_ess,
        finalize as ns_finalize,
        sample as ns_sample,
    )

    _gaussian_prior_or_refuse(target, "nss")

    def single(tgt, k):
        guards._require_nondegenerate_live_set(cfg["nss_n_live"], tgt.n_dim, "nss")
        num_inner = tgt.n_dim if cfg["nss_num_inner_steps"] is None else cfg["nss_num_inner_steps"]
        loglik = tgt.likelihood()
        mean, precision = _gaussian_prior_or_refuse(tgt, "nss")
        # NSS's native compile-once mode: a 2-arg likelihood plus data_args carried in the
        # sampler state, so the data stays a traced value. max_shrinkage=20 keeps the
        # compiled vmap(while_loop) graph bounded (100 compiled a 20+ GB graph).
        algo = as_top_level_api(
            lambda x: -0.5 * jnp.sum((precision @ (x - mean)) ** 2),
            loglik,
            num_inner,
            num_delete=cfg["nss_num_delete"],
            max_steps=10,
            max_shrinkage=20,
            data_args=tgt.item_args,
        )
        init_key, loop_key = jax.random.split(k)
        eps = jax.random.normal(init_key, (cfg["nss_n_live"], tgt.n_dim))
        # A draw from the prior: mean + P^-1 eps, row-wise.
        draws = mean + jax.scipy.linalg.solve(precision, eps.T).T
        live = algo.init(draws, data_args=tgt.item_args)
        step = jax.jit(algo.step)
        dead, n_iter = [], 0
        while True:
            loop_key, sk = jax.random.split(loop_key)
            live, dead_info = step(sk, live)
            dead.append(dead_info._replace(update_info=None))
            n_iter += 1
            remaining = float(live.integrator.logZ_live - live.integrator.logZ)
            if remaining < cfg["nss_log_evidence_tol"] or n_iter >= cfg["nss_max_iterations"]:
                break
        _require_converged_evidence(
            n_iter, cfg["nss_max_iterations"], remaining, cfg["nss_log_evidence_tol"], "nss"
        )
        log_z = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
        ns_run = ns_finalize(live, dead, update_info=False)
        sample_key, ess_key = jax.random.split(jax.random.fold_in(k, 2))
        chain = ns_sample(sample_key, ns_run, cfg["n_samples"]).position
        return SamplerResult(
            chain[None],
            None,
            {
                "log_evidence": log_z,
                "ess": float(ns_ess(ess_key, ns_run)),
                "n_live": cfg["nss_n_live"],
                "num_delete": cfg["nss_num_delete"],
                "num_inner_steps": num_inner,
                "n_iterations": n_iter,
                "n_dead": n_iter * cfg["nss_num_delete"],
            },
        )

    return _independent(single, target, key, n_chains, cfg)


# --------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------


def _curvature_pilot(target, key, n_chains, metric, cfg):
    """Pilot draws for a point estimate: Gaussian draws from the curvature at a diagonal MAP.

    The diagonal MAP ascent is the sampler's own cheap pass; one dense Hessian at its end
    (D forward-over-reverse passes) gives ``Sigma = (-H)^-1`` with the eigenvalues taken in
    absolute value and floored at ``1e-6`` of the largest, so a point that is not quite a mode
    still yields a positive definite covariance. About ``map_steps + D`` gradient evaluations,
    against a NUTS pilot's warmup plus ``4 (D + 1)`` draws of tens of gradients each.
    """
    logp = _bound(target)
    mode = guards.adam_map_ascent(logp, target.init, cfg["map_steps"], cfg["map_learning_rate"])
    values, vectors = jnp.linalg.eigh(-jax.hessian(logp)(mode))
    magnitude = jnp.abs(values)
    magnitude = jnp.maximum(magnitude, 1e-6 * jnp.max(magnitude))
    factor = vectors / jnp.sqrt(magnitude)
    eps = jax.random.normal(key, (cfg["n_samples"], target.n_dim), dtype=mode.dtype)
    return SamplerResult((mode[None, :] + eps @ factor.T)[None], None, {"pilot": "curvature"})


def _run_map(target, key, n_chains, metric, cfg):
    def single(tgt, k):
        logp = _bound(tgt)
        best = guards.adam_map_ascent(logp, tgt.init, cfg["map_steps"], cfg["map_learning_rate"])
        # A point estimate is a length-1 "chain", so downstream extraction is shared.
        return SamplerResult(
            best[None, None],
            None,
            {"n_steps": cfg["map_steps"], "log_prob": float(logp(best))},
        )

    return _independent(single, target, key, n_chains, cfg)


def _run_laplace(target, key, n_chains, metric, cfg):
    def single(tgt, k):
        logp = _bound(tgt)
        # MAP + a Gaussian covariance from the curvature AT A VERIFIED MODE: the bare point
        # estimate under 'laplace' would silently drop the error bars, and curvature off a
        # mode is a plausible wrong answer (#1537).
        # The mode comes from the MAP sampler, under the requested metric, so a dense request
        # optimizes in whitened coordinates before the Hessian is read.
        mode = run_sampler(
            "map",
            tgt,
            key=k,
            metric=metric,
            **{**cfg, "health_warnings": False},
        ).draws[0, 0]
        # Adam alone plateaus at hierarchical D; the Newton polish reaches the actual mode
        # with the same Hessian machinery the covariance needs.
        mode = guards.newton_polish_mode(logp, mode, tol=cfg["laplace_grad_tol"], max_iters=40)
        grad_at_mode = jax.grad(logp)(mode)
        guards._require_converged_mode(
            grad_at_mode, "laplace", cfg["map_steps"], tol=cfg["laplace_grad_tol"]
        )
        chol = jnp.linalg.cholesky(-jax.hessian(logp)(mode))
        guards._require_psd_curvature(chol, "laplace")
        eps = jax.random.normal(k, (cfg["n_samples"], tgt.n_dim))
        # cov = (-H)^{-1}: -H = L L^T, draw = mode + solve(L^T, eps).
        chain = mode[None, :] + jax.scipy.linalg.solve_triangular(chol.T, eps.T, lower=False).T
        return SamplerResult(
            chain[None],
            None,
            {
                "n_steps": cfg["map_steps"],
                "log_prob": float(logp(mode)),
                "grad_inf_norm": float(jnp.max(jnp.abs(grad_at_mode))),
                "n_samples": cfg["n_samples"],
            },
        )

    return _independent(single, target, key, n_chains, cfg)


# --------------------------------------------------------------------------------------
# Variational
# --------------------------------------------------------------------------------------

MEANFIELD_WARNING = (
    "vi_meanfield: mean-field drops cross-level correlations. Every galaxy's latents correlate "
    "with the population hyperparameters, and a diagonal Gaussian reports conditional widths."
)


def _vi_runner(family: str):
    def run(target, key, n_chains, metric, cfg):
        from tengri.inference.backends.vi.gaussian import fit_gaussian_vi

        if family == "fullrank":
            _warn_footprint("vi_fullrank", _BYTES * target.n_dim * (target.n_dim + 1), cfg)

        def single(tgt, k):
            samples, info = fit_gaussian_vi(
                k,
                _bound(tgt),
                tgt.init,
                family=family,
                n_steps=cfg["vi_n_steps"],
                n_samples=cfg["n_samples"],
                n_mc_samples=cfg["vi_n_mc_samples"],
                learning_rate=cfg["vi_learning_rate"],
            )
            return SamplerResult(samples[None], None, info)

        return _independent(single, target, key, n_chains, cfg)

    return run


# --------------------------------------------------------------------------------------
# ChEES ensemble
# --------------------------------------------------------------------------------------


def _run_chees(target, key, n_chains, metric, cfg):
    from tengri.inference.backends.mcmc._shared import (
        _CHEES_JITTER_SCALE,
        _chees_scan,
        _resolve_chees_ensemble,
    )
    from tengri.inference.backends.mcmc.chees import CHEES_TARGET_ACCEPT_RATE

    ensemble = _resolve_chees_ensemble(cfg["n_ensemble"], n_chains)
    n_chain = cfg["n_burnin"] + cfg["n_samples"]
    _warn_footprint("chees ensemble", ensemble * (n_chain + 4) * target.n_dim * _BYTES, cfg)
    # ChEES's fixed-length proposal has its own asymptotic optimum (0.651); the HMC-family
    # default is not carried across.
    accept = (
        CHEES_TARGET_ACCEPT_RATE
        if cfg["target_accept_rate"] is None
        else cfg["target_accept_rate"]
    )
    wkey, ckey = jax.random.split(key)
    positions, divergent, step_size, _inv_mass, n_leapfrog = _chees_scan(
        target.init,
        wkey,
        jax.random.split(ckey, n_chains * n_chain).reshape(n_chains, n_chain, 2),
        target.logdensity,
        target.item_args,
        cfg["n_warmup"],
        ensemble,
        n_chains,
        n_chain,
        float(_CHEES_JITTER_SCALE),
        1.0,
        accept,
        cfg["max_leapfrog_steps"],
        0.025,
        None,
        None if cfg["chain_jitter"] is None else float(cfg["chain_jitter"]),
    )
    n_burn = cfg["n_burnin"]
    return SamplerResult(
        positions[:, n_burn:],
        divergent[:, n_burn:],
        {
            "step_size": float(jnp.mean(step_size)),
            "n_ensemble": ensemble,
            "n_leapfrog": int(jnp.max(n_leapfrog)),
            "divergent": int(jnp.sum(divergent[:, n_burn:])),
        },
    )


# --------------------------------------------------------------------------------------
# Tempered SMC
# --------------------------------------------------------------------------------------


def _run_smc(target, key, n_chains, metric, cfg):
    from tengri.inference.backends.mcmc._shared import (
        _SMC_INIT_STEP_SIZE,
        _SMC_STEP_SIZE_GAIN,
        _smc_scan,
    )
    from tengri.inference.backends.mcmc.smc import SMC_TARGET_ACCEPT_RATE

    mean, precision = _gaussian_prior_or_refuse(target, "mcmc_smc")
    n_particles = cfg["smc_n_particles"]
    _warn_footprint("smc", n_chains * n_particles * target.n_dim * _BYTES * 8, cfg)
    run_keys = jax.random.split(key, n_chains)
    particles, log_z, n_temp, ladder, _step, _n_div, _accept, min_ess = _smc_scan(
        jnp.linalg.inv(precision),
        run_keys,
        (target.item_args, (mean, precision)),
        _gaussian_logprior,
        _likelihood_of_pair(target.likelihood()),
        n_particles,
        int(cfg["smc_n_mcmc_steps"]),
        int(cfg["n_leapfrog"]) * 2,
        float(cfg["smc_target_ess"]),
        float(_SMC_INIT_STEP_SIZE),
        float(_SMC_STEP_SIZE_GAIN),
        float(SMC_TARGET_ACCEPT_RATE),
        int(cfg["smc_max_temperatures"]),
        None,
        mean,
    )
    reached = all(float(v) >= 1.0 for v in np.asarray(ladder).reshape(-1))
    return SamplerResult(
        particles.reshape(n_chains, -1, target.n_dim),
        None,
        {
            "log_evidence": [float(v) for v in np.asarray(log_z).reshape(-1)],
            "n_temperatures": [int(v) for v in np.asarray(n_temp).reshape(-1)],
            "reached_target": reached,
            "min_ancestor_ess": float(jnp.min(min_ess)),
            "n_particles": n_particles,
        },
    )


# --------------------------------------------------------------------------------------
# Low-rank HMC
# --------------------------------------------------------------------------------------


def _run_lowrank(target, key, n_chains, metric, cfg):
    import blackjax

    from tengri.inference.backends.mcmc._shared import (
        _get_hmc_kernel,
        _hmc_low_rank_warmup_only,
        _stabilize_dense_mass_step,
    )
    from tengri.inference.backends.mcmc.chain_program import HMC, HMCConfig, sample_chain

    n_chain = cfg["n_burnin"] + cfg["n_samples"]
    accept = resolve_accept("hmc", cfg)
    wkey, ckey = jax.random.split(key)
    ld, args = target.logdensity, target.item_args
    # One warmup shared by every chain, as run_hmc does, from the un-jittered center: the
    # adaptation then tunes at the point the others are dispersed around, and its metric
    # sets how far they are dispersed.
    step_size, inv_mass, warm_div = _hmc_low_rank_warmup_only(
        target.init,
        wkey,
        ld,
        args,
        int(cfg["n_warmup"]),
        int(cfg["n_leapfrog"]),
        int(cfg["max_rank"]),
        float(accept),
    )
    if n_chains == 1:
        starts = target.init[None, :]
    else:
        scale = start_scale(
            inv_mass, PRIOR_SCALE if _jitter(cfg, target) is None else _jitter(cfg, target)
        )
        starts = jittered_starts(target.init, wkey, n_chains, scale)
    state0 = blackjax.mcmc.hmc.init(starts[0], lambda p: ld(p, args))
    step_size, _backoff = _stabilize_dense_mass_step(
        _get_hmc_kernel(),
        state0,
        ld,
        args,
        float(step_size),
        inv_mass,
        int(cfg["n_leapfrog"]),
        sampler_name="HMC low-rank",
    )
    chain_keys = jax.random.split(ckey, n_chains * n_chain).reshape(n_chains, n_chain, 2)
    hmc_cfg = HMCConfig(
        n_leapfrog=int(cfg["n_leapfrog"]), use_dense=True, target_accept_rate=accept
    )
    positions, divergent = [], []
    for c in range(n_chains):
        state = blackjax.mcmc.hmc.init(starts[c], lambda p: ld(p, args))
        draws = sample_chain(
            HMC, state, chain_keys[c], ld, args, (jnp.asarray(step_size), inv_mass), hmc_cfg
        )
        positions.append(draws.position[cfg["n_burnin"] :])
        divergent.append(draws.divergent[cfg["n_burnin"] :])
    div = jnp.stack(divergent)
    return SamplerResult(
        jnp.stack(positions),
        div,
        {
            "step_size": float(step_size),
            "max_rank": int(cfg["max_rank"]),
            "divergent": int(jnp.sum(div)),
            "n_warmup": cfg["n_warmup"],
            "warmup_divergent": int(jnp.sum(warm_div)),
            "lowrank_rank": int(inv_mass.U.shape[1]),
            "lowrank_dim": int(target.n_dim),
            "lowrank_weakest_log_lambda": float(jnp.min(jnp.abs(jnp.log(inv_mass.lam)))),
        },
    )


# --------------------------------------------------------------------------------------
# HMC with importance-sampled evidence
# --------------------------------------------------------------------------------------


def _run_hmc_is(target, key, n_chains, metric, cfg):
    from tengri.inference.backends.mcmc.hmc_is import _fit_proposal
    from tengri.inference.importance import importance_sample

    hmc_key, is_key = jax.random.split(key)
    chain = run_sampler(
        "hmc",
        target,
        key=hmc_key,
        n_chains=n_chains,
        metric=metric,
        **{**cfg, "health_warnings": False},
    )
    flat = np.asarray(chain.draws.reshape(-1, target.n_dim))
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] < max(10, 2 * target.n_dim):
        raise ValueError(
            f"hmc_is: only {flat.shape[0]} finite chain draws remain (D={target.n_dim}); "
            "the posterior is pinned to a boundary. Use nss for this model."
        )
    proposal = _fit_proposal(flat, cfg["is_proposal_df"], cfg["is_proposal_inflation"])
    result = importance_sample(
        target.logdensity,
        target.item_args,
        proposal,
        cfg["is_n_draws"],
        is_key,
        log_norm=float(-0.5 * target.n_dim * np.log(2.0 * np.pi)),
        batch_size=4096,
    )
    return SamplerResult(
        chain.draws,
        chain.divergent,
        {
            **chain.diagnostics,
            "log_evidence": float(result.log_z),
            "log_evidence_err": float(result.log_z_err),
            "is_ess": float(result.ess),
            "max_weight_frac": float(result.max_weight_frac),
            "pareto_k": float(result.psis.pareto_k),
        },
    )


# --------------------------------------------------------------------------------------
# Registrations
# --------------------------------------------------------------------------------------

#: Chain families the vmapped catalog engine runs (``nuts_fast`` as dense ``nuts``).
_VMAPPABLE = ("nuts", "hmc", "dynamic_hmc", "ghmc", "barker", "mala", "nuts_fast")

#: Why the samplers below have no vmapped catalog engine.
_PATHFINDER_NOT_VMAPPABLE = (
    "its L-BFGS Gaussian warmup is registered tier='broken' (SIGKILL out of memory at "
    "D=18 on a two-galaxy hierarchy), and no run has measured it under a galaxy vmap, where "
    "K copies of its optimizer state would be live at once"
)

for _name, _methods, _family, _dense in (
    ("nuts", ("mcmc_nuts", "mcmc"), "nuts", False),
    ("nuts_pathfinder", ("pathfinder",), "nuts_pathfinder", False),
    ("hmc", ("mcmc_hmc",), "hmc", False),
    ("dynamic_hmc", ("mcmc_dynamic_hmc",), "dynamic_hmc", False),
    ("ghmc", ("mcmc_ghmc",), "ghmc", False),
    ("barker", ("mcmc_barker",), "barker", False),
    ("mala", ("mcmc_mala",), "mala", False),
    ("nuts_fast", ("mcmc_nuts_fast",), "nuts", True),
):
    register_sampler(
        Sampler(
            _name,
            "chain",
            _methods,
            _chain_runner(_family, force_dense=_dense),
            metric_mode="native" if _family in DENSE_FAMILIES else "layer",
            vmappable=_name
            in ("nuts", "hmc", "dynamic_hmc", "ghmc", "barker", "mala", "nuts_fast"),
            engine="nuts" if _name == "nuts_fast" else None,
            dense_default=_dense,
            not_vmappable=_PATHFINDER_NOT_VMAPPABLE if _name == "nuts_pathfinder" else None,
        )
    )

register_sampler(Sampler("chees", "ensemble", ("mcmc_chees",), _run_chees, vmappable=True))
register_sampler(Sampler("smc", "ensemble", ("mcmc_smc",), _run_smc))
register_sampler(Sampler("ess", "chain", ("mcmc_ess",), _run_ess))
register_sampler(Sampler("mclmc", "chain", ("mcmc_mclmc",), _mclmc_runner(False)))
register_sampler(Sampler("adjusted_mclmc", "chain", ("mcmc_adjusted_mclmc",), _mclmc_runner(True)))
register_sampler(Sampler("hmc_lowrank", "chain", ("mcmc_hmc_lowrank",), _run_lowrank))
register_sampler(Sampler("hmc_is", "chain", ("hmc_is",), _run_hmc_is, metric_mode="internal"))
register_sampler(Sampler("nss", "nested", ("nss",), _run_nss))
register_sampler(
    Sampler("map", "point", ("map",), _run_map, fidelity="point", pilot=_curvature_pilot)
)
register_sampler(Sampler("laplace", "point", ("laplace",), _run_laplace, metric_mode="invariant"))
register_sampler(Sampler("vi_fullrank", "variational", ("vi_fullrank",), _vi_runner("fullrank")))
register_sampler(
    Sampler(
        "vi_meanfield",
        "variational",
        ("vi_meanfield",),
        _vi_runner("meanfield"),
        fidelity="mean_only",
        advisory=MEANFIELD_WARNING,
    )
)
