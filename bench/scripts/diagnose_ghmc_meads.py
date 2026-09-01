#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Why MEADS-adapted GHMC does or does not converge on a notebook posterior.

``benchmark_notebook_sampler.py`` answers *whether* a sampler clears the bar.
This script answers *why*, which a table of R-hat cannot. Five probes, each
isolating one candidate explanation, all on the same model and seed the
benchmark uses:

``curvature``
    The latent-space Hessian spectrum at the MAP seed. MEADS's step-size rule
    is ``min(0.5 / sqrt(lambda_max(grad * sigma)), 1)``, which is an O(1) answer
    only when ``sigma`` is the posterior scale in every direction at once. This
    prints what that scale actually is, per direction.

``trace``
    Step size, damping, per-coordinate ensemble spread and acceptance across
    every adaptation step. A collapse shows up here as a step index, not as a
    final number.

``sweep``
    ``step_size_multiplier`` x ``damping_slowdown`` x dispersion x warmup. Rules
    out "the paper's constants happen to be wrong for this posterior".

``lrd``
    MEADS-LRD: the low-rank momentum metric (rank = D is the full dense metric).
    Rules out "the *diagonal* metric is the limitation".

``laplace``
    Seeds the ensemble from ``N(MAP, H^-1)`` so its covariance is already the
    posterior's. The most favorable initialization MEADS can be given short of
    sampling the posterior first; rules out "the ensemble just needed a better
    start".

``whiten``
    Runs the *same* MEADS on a linearly reparameterized target whose Hessian at
    the seed is the identity. If MEADS converges here and nowhere else, the
    limitation is the conditioning rather than the adaptation -- which is a
    statement about what would have to change, not just that something failed.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \\
        --notebook 05 --probe curvature
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

import tengri
from tengri import Fitter, generate_mock
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size, rhat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_notebook_sampler import NOTEBOOKS  # noqa: E402

from tengri.inference._sample_utils import _maybe_map_init  # noqa: E402
from tengri.inference.backends.mcmc._shared import _get_flat_logdensity  # noqa: E402


def build_target(nb: str):
    """Return ``(logdensity, seed_position, free-parameter names)`` in latent space."""
    cfg = NOTEBOOKS[nb]
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    fitter = Fitter(
        sed,
        np.asarray(mock["flux_obs"]),
        np.asarray(mock["noise"]),
        data_type="photometry",
    )
    init_params, _ = _maybe_map_init(fitter, key_fit, None, False)
    ld2, _unravel, x0, data_args = _get_flat_logdensity(fitter, init_params)

    def logdensity(x):
        return ld2(x, data_args)

    return logdensity, x0, list(init_params.keys())


def _sample_and_score(logdensity, last_states, params, names, n_chains=4, n_samples=4000):
    """Run ``n_chains`` GHMC chains from an adapted ensemble and score them."""
    kernel = blackjax.mcmc.ghmc.build_kernel()
    eps = params["step_size"]
    mis = params["momentum_inverse_scale"]
    alpha, delta = params["alpha"], params["delta"]
    dim = last_states.position.shape[-1]
    states = jax.tree.map(lambda leaf: leaf[:n_chains], last_states)
    keys = jax.random.split(jax.random.PRNGKey(13), n_chains * n_samples)
    keys = keys.reshape(n_chains, n_samples, 2)

    def step(s, k):
        s, info = kernel(k, s, logdensity, eps, mis, alpha, delta)
        return s, (s.position, info.is_divergent)

    positions, divergent = jax.vmap(lambda s, ks: jax.lax.scan(step, s, ks)[1])(states, keys)
    flat = np.asarray(positions.reshape(-1, dim))
    samples = {n: flat[:, i] for i, n in enumerate(names)}
    try:
        max_rhat = max(rhat(samples).values())
    except Exception:
        max_rhat = float("nan")
    ess = effective_sample_size(samples)
    finite = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
    worst = min(finite, key=lambda p: p[1]) if finite else ("?", float("nan"))
    return max_rhat, int(np.sum(np.asarray(divergent))), worst


def probe_curvature(logdensity, x0, names) -> None:
    hess = np.asarray(jax.hessian(lambda x: -logdensity(x))(x0))
    eig = np.linalg.eigvalsh(hess)
    print("D =", x0.shape[0])
    print("free parameters:", ", ".join(names))
    print("latent Hessian eigenvalues:", np.array2string(eig, precision=3))
    positive = eig[eig > 0]
    if positive.size:
        print(f"condition number over positive eigenvalues: {positive.max() / positive.min():.4g}")
    print(
        "implied latent sigma (1/sqrt|eig|):",
        np.array2string(1.0 / np.sqrt(np.abs(eig)), precision=4),
    )
    print(f"|grad| at the seed: {float(jnp.linalg.norm(jax.grad(logdensity)(x0))):.6g}")
    grad = jax.vmap(jax.grad(logdensity))
    for scale in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        pts = x0[None, :] + scale * jax.random.normal(
            jax.random.PRNGKey(3), (32, x0.shape[0]), dtype=x0.dtype
        )
        norms = jnp.linalg.norm(grad(pts), axis=1)
        print(
            f"  dispersion {scale:<5}: median |grad| {float(jnp.median(norms)):.4g}"
            f"  max {float(jnp.max(norms)):.4g}"
        )


def probe_trace(logdensity, x0, names, *, dispersion, n_warmup, n_ensemble) -> None:
    ensemble = x0[None, :] + dispersion * jax.random.normal(
        jax.random.PRNGKey(11), (n_ensemble, x0.shape[0]), dtype=x0.dtype
    )
    warmup = blackjax.meads_adaptation(logdensity, num_chains=n_ensemble, num_folds=4)
    (_last, params), info = warmup.run(jax.random.PRNGKey(12), ensemble, num_steps=n_warmup)
    step_size = np.asarray(info.adaptation_state.step_size)
    alpha = np.asarray(info.adaptation_state.alpha)
    position = np.asarray(info.state.position)
    accept = np.asarray(info.info.acceptance_rate)
    print(f"dispersion={dispersion} warmup={n_warmup} ensemble={n_ensemble}")
    print(f"{'step':>6}{'eps':>12}{'alpha':>10}{'accept':>9}{'sd_min':>11}{'sd_max':>11}{'|x|':>11}")
    steps = sorted({*range(min(12, n_warmup)), *range(0, n_warmup, max(1, n_warmup // 12))})
    for t in steps:
        sd = position[t].std(axis=0)
        print(
            f"{t:>6}{step_size[t].mean():>12.3g}{alpha[t].mean():>10.4g}"
            f"{np.nanmean(accept[t]):>9.3f}{sd.min():>11.3g}{sd.max():>11.3g}"
            f"{np.abs(position[t]).max():>11.3g}"
        )
    print("final:", {k: np.asarray(v).round(6).tolist() for k, v in params.items()})


_HEADER = (
    f"{'variant':<34}{'eps':>11}{'alpha':>10}{'wall':>8}{'maxRhat':>12}"
    f"{'div':>7}{'minESS':>9}  worst"
)


def _row(label, params, wall, score) -> None:
    max_rhat, divergences, worst = score
    print(
        f"{label:<34}{float(params['step_size']):>11.3g}{float(params['alpha']):>10.4g}"
        f"{wall:>8.1f}{max_rhat:>12.4f}{divergences:>7}{worst[1]:>9.1f}  {worst[0]}",
        flush=True,
    )


def probe_sweep(logdensity, x0, names) -> None:
    print(_HEADER)
    grid = itertools.product((0.5, 0.2, 0.1, 0.05, 0.02), (1.0, 4.0), (0.02, 0.05), (300, 1000))
    for mult, slowdown, dispersion, n_warmup in grid:
        ensemble = x0[None, :] + dispersion * jax.random.normal(
            jax.random.PRNGKey(11), (32, x0.shape[0]), dtype=x0.dtype
        )
        warmup = blackjax.meads_adaptation(
            logdensity,
            num_chains=32,
            num_folds=4,
            step_size_multiplier=mult,
            damping_slowdown=slowdown,
            adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
        )
        started = time.perf_counter()
        (last, params), _ = warmup.run(jax.random.PRNGKey(12), ensemble, num_steps=n_warmup)
        score = _sample_and_score(logdensity, last, params, names, n_samples=2000)
        _row(f"mult={mult} slow={slowdown} d={dispersion} w={n_warmup}",
             params, time.perf_counter() - started, score)


def probe_lrd(logdensity, x0, names) -> None:
    dim = x0.shape[0]
    print("LRD support in the installed blackjax:",
          hasattr(blackjax.mcmc.ghmc, "_metric_from_momentum_inverse_scale"))
    print(_HEADER)
    for rank, n_ensemble, n_warmup, dispersion in (
        (dim, 64, 1000, 0.05),
        (dim, 64, 4000, 0.05),
        (dim, 128, 4000, 0.05),
        (dim, 128, 4000, 0.2),
        (max(1, dim // 2), 64, 4000, 0.05),
    ):
        ensemble = x0[None, :] + dispersion * jax.random.normal(
            jax.random.PRNGKey(11), (n_ensemble, dim), dtype=x0.dtype
        )
        warmup = blackjax.meads_adaptation(
            logdensity,
            num_chains=n_ensemble,
            num_folds=4,
            low_rank_rank=rank,
            adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
        )
        started = time.perf_counter()
        (last, params), _ = warmup.run(jax.random.PRNGKey(12), ensemble, num_steps=n_warmup)
        score = _sample_and_score(logdensity, last, params, names)
        _row(f"rank={rank} E={n_ensemble} w={n_warmup} d={dispersion}",
             params, time.perf_counter() - started, score)


def probe_laplace(logdensity, x0, names) -> None:
    dim = x0.shape[0]
    hess = np.asarray(jax.hessian(lambda x: -logdensity(x))(x0))
    eig, vec = np.linalg.eigh(hess)
    cov = vec @ np.diag(1.0 / np.clip(eig, 1e-2, None)) @ vec.T
    chol = jnp.asarray(np.linalg.cholesky(cov + 1e-12 * np.eye(dim)))
    print("Laplace marginal sd:", np.array2string(np.sqrt(np.diag(cov)), precision=4))
    print(_HEADER)
    for n_ensemble, n_warmup, scale in ((64, 1000, 1.0), (64, 4000, 1.0), (64, 1000, 0.3),
                                        (128, 4000, 1.0)):
        z = jax.random.normal(jax.random.PRNGKey(11), (n_ensemble, dim), dtype=x0.dtype)
        ensemble = x0[None, :] + scale * z @ chol.T
        warmup = blackjax.meads_adaptation(
            logdensity,
            num_chains=n_ensemble,
            num_folds=4,
            adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
        )
        started = time.perf_counter()
        (last, params), _ = warmup.run(jax.random.PRNGKey(12), ensemble, num_steps=n_warmup)
        score = _sample_and_score(logdensity, last, params, names)
        _row(f"laplace E={n_ensemble} w={n_warmup} s={scale}",
             params, time.perf_counter() - started, score)


def probe_whiten(logdensity, x0, names) -> None:
    """MEADS on the same posterior, linearly whitened at the seed."""
    dim = x0.shape[0]
    hess = np.asarray(jax.hessian(lambda x: -logdensity(x))(x0))
    eig, vec = np.linalg.eigh(hess)
    # The seed is not always a mode -- negative curvature directions are real
    # here -- so clip rather than pretend. The point is a target whose scales are
    # comparable, not an exact Laplace approximation.
    transform = jnp.asarray(vec @ np.diag(1.0 / np.sqrt(np.clip(np.abs(eig), 1e-2, None))))
    print("condition number before whitening:",
          f"{np.abs(eig).max() / np.abs(eig).min():.4g}")

    def whitened(u):
        return logdensity(x0 + transform @ u)

    z0 = jnp.zeros(dim, dtype=x0.dtype)
    print(_HEADER)
    for n_ensemble, n_warmup, dispersion in ((64, 1000, 0.5), (64, 4000, 0.5), (128, 4000, 1.0)):
        ensemble = z0[None, :] + dispersion * jax.random.normal(
            jax.random.PRNGKey(11), (n_ensemble, dim), dtype=x0.dtype
        )
        warmup = blackjax.meads_adaptation(
            whitened,
            num_chains=n_ensemble,
            num_folds=4,
            adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
        )
        started = time.perf_counter()
        (last, params), _ = warmup.run(jax.random.PRNGKey(12), ensemble, num_steps=n_warmup)
        score = _sample_and_score(whitened, last, params, names)
        _row(f"whitened E={n_ensemble} w={n_warmup} d={dispersion}",
             params, time.perf_counter() - started, score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), default="05")
    parser.add_argument(
        "--probe",
        choices=("curvature", "trace", "sweep", "lrd", "laplace", "whiten"),
        default="curvature",
    )
    parser.add_argument("--dispersion", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--ensemble", type=int, default=32)
    args = parser.parse_args()

    logdensity, x0, names = build_target(args.notebook)
    if args.probe == "curvature":
        probe_curvature(logdensity, x0, names)
    elif args.probe == "trace":
        probe_trace(
            logdensity, x0, names,
            dispersion=args.dispersion, n_warmup=args.warmup, n_ensemble=args.ensemble,
        )
    elif args.probe == "sweep":
        probe_sweep(logdensity, x0, names)
    elif args.probe == "lrd":
        probe_lrd(logdensity, x0, names)
    elif args.probe == "laplace":
        probe_laplace(logdensity, x0, names)
    else:
        probe_whiten(logdensity, x0, names)


if __name__ == "__main__":
    main()
