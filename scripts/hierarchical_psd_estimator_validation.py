# SPDX-License-Identifier: BSD-3-Clause
r"""Validate the two-step PSD estimator against an EXACT analytic posterior.

The population estimator ``shared_log_posterior`` is only correct given draws
from each galaxy's true interim posterior. Every end-to-end run so far supplies
approximate draws (Laplace, or HMC on a funnel geometry), so a failure cannot be
attributed to the estimator or to the approximation without a case where the
draws are exact by construction. This script is that case.

Construction
------------
The field ``m`` lives on the real 16-node age grid. The prior is the interim
prior the estimator already assumes: a finite mixture over the ``(sigma, tau)``
quadrature nodes of ``SharedGrid``, component ``(a, b)`` being the OU/DRW
Gaussian with covariance ``K_ij = (sigma ln10)^2 exp(-|t_i - t_j| / tau)`` and
mean ``-0.5 (sigma ln10)^2``. The observable is a linear projection onto the
``n_modes`` leading PCA modes of the true covariance, with Gaussian noise --
a stand-in for the fact that broadband photometry constrains only a handful of
field directions (measured ``n_eff`` approx 3.2-4.4 for real observables).

A Gaussian likelihood against a Gaussian-mixture prior gives a Gaussian-mixture
posterior in closed form, so draws are i.i.d. and exact:

.. math::

    S     &= P C P^T + \sigma_n^2 I \\
    G     &= C P^T S^{-1} \\
    \mu^* &= \mu + G (d - P \mu) \\
    \Sigma^* &= C - G P C \\
    \log w &= \log p_0 + \log N(d;\, P\mu,\, S)

Component weights come from ``grid.log_prior`` itself, so the draw density and
the ``p_0`` the estimator divides by cannot disagree -- the mismatch that
``SharedGrid.uniform``'s docstring warns costs a factor proportional to tau.

Why not importance sampling
---------------------------
The obvious alternative -- sample the prior, reweight by the likelihood -- fails
catastrophically here and silently. The prior standard deviation along the
leading modes is 4.59, 2.04, 1.74, ... against a likelihood width of 0.15, so
the proposal is 11-31x too broad *per constrained direction* and the
inefficiency compounds. Measured on this exact setup: Kish ESS of 1-5 out of
200,000 particles, and at 8 or more modes every 800-draw ensemble was 800 copies
of a single particle. An earlier version of this experiment used that scheme and
reached the opposite -- wrong -- conclusion. Prefer the closed form.

Interpretation
--------------
If the estimator recovers ``(sigma, tau)`` here but not end-to-end, the estimator
and the information content are exonerated and the fault is in the per-galaxy
posteriors. If it fails here too, the estimator itself is at fault.

Run from the worktree root::

    PYTHONPATH=src:. JAX_PLATFORMS=cpu python \
        scripts/hierarchical_psd_estimator_validation.py --bank psd_bank_fixed
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tengri.inference.population.diagnostics import credible_interval
from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

LOG10 = np.log(10.0)


def ou_cov(times_yr, sigma, tau_yr):
    """OU/DRW covariance of the log-SFR field.

    Parameters
    ----------
    times_yr : array_like, shape (n_grid,)
        Age-grid node times [yr].
    sigma : float
        Field amplitude [dex].
    tau_yr : float
        Correlation timescale [yr].

    Returns
    -------
    cov : ndarray, shape (n_grid, n_grid)
        Covariance in natural-log units [dimensionless].
    """
    lag = np.abs(times_yr[:, None] - times_yr[None, :])
    return (sigma * LOG10) ** 2 * np.exp(-lag / tau_yr)


def build_components(times_yr, grid, projection, noise):
    """Galaxy-independent per-node terms of the posterior mixture.

    Parameters
    ----------
    times_yr : ndarray, shape (n_grid,)
        Age-grid node times [yr].
    grid : SharedGrid
        Quadrature grid; its ``log_prior`` supplies the mixture weights.
    projection : ndarray, shape (n_modes, n_grid)
        Observation operator ``P``.
    noise : float
        Per-mode observation noise [dimensionless].

    Returns
    -------
    comps : dict
        Arrays keyed ``p_mu``, ``s_inv``, ``log_norm``, ``gain``, ``sqrt_cov``,
        ``mu``, each leading with the node axis.
    """
    sigma_nodes = np.repeat(np.asarray(grid.sigma), np.asarray(grid.tau_yr).size)
    tau_nodes = np.tile(np.asarray(grid.tau_yr), np.asarray(grid.sigma).size)
    n_nodes = sigma_nodes.size
    n_grid = times_yr.size
    n_modes = projection.shape[0]

    comps = {
        "p_mu": np.empty((n_nodes, n_modes)),
        "s_inv": np.empty((n_nodes, n_modes, n_modes)),
        "log_norm": np.empty(n_nodes),
        "gain": np.empty((n_nodes, n_grid, n_modes)),
        "sqrt_cov": np.empty((n_nodes, n_grid, n_grid)),
        "mu": np.empty((n_nodes, n_grid)),
    }
    eye_grid = np.eye(n_grid)
    eye_modes = np.eye(n_modes)
    for j in range(n_nodes):
        cov = ou_cov(times_yr, sigma_nodes[j], tau_nodes[j]) + 1e-10 * eye_grid
        mu = np.full(n_grid, -0.5 * (sigma_nodes[j] * LOG10) ** 2)
        cov_pt = cov @ projection.T
        marg = projection @ cov_pt + noise**2 * eye_modes
        _, logdet = np.linalg.slogdet(marg)
        s_inv = np.linalg.inv(marg)
        gain = cov_pt @ s_inv
        post_cov = cov - gain @ (projection @ cov)
        evals, evecs = np.linalg.eigh(0.5 * (post_cov + post_cov.T))
        comps["sqrt_cov"][j] = evecs * np.sqrt(np.clip(evals, 0.0, None))
        comps["p_mu"][j] = projection @ mu
        comps["s_inv"][j] = s_inv
        comps["gain"][j] = gain
        comps["mu"][j] = mu
        comps["log_norm"][j] = -0.5 * (logdet + n_modes * np.log(2 * np.pi))
    return comps


def draw_posterior(data, comps, log_prior, n_samples, rng):
    """Draw i.i.d. samples from the exact posterior mixture.

    Returns
    -------
    draws : ndarray, shape (n_samples, n_grid)
        Exact posterior draws [natural log units].
    """
    resid = data[None, :] - comps["p_mu"]
    quad = np.einsum("jk,jkl,jl->j", resid, comps["s_inv"], resid)
    log_w = log_prior + comps["log_norm"] - 0.5 * quad
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    n_grid = comps["mu"].shape[1]
    which = rng.choice(log_prior.size, n_samples, p=weights)
    draws = np.empty((n_samples, n_grid))
    for j in np.unique(which):
        sel = which == j
        mean = comps["mu"][j] + comps["gain"][j] @ resid[j]
        draws[sel] = mean + rng.standard_normal((int(sel.sum()), n_grid)) @ comps["sqrt_cov"][j].T
    return draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="psd_bank_fixed", help="bank dir supplying the age grid")
    parser.add_argument("--sigma", type=float, default=0.75, help="injected sigma [dex]")
    parser.add_argument("--tau-myr", type=float, default=150.0, help="injected tau [Myr]")
    parser.add_argument("--noise", type=float, default=0.15, help="per-mode observation noise")
    parser.add_argument("--n-samples", type=int, default=800, help="draws per galaxy")
    parser.add_argument("--modes", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--n-galaxies", type=int, nargs="+", default=[64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--node-chunk", type=int, default=32)
    args = parser.parse_args()

    with open(f"{args.bank}/bank_meta.json") as fh:
        meta = json.load(fh)
    times_yr = 10.0 ** np.asarray(meta["log_age_grid"])
    tau_yr = args.tau_myr * 1e6

    grid = SharedGrid.uniform(
        tau_prior="uniform",
        sigma_bounds=tuple(meta["interim_sigma_bounds"]),
        tau_bounds_yr=(
            meta["interim_tau_bounds_myr"][0] * 1e6,
            meta["interim_tau_bounds_myr"][1] * 1e6,
        ),
        n_sigma=60,
        n_tau=60,
    )
    log_prior = np.asarray(grid.log_prior)

    cov_true = ou_cov(times_yr, args.sigma, tau_yr) + 1e-10 * np.eye(times_yr.size)
    chol_true = np.linalg.cholesky(cov_true)
    mean_true = -0.5 * (args.sigma * LOG10) ** 2
    evals, evecs = np.linalg.eigh(cov_true)
    order = np.argsort(evals)[::-1]

    print(f"prior std per mode: {'  '.join(f'{np.sqrt(evals[j]):.2f}' for j in order)}")
    print(f"observation noise:  {args.noise}\n")
    print(
        f"{'modes':>5} {'N':>5} {'seed':>4}  "
        f"{f'sigma 68% ({args.sigma})':>21} {f'tau 68% Myr ({args.tau_myr:.0f})':>24} "
        f"{'resid':>6}"
    )

    for n_modes in args.modes:
        projection = evecs[:, order[:n_modes]].T
        comps = build_components(times_yr, grid, projection, args.noise)
        for n_gal in args.n_galaxies:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                fields = np.empty((n_gal, args.n_samples, times_yr.size))
                resid = []
                for i in range(n_gal):
                    truth = mean_true + chol_true @ rng.standard_normal(times_yr.size)
                    data = projection @ truth + args.noise * rng.standard_normal(n_modes)
                    fields[i] = draw_posterior(data, comps, log_prior, args.n_samples, rng)
                    resid.append(np.sqrt(np.mean((fields[i] @ projection.T - data) ** 2)))
                log_post, _ = shared_log_posterior(
                    fields, times_yr, grid, node_chunk=args.node_chunk
                )
                ci = credible_interval(np.asarray(log_post), grid)
                ok_s = "ok" if ci["sigma_lower"] <= args.sigma <= ci["sigma_upper"] else "MISS"
                ok_t = "ok" if ci["tau_lower_yr"] <= tau_yr <= ci["tau_upper_yr"] else "MISS"
                print(
                    f"{n_modes:5d} {n_gal:5d} {seed:4d}  "
                    f"{ci['sigma_lower']:8.3f}-{ci['sigma_upper']:<8.3f} {ok_s:>4} "
                    f"{ci['tau_lower_yr'] / 1e6:9.1f}-{ci['tau_upper_yr'] / 1e6:<9.1f} {ok_t:>4} "
                    f"{np.median(resid):6.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
