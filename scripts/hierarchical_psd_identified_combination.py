# SPDX-License-Identifier: BSD-3-Clause
"""Which PSD combination does the population actually constrain?

The scaling driver reports marginal ``sigma`` and ``tau`` separately. For a
damped random walk that is not obviously the right pair to quote: below the
break frequency ``1 / tau`` the power spectral density is flat at

.. math::

    S(f \\to 0) = 4 \\sigma^2 \\tau

so unless the observable resolves the break, the population constrains the
**product** ``sigma^2 tau`` -- the integrated short-timescale power, "how bursty
overall" -- and the individual coordinates slide along a degeneracy ridge. That
is a statement about identifiability, not about estimator quality, and it makes
a 10x error in ``tau`` alone uninterpretable on its own.

This module does not assume the degeneracy. It measures it: the mass-weighted
covariance of the shared posterior in ``(log sigma, log tau)`` is
eigendecomposed, and the eigenvector with the SMALLEST eigenvalue is the
tightly-constrained direction. Its components are the exponents ``(a, b)`` of
the identified combination ``sigma^a tau^b``. A pure ``sigma^2 tau`` degeneracy
predicts ``b / a = 0.5``.

Run::

  PYTHONPATH=<worktree>/src:. JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_identified_combination.py --bank psd_bank_fixed
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from tengri.inference.population.estimator import SharedGrid, shared_log_posterior
from tengri.inference.population.reconstruct import centered_fields

# A node carrying this fraction of the total posterior mass on the grid
# boundary means the posterior is truncated, not resolved. Every summary below
# (interval, mode, degeneracy direction) is then a statement about where the
# grid was cut, so they are reported with an explicit flag rather than silently.
_EDGE_MASS_WARN = 0.10


def load_bank(bank_dir, n_max=None):
    """Load contiguous galaxies ``0..M`` from a bank directory.

    Stops at the first missing index rather than skipping it: the bank is a
    prefix of a keyed stream, so galaxies ``0..M-1`` are the M-galaxy
    population only if none is missing.

    Parameters
    ----------
    bank_dir : str
        Directory holding ``bank_meta.json`` and ``gal_*.npz``.
    n_max : int, optional
        Stop after this many galaxies [count].

    Returns
    -------
    meta : dict
        Bank metadata, including truth values and the interim prior bounds.
    xi : ndarray, shape (N, K, n)
        Interim latent draws [dimensionless].
    sigma : ndarray, shape (N, K)
        Interim ``sigma`` draws [dex].
    tau_myr : ndarray, shape (N, K)
        Interim ``tau`` draws [Myr].
    rhat_field : ndarray, shape (N,)
        Per-galaxy max R-hat on the reconstructed field [dimensionless].
    """
    with open(os.path.join(bank_dir, "bank_meta.json")) as fh:
        meta = json.load(fh)

    xi, sig, tau_myr, rhat_f = [], [], [], []
    i = 0
    while True:
        path = os.path.join(bank_dir, f"gal_{i:04d}.npz")
        if not os.path.exists(path) or (n_max is not None and i >= n_max):
            break
        with np.load(path) as d:
            xi.append(d["xi"])
            sig.append(d["sigma"])
            tau_myr.append(d["tau_myr"])
            rhat_f.append(float(np.max(d["rhat_field"])) if "rhat_field" in d else np.nan)
        i += 1

    if not xi:
        raise SystemExit(f"No gal_*.npz found in {bank_dir}")
    return meta, np.stack(xi), np.stack(sig), np.stack(tau_myr), np.array(rhat_f)


def posterior_mass(log_posterior, grid):
    """Normalized posterior mass per grid node.

    Parameters
    ----------
    log_posterior : array_like, shape (A * B,)
        Unnormalized log-posterior [nats], C-ordered so node ``a * B + b`` is
        ``(sigma[a], tau_yr[b])``.
    grid : SharedGrid
        Quadrature grid.

    Returns
    -------
    mass : ndarray, shape (A, B)
        Posterior mass summing to 1 [dimensionless].

    Notes
    -----
    The grid is uniform in ``sigma`` and geometric in ``tau``, so nodes carry
    equal quadrature weight in ``(sigma, log tau)`` coordinates and the mass is
    a plain normalized exponential. The max is subtracted before exponentiating
    because ``log_posterior`` sums one term per galaxy and reaches the thousands
    at large N, where a bare ``np.exp`` underflows every node to zero and the
    normalization returns NaN for the whole grid.
    """
    n_sigma = len(np.asarray(grid.sigma))
    n_tau = len(np.asarray(grid.tau_yr))
    lp = np.asarray(log_posterior).reshape(n_sigma, n_tau)
    mass = np.exp(lp - np.max(lp))
    return mass / mass.sum()


def edge_mass_fraction(mass):
    """Fraction of posterior mass on the grid boundary [dimensionless]."""
    edge = np.zeros_like(mass, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    return float(mass[edge].sum())


def marginal_interval(values, mass, level=0.68):
    """Equal-tailed credible interval of a derived quantity.

    Parameters
    ----------
    values : array_like, shape (A, B)
        Value of the derived quantity at each node, any units.
    mass : array_like, shape (A, B)
        Normalized posterior mass [dimensionless].
    level : float, optional
        Credible level [dimensionless]. Default 0.68.

    Returns
    -------
    lower, median, upper : float
        Quantiles of the derived quantity, in the units of ``values``.

    Notes
    -----
    Works on the sorted mass-weighted CDF rather than a histogram, so the
    result does not depend on a bin choice. The nodes are an irregular sample
    of the derived quantity -- ``sigma^2 tau`` is not monotone in either grid
    axis -- which is exactly the case a histogram handles badly.
    """
    v = np.asarray(values).ravel()
    w = np.asarray(mass).ravel()
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    tail = 0.5 * (1.0 - level)
    lo, med, hi = np.interp([tail, 0.5, 1.0 - tail], cdf, v)
    return float(lo), float(med), float(hi)


def identified_direction(mass, grid):
    """Principal axes of the shared posterior in ``(log sigma, log tau)``.

    Parameters
    ----------
    mass : array_like, shape (A, B)
        Normalized posterior mass [dimensionless].
    grid : SharedGrid
        Quadrature grid.

    Returns
    -------
    summary : dict
        ``"tight_exponents"`` is the ``(a, b)`` eigenvector of the smallest
        eigenvalue, normalized to ``a = 1`` where possible, so the identified
        combination is ``sigma^a tau^b``. ``"tau_exponent"`` is ``b / a``
        (0.5 for a pure ``sigma^2 tau`` degeneracy, 0 if ``sigma`` alone is
        constrained). ``"anisotropy"`` is the ratio of principal standard
        deviations [dimensionless]; near 1 means no degeneracy at all.

    Notes
    -----
    The covariance is taken in base-10 log coordinates, treated as Euclidean.
    The eigenvectors are therefore orthogonal by construction, which is an
    assumption about the ridge being locally straight -- adequate for
    characterizing a linear PSD degeneracy, not for a curved one. Meaningless
    when the posterior is truncated by the grid; check ``edge_mass_fraction``
    first.
    """
    log_sigma = np.log10(np.asarray(grid.sigma))
    log_tau = np.log10(np.asarray(grid.tau_yr))
    ls, lt = np.meshgrid(log_sigma, log_tau, indexing="ij")

    w = np.asarray(mass).ravel()
    x = np.stack([ls.ravel(), lt.ravel()])
    mean = np.einsum("ik,k->i", x, w)
    dx = x - mean[:, None]
    # einsum, not ``(dx * w) @ dx.T``: numpy 2.2's BLAS path raises spurious
    # divide-by-zero and overflow FP flags on this 2xM matmul even though every
    # input is finite and the result is bit-identical to einsum and np.cov
    # (verified). The warning is noise, but noise on the one number this
    # function exists to produce is worth not shipping.
    cov = np.einsum("ik,k,jk->ij", dx, w, dx)

    evals, evecs = np.linalg.eigh(cov)
    tight = evecs[:, 0]
    if tight[0] < 0:
        tight = -tight
    a, b = float(tight[0]), float(tight[1])
    return {
        "tight_exponents": (a, b),
        "tau_exponent": b / a if abs(a) > 1e-12 else np.inf,
        "anisotropy": float(np.sqrt(evals[1] / evals[0])) if evals[0] > 0 else np.inf,
        "sd_tight": float(np.sqrt(max(evals[0], 0.0))),
        "sd_loose": float(np.sqrt(max(evals[1], 0.0))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="psd_bank_fixed")
    ap.add_argument("--ns", default="", help="comma-separated N values; default powers of 2")
    ap.add_argument("--method", default="b2", choices=["b2", "b1"])
    ap.add_argument("--node-chunk", type=int, default=4)
    ap.add_argument("--n-sigma", type=int, default=60)
    ap.add_argument("--n-tau", type=int, default=60)
    ap.add_argument("--tau-prior", default="uniform", choices=["uniform", "log_uniform"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta, xi, sig, tau_myr, _rhat_f = load_bank(args.bank)
    n_have = xi.shape[0]
    log_age = np.asarray(meta["log_age_grid"])
    times_yr = 10.0**log_age
    truth_sigma = float(meta["truth_sigma"])
    truth_tau_myr = float(meta["truth_tau_myr"])
    truth_power = truth_sigma**2 * truth_tau_myr

    print(f"bank {args.bank}: {n_have} galaxies, K={xi.shape[1]}, n_grid={xi.shape[2]}")
    print(
        f"truth: sigma={truth_sigma} dex, tau={truth_tau_myr} Myr, "
        f"sigma^2 tau={truth_power:.1f} dex^2 Myr"
    )

    grid = SharedGrid.uniform(
        tau_prior=args.tau_prior,
        sigma_bounds=tuple(meta["interim_sigma_bounds"]),
        tau_bounds_yr=(
            meta["interim_tau_bounds_myr"][0] * 1e6,
            meta["interim_tau_bounds_myr"][1] * 1e6,
        ),
        n_sigma=args.n_sigma,
        n_tau=args.n_tau,
    )

    # Same pathological-galaxy rejection as the scaling driver: one galaxy with
    # a broken Laplace covariance has near-zero OU density everywhere but the
    # most extreme node, and its weight alone pins the shared posterior to a
    # grid corner at every N.
    sigma_max = float(meta["interim_sigma_bounds"][1])
    ceiling = 5.0 * sigma_max * np.log(10.0)
    all_fields = centered_fields(xi, sig, tau_myr * 1e6, log_age)
    per_gal_std = np.asarray(np.std(np.asarray(all_fields), axis=(1, 2)))
    keep = per_gal_std <= ceiling
    if not keep.all():
        print(f"  dropped {int((~keep).sum())} galaxies with implausible field scale")
        xi, sig, tau_myr = xi[keep], sig[keep], tau_myr[keep]
        n_have = int(keep.sum())

    n_values = (
        [int(x) for x in args.ns.split(",")]
        if args.ns
        else [n for n in (4, 8, 16, 32, 64, 128, 256, 512) if n <= n_have]
    )
    n_values = [n for n in n_values if n <= n_have]

    sigma_nodes = np.asarray(grid.sigma)
    tau_myr_nodes = np.asarray(grid.tau_yr) / 1e6
    s2d, t2d = np.meshgrid(sigma_nodes, tau_myr_nodes, indexing="ij")
    power = s2d**2 * t2d

    rows = []
    print(
        f"\n{'N':>5} {'sigma 68%':>20} {'tau/Myr 68%':>20} "
        f"{'s^2 tau 68%':>22} {'tau exp':>8} {'aniso':>7} {'edge':>6}"
    )
    print("-" * 92)
    for n in n_values:
        fields = centered_fields(xi[:n], sig[:n], tau_myr[:n] * 1e6, log_age)
        lp, _ess = shared_log_posterior(
            fields, times_yr, grid, method=args.method, node_chunk=args.node_chunk
        )
        mass = posterior_mass(np.asarray(lp), grid)
        edge = edge_mass_fraction(mass)

        s_lo, s_med, s_hi = marginal_interval(s2d, mass)
        t_lo, t_med, t_hi = marginal_interval(t2d, mass)
        p_lo, p_med, p_hi = marginal_interval(power, mass)
        direction = identified_direction(mass, grid)

        cov_s = s_lo <= truth_sigma <= s_hi
        cov_t = t_lo <= truth_tau_myr <= t_hi
        cov_p = p_lo <= truth_power <= p_hi
        flag = "*" if edge > _EDGE_MASS_WARN else " "
        print(
            f"{n:>5} {s_lo:6.3f}-{s_hi:6.3f} {'OK' if cov_s else '  ':>2}"
            f" {t_lo:7.1f}-{t_hi:7.1f} {'OK' if cov_t else '  ':>2}"
            f" {p_lo:8.1f}-{p_hi:8.1f} {'OK' if cov_p else '  ':>2}"
            f" {direction['tau_exponent']:>8.2f} {direction['anisotropy']:>7.1f}"
            f" {edge:>5.2f}{flag}"
        )
        rows.append(
            {
                "n": n,
                "sigma": [s_lo, s_med, s_hi],
                "tau_myr": [t_lo, t_med, t_hi],
                "power": [p_lo, p_med, p_hi],
                "covers_sigma": bool(cov_s),
                "covers_tau": bool(cov_t),
                "covers_power": bool(cov_p),
                "edge_mass": edge,
                **{k: v for k, v in direction.items()},
            }
        )

    print(
        f"\n* = more than {_EDGE_MASS_WARN:.0%} of posterior mass on the grid boundary: "
        "the posterior is truncated, so its interval and degeneracy direction "
        "describe where the grid was cut, not the population."
    )
    print(
        "tau exp = exponent b in the identified combination sigma^1 tau^b "
        "(0.5 = pure sigma^2 tau power degeneracy; 0.0 = sigma alone constrained).\n"
        "aniso   = ratio of principal standard deviations; 1 means no degeneracy."
    )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "bank": args.bank,
                    "truth": {
                        "sigma": truth_sigma,
                        "tau_myr": truth_tau_myr,
                        "power": truth_power,
                    },
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
