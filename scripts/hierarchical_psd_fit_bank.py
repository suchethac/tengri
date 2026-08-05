# SPDX-License-Identifier: BSD-3-Clause
"""Fit a bank of per-galaxy interim posteriors once, to be subset many times.

Why a bank. The shared-PSD width-scaling measurement needs the interval at
several population sizes. Refitting at each N wastes the expensive stage:
a sweep over N = 4, 8, 12, 16 costs 40 galaxy-fits to produce 4 points, when
16 fits would do. ``make_population`` seeds galaxies with
``jax.random.split(key, N)``, whose first N outputs are a prefix of the longer
stream -- galaxy *i* is the same galaxy at every population size. So one bank
of N_MAX fits yields **every** smaller N by subsetting, and the points are
directly comparable to earlier sweeps rather than merely similar.

It also splits the 11-hour stage from the 14-second one. Once the bank is on
disk, the grid, the thinning, and b1-vs-b2 can all be varied for free by
rerunning :mod:`scripts.hierarchical_psd_subset_scaling` -- instead of paying
the fitting cost again to change one analysis knob.

Each galaxy is checkpointed to its own ``.npz`` the moment it finishes, and an
existing file is skipped. The failure mode this guards against is a kernel
SIGKILL under memory pressure, which produces no traceback and would otherwise
discard every completed fit.

Run with the worktree on the path (the editable install points at the MAIN
checkout, which does not carry tengri.inference.population)::

  PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_fit_bank.py --n 256 --out psd_bank/
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
import warnings

import jax
import numpy as np
from scripts.hierarchical_psd_recovery_run import (
    INTERIM_SIGMA_BOUNDS,
    INTERIM_TAU_BOUNDS_MYR,
    SNR_LINE,
    SNR_PHOT,
    TRUTH_SIGMA,
    TRUTH_TAU_MYR,
    build_model,
)

from tengri import Fitter, Parameters, Uniform, load_ssp_data
from tengri.analysis.diagnostics.autocorrelation import split_rhat
from tengri.analysis.population_mocks import make_population
from tengri.config.exceptions import LaplaceNotAtModeWarning
from tengri.inference.population.reconstruct import centered_fields

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


def _peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _interim_model(model):
    """Same model with the shared block widened to its interim prior.

    The interim prior must be broad enough to cover the shared posterior the
    population step will find, since the reweighting can only move mass to
    where the interim chains actually went.
    """
    spec_dict = {}
    for name in model.spec.free_params:
        if name == "sfh_field_psd_sigma":
            spec_dict[name] = Uniform(*INTERIM_SIGMA_BOUNDS)
        elif name == "sfh_field_psd_tau_myr":
            spec_dict[name] = Uniform(*INTERIM_TAU_BOUNDS_MYR)
        else:
            spec_dict[name] = model.spec.get_distribution(name)
    spec = Parameters(**spec_dict, n_grid=model.spec.n_grid)
    return type(model)(spec, model.ssp_data, observation=model.observation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="bank size (galaxies)")
    ap.add_argument("--out", default="psd_bank", help="checkpoint directory")
    ap.add_argument("--start", type=int, default=0, help="first galaxy index")
    ap.add_argument("--end", type=int, default=None, help="one past last galaxy index")
    ap.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=None,
        help="fit exactly these galaxy indices, ignoring --start/--end. For "
        "repairing a scattered subset -- e.g. the ~14%% of galaxies whose "
        "Laplace Hessian collapsed (#1537) -- without paying a model rebuild "
        "per galaxy the way one invocation each would.",
    )
    ap.add_argument("--n-warmup", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--n-leapfrog-steps", type=int, default=100)
    ap.add_argument(
        "--thin",
        type=int,
        default=None,
        help="store every thin-th draw. Default 8 for mcmc_hmc, 1 for laplace.",
    )
    ap.add_argument(
        "--method",
        default="mcmc_hmc",
        choices=["mcmc_hmc", "mcmc_nuts", "laplace"],
        help="interim backend. laplace is ~1-2 s/galaxy warm vs ~155 s for HMC, "
        "at the cost of a Gaussian approximation to each per-galaxy posterior. "
        "mcmc_nuts adapts its trajectory length, which static HMC cannot do on "
        "the bilinear (sigma, xi) funnel; it runs with dense_mass_matrix=False "
        "because a dense matrix at D=26 risks the 20+ GB warmup in CLAUDE.md.",
    )
    ap.add_argument("--n-map-steps", type=int, default=4000, help="laplace only")
    ap.add_argument(
        "--max-map-escalations",
        type=int,
        default=3,
        help=(
            "laplace only: how many times to triple n_map_steps while the fit "
            "reports it is not at a mode (#1537). 0 disables. Default 3 spans "
            "1x-27x, which covered every galaxy measured so far."
        ),
    )
    ap.add_argument(
        "--min-eigenvalue",
        type=float,
        default=1.0,
        help="laplace only: floor on Hessian eigenvalues. The backend default of "
        "1e-6 assigns clipped directions variance 1e6 (std 1000); 1.0 caps them at "
        "the unit-normal prior curvature instead. See the note in main().",
    )
    args = ap.parse_args()

    if args.thin is None:
        # Thinning removes MCMC autocorrelation -- measured ESS is ~600 of 4000
        # HMC draws, so keeping every 8th costs little. Laplace draws are i.i.d.
        # from a fitted Gaussian, so thinning them removes nothing and simply
        # discards samples. That matters here because the importance-weight ESS
        # scales with the draw count and is the estimator's scarcest resource:
        # on identical galaxies, K=500 thinned vs K=4000 unthinned moved ESS
        # from 9.4/10.3/3.8 to 25.0/92.3/46.3 for no extra fitting cost.
        args.thin = 1 if args.method == "laplace" else 8

    import tengri

    print("tengri:", tengri.__file__, flush=True)
    if "worktrees/hierarchical-psd-spec" not in tengri.__file__:
        raise SystemExit(
            "WRONG CHECKOUT — set PYTHONPATH=<worktree>/src. The editable install "
            "resolves to the main checkout, which has no inference.population."
        )

    os.makedirs(args.out, exist_ok=True)
    end = args.n if args.end is None else args.end

    ssp = load_ssp_data(SSP_PATH)
    model = build_model(ssp)
    pop = make_population(
        model,
        n_galaxies=args.n,
        sigma_true=TRUTH_SIGMA,
        tau_true_myr=TRUTH_TAU_MYR,
        key=jax.random.PRNGKey(0),
        snr_phot=SNR_PHOT,
        snr_line=SNR_LINE,
    )
    fit_model = _interim_model(model)

    # Persist the bank's identity next to the fits. A bank silently built at a
    # different truth, grid, or chain length would otherwise be indistinguishable
    # from a valid one, and the scaling analysis reads only the .npz files.
    meta = {
        "n": args.n,
        "truth_sigma": TRUTH_SIGMA,
        "truth_tau_myr": TRUTH_TAU_MYR,
        "interim_sigma_bounds": list(INTERIM_SIGMA_BOUNDS),
        "interim_tau_bounds_myr": list(INTERIM_TAU_BOUNDS_MYR),
        "n_grid": int(model.spec.n_grid),
        "n_warmup": args.n_warmup,
        "n_samples": args.n_samples,
        "n_chains": args.n_chains,
        "n_leapfrog_steps": args.n_leapfrog_steps,
        "method": args.method,
        "min_eigenvalue": args.min_eigenvalue,
        "snr_phot": SNR_PHOT,
        "snr_line": SNR_LINE,
        "thin": args.thin,
        "log_age_grid": np.asarray(model.log_age_grid).tolist(),
    }
    meta_path = os.path.join(args.out, "bank_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh)
        differing = {k for k in meta if k != "n" and old.get(k) != meta[k]}
        if differing:
            raise SystemExit(
                f"Bank at {args.out} was built with different settings: {sorted(differing)}. "
                "Mixing fits from different configurations would silently corrupt the "
                "scaling curve. Use a fresh --out directory."
            )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    flux = np.asarray(pop.table["phot_flux_obs"])
    err = np.asarray(pop.table["phot_flux_err"])
    keys = jax.random.split(jax.random.PRNGKey(1234), args.n)

    targets = list(range(args.start, end)) if args.only is None else sorted(set(args.only))
    if args.only is not None and (min(targets) < 0 or max(targets) >= args.n):
        raise SystemExit(
            f"--only indices must lie in [0, {args.n}); got "
            f"{min(targets)}..{max(targets)}. Galaxy i is only the same galaxy "
            "across runs when --n matches the bank it came from."
        )
    span = f"{len(targets)} listed" if args.only is not None else f"[{args.start}, {end})"
    print(
        f"bank {args.out}: galaxies {span} of {args.n}  "
        f"chains {args.n_chains}x({args.n_warmup}+{args.n_samples}) "
        f"L={args.n_leapfrog_steps} thin={args.thin}",
        flush=True,
    )

    t_start = time.time()
    n_done = 0
    for i in targets:
        path = os.path.join(args.out, f"gal_{i:04d}.npz")
        if os.path.exists(path):
            continue

        t0 = time.time()
        fitter = Fitter(fit_model, flux[i], err[i])
        map_steps_used = args.n_map_steps
        if args.method == "laplace":
            # min_eigenvalue is a variance CEILING, not a regularizer.
            #
            # run_laplace floors Hessian eigenvalues at min_eigenvalue to force
            # positive-definiteness, then takes cov = H^-1 -- so the floor sets
            # the variance of every clipped direction to 1/min_eigenvalue. At
            # the backend default of 1e-6 that is variance 1e6, i.e. std 1000.
            #
            # Ten broadband filters constrain ~4 modes of a 16-node field, so
            # ~12 field directions are unconstrained; galaxy 4 had 11 of 26
            # eigenvalues clipped and xi std 682, against an expected 1. But an
            # unconstrained direction is not infinitely uncertain: xi has a
            # N(0, 1) prior, so its variance is 1 and the posterior can never be
            # broader. Flooring at 1.0 caps posterior variance at prior variance.
            #
            # Measured, galaxies 0/3/4/5: xi std 1.20/108.1/682.2/85.8 at 1e-6
            # becomes 0.98/0.94/0.96/0.95 at 1.0, and the well-conditioned
            # galaxy 0 is unchanged. The heuristic holds because these latents
            # are unit-normal in unbounded space; it is NOT universal, and a
            # parameter whose unbounded prior is much wider than unit would be
            # over-constrained by it.
            # Escalate n_map_steps until the expansion point IS a mode.
            #
            # cov = H^-1 is a covariance only at a stationary point, and run_map
            # takes a fixed number of Adam steps with no convergence test. At
            # 4000 steps, 9 of the first 64 galaxies came back on a slope --
            # |grad| up to 3e5, and a xi covariance total of 5.2 against a prior
            # total of 16, i.e. a posterior 3x too narrow with nothing raised.
            # Measured at 40000 steps: all 9 repaired, covtot 16.6-17.7 (#1537).
            #
            # A fixed 10x is not a rule though -- it is one bank's answer. The
            # honest version is to ask the fit whether it converged and keep
            # going while it says no, which is what LaplaceNotAtModeWarning is
            # for. Galaxies that need more get more; the rest pay nothing.
            map_steps = args.n_map_steps
            for _attempt in range(args.max_map_escalations + 1):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", LaplaceNotAtModeWarning)
                    post = fitter.run(
                        "laplace",
                        key=keys[i],
                        n_map_steps=map_steps,
                        n_samples=args.n_samples * args.n_chains,
                        min_eigenvalue=args.min_eigenvalue,
                    )
                map_steps_used = map_steps
                if not any(isinstance(w.message, LaplaceNotAtModeWarning) for w in caught):
                    break
                map_steps *= 3
            else:
                # Exhausted the ladder. Record it rather than drop the galaxy:
                # excluding fits on an inferred quantity biases the population,
                # and the decrement is written to the npz so the pooling step
                # can weigh or report them explicitly.
                print(
                    f"  gal {i:4d}: STILL not at a mode after {map_steps // 3} "
                    f"steps (decrement {post.diagnostics['newton_decrement']:.4g})",
                    flush=True,
                )
        elif args.method == "mcmc_nuts":
            # The explicit per-galaxy MAP is NOT an optimization -- it is a
            # correctness requirement. Every sampler backend routes its
            # initialization through _maybe_map_init, which caches the MAP point
            # on the MODEL. This loop reuses one model and varies only the data,
            # so from the second galaxy onward every fit would silently start at
            # the PREVIOUS galaxy's MAP. Measured, galaxies 2-7 of this bank:
            # 5.1 s each instead of ~200 s, R-hat field max up to 10.74, and
            # zero divergences to mask it -- chains that never moved. Passing
            # init_from short-circuits the cache lookup. See issue #1529.
            #
            # No n_leapfrog_steps: NUTS chooses its own trajectory length, which
            # is the point of running it here. dense_mass_matrix=False because a
            # dense matrix at D=26 risks the 20+ GB warmup documented in
            # CLAUDE.md, and this backend exists to measure the xi covariance
            # SHAPE without a Gaussian approximation, not to be fast.
            map_post = fitter.run("map", key=keys[i], n_steps=args.n_map_steps, verbose=False)
            post = fitter.run(
                "mcmc_nuts",
                key=keys[i],
                init_from=map_post,
                n_warmup=args.n_warmup,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                dense_mass_matrix=False,
            )
        else:
            post = fitter.run(
                "mcmc_hmc",
                key=keys[i],
                n_warmup=args.n_warmup,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                n_leapfrog_steps=args.n_leapfrog_steps,
                dense_mass_matrix=True,
            )

        xi_full = np.asarray(post.samples["psd_xi"])
        sig_full = np.asarray(post.samples["sfh_field_psd_sigma"])
        tau_full = np.asarray(post.samples["sfh_field_psd_tau_myr"])
        # Laplace draws are i.i.d. from a fitted Gaussian, so R-hat is
        # identically ~1 by construction and diagnoses nothing. Recording it as
        # if it were a convergence check would manufacture false confidence --
        # the Laplace failure mode is that the Gaussian is the wrong SHAPE, and
        # no between-chain statistic can see that.
        if args.method == "laplace":
            rhat = {"sfh_field_psd_sigma": np.nan, "sfh_field_psd_tau_myr": np.nan}
            rhat["psd_xi"] = np.array([np.nan])
        else:
            rhat = post.rhat(exclude_prefixes=())
        n_div = getattr(post, "n_divergent", 0)
        if isinstance(n_div, dict):
            n_div = n_div.get("total", 0)

        # R-hat on the RECONSTRUCTED FIELD, which is what the estimator consumes.
        #
        # R-hat on psd_xi is not the relevant gate and reads as false
        # reassurance. The field is m = L(sigma, tau) . xi, so chains that agree
        # on xi but disagree on sigma disagree on m. That is exactly the
        # measured situation -- xi R-hat 0.994-0.998 against sigma R-hat 4.4 --
        # and reading the xi number alone says "converged" about draws that are
        # not draws from the interim posterior at all. Posterior.rhat even
        # excludes psd_xi by default, so the field is invisible unless asked for.
        if args.method == "laplace":
            rhat_field = np.full(int(model.spec.n_grid), np.nan)
        else:
            m_full = centered_fields(
                xi_full[None, :, :], sig_full[None, :], tau_full[None, :] * 1e6, model.log_age_grid
            )[0]
            rhat_field = np.array(
                [split_rhat(np.asarray(m_full[:, j])) for j in range(m_full.shape[1])]
            )

        xi = xi_full[:: args.thin]
        sig = sig_full[:: args.thin]
        tau = tau_full[:: args.thin]

        # Reject degenerate draws BEFORE checkpointing. A sampler whose chains
        # never move returns the right number of samples, all identical, with
        # zero divergences -- and the downstream estimator happily consumes them.
        # Measured: a 2-chain mcmc_nuts run produced seven such fits out of nine,
        # every draw bit-identical, R-hat 1e13 purely because the within-chain
        # variance underflowed. Writing those would poison the bank silently, so
        # skip and let the caller retry rather than record a fit that is not one.
        flat_xi = np.asarray(xi).reshape(-1, np.asarray(xi).shape[-1])
        n_draw, n_node = flat_xi.shape
        # Count DISTINCT draws, not a variance threshold. A variance floor is the
        # wrong instrument: chains frozen at their four starting points still
        # jitter by ~1e-7 in float64, which cleared an earlier 1e-8 cutoff while
        # the ensemble held four unique rows out of four thousand. The covariance
        # total is the second check because it is the statistic the estimator
        # actually consumes -- a healthy fit returns ~13-15 of 16, a dead one 0.00.
        n_unique = len(np.unique(flat_xi, axis=0))
        cov_total = float(np.linalg.eigvalsh(np.cov(flat_xi, rowvar=False)).sum())
        degenerate = (
            not np.isfinite(cov_total)
            or n_unique < max(10, 0.01 * n_draw)
            or cov_total < 0.05 * n_node
        )
        if degenerate:
            print(
                f"  gal {i:4d}: REJECTED — degenerate draws "
                f"({n_unique} unique of {n_draw}, xi covariance total "
                f"{cov_total:.3f} of {n_node}); the chains never moved. "
                "Not checkpointed.",
                flush=True,
            )
            del post, xi_full, sig_full, tau_full
            gc.collect()
            continue

        # Write to a temporary name and rename, so a kill mid-write cannot leave
        # a truncated .npz that the analysis would later load as valid.
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            xi=xi,
            sigma=sig,
            tau_myr=tau,
            n_divergent=np.asarray(int(n_div)),
            rhat_sigma=np.asarray(float(rhat.get("sfh_field_psd_sigma", np.nan))),
            rhat_tau=np.asarray(float(rhat.get("sfh_field_psd_tau_myr", np.nan))),
            rhat_xi_max=np.asarray(float(np.max(np.asarray(rhat["psd_xi"])))),
            rhat_field=rhat_field,
            # The convergence check R-hat cannot perform for a Laplace fit.
            # R-hat is ~1 by construction on i.i.d. Gaussian draws; the Newton
            # decrement is the number that separates a mode from a slope, so it
            # is the one worth carrying to the pooling step (#1537).
            newton_decrement=np.asarray(float(post.diagnostics.get("newton_decrement", np.nan))),
            map_steps_used=np.asarray(int(map_steps_used)),
        )
        os.replace(tmp, path)

        del post, fitter
        gc.collect()
        # Compiled programs accumulate across galaxies even at identical shapes;
        # two earlier sweeps were OOM-killed with no traceback for want of this.
        jax.clear_caches()

        n_done += 1
        dt = time.time() - t0
        rate = (time.time() - t_start) / n_done
        left = (end - i - 1) * rate
        print(
            f"  gal {i:4d}: {dt:5.1f}s  R-hat field max {float(np.max(rhat_field)):.2f} | "
            f"sigma {float(rhat.get('sfh_field_psd_sigma', np.nan)):.2f} "
            f"tau {float(rhat.get('sfh_field_psd_tau_myr', np.nan)):.2f}  "
            f"div {int(n_div):3d}  peak {_peak_rss_gb():.1f} GB  "
            f"ETA {left / 3600:.1f} h",
            flush=True,
        )

    have = len([f for f in os.listdir(args.out) if f.startswith("gal_")])
    print(f"\nbank {args.out}: {have}/{args.n} galaxies fitted in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
