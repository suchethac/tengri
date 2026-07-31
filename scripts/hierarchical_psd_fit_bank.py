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
        choices=["mcmc_hmc", "laplace"],
        help="interim backend. laplace is ~1-2 s/galaxy warm vs ~155 s for HMC, "
        "at the cost of a Gaussian approximation to each per-galaxy posterior.",
    )
    ap.add_argument("--n-map-steps", type=int, default=4000, help="laplace only")
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

    print(
        f"bank {args.out}: galaxies [{args.start}, {end}) of {args.n}  "
        f"chains {args.n_chains}x({args.n_warmup}+{args.n_samples}) "
        f"L={args.n_leapfrog_steps} thin={args.thin}",
        flush=True,
    )

    t_start = time.time()
    n_done = 0
    for i in range(args.start, end):
        path = os.path.join(args.out, f"gal_{i:04d}.npz")
        if os.path.exists(path):
            continue

        t0 = time.time()
        fitter = Fitter(fit_model, flux[i], err[i])
        if args.method == "laplace":
            post = fitter.run(
                "laplace",
                key=keys[i],
                n_map_steps=args.n_map_steps,
                n_samples=args.n_samples * args.n_chains,
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
