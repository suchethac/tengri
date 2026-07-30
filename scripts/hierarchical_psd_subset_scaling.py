# SPDX-License-Identifier: BSD-3-Clause
"""Width-scaling curve from a fitted bank, by subsetting rather than refitting.

The bank written by :mod:`scripts.hierarchical_psd_fit_bank` holds one interim
posterior per galaxy. Because ``jax.random.split(key, N)`` yields a prefix
stream, the first N galaxies of the bank ARE the N-galaxy population -- so
every population size is a subset, and the whole scaling curve costs one pass
of the estimator (~14 s per point) instead of a refit (~hours per point).

The gate is ``interval_width_scaling``: regress log(width) on log(N) and ask
whether the slope excludes zero. A flat slope means the shared posterior is
prior-dominated and the companion paper's 1/sqrt(N) claim does not hold.

Run::

  PYTHONPATH=<worktree>/src:. JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_subset_scaling.py --bank psd_bank/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from tengri.inference.population.diagnostics import (
    credible_interval,
    interval_width_scaling,
)
from tengri.inference.population.estimator import SharedGrid, shared_log_posterior
from tengri.inference.population.reconstruct import centered_fields


def load_bank(bank_dir, n_max=None):
    """Load contiguous galaxies 0..M from a bank directory.

    Stops at the first missing index rather than skipping it: the bank is a
    prefix of a keyed stream, so galaxies 0..M-1 are the M-galaxy population
    only if none is missing. Silently closing a gap would relabel which
    galaxies a given N refers to.
    """
    with open(os.path.join(bank_dir, "bank_meta.json")) as fh:
        meta = json.load(fh)

    xi, sig, tau_myr, rhat_s, rhat_t, rhat_x, rhat_f, ndiv = [], [], [], [], [], [], [], []
    i = 0
    while True:
        path = os.path.join(bank_dir, f"gal_{i:04d}.npz")
        if not os.path.exists(path) or (n_max is not None and i >= n_max):
            break
        with np.load(path) as d:
            xi.append(d["xi"])
            sig.append(d["sigma"])
            tau_myr.append(d["tau_myr"])
            rhat_s.append(float(d["rhat_sigma"]))
            rhat_t.append(float(d["rhat_tau"]))
            rhat_x.append(float(d["rhat_xi_max"]))
            # The field R-hat is the gate; older banks predate it.
            rhat_f.append(float(np.max(d["rhat_field"])) if "rhat_field" in d else np.nan)
            ndiv.append(int(d["n_divergent"]))
        i += 1

    if not xi:
        raise SystemExit(f"No gal_*.npz found in {bank_dir}")
    return (
        meta,
        np.stack(xi),
        np.stack(sig),
        np.stack(tau_myr),
        np.array(rhat_s),
        np.array(rhat_t),
        np.array(rhat_x),
        np.array(rhat_f),
        np.array(ndiv),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="psd_bank")
    ap.add_argument("--ns", default="", help="comma-separated N values; default powers of 2")
    ap.add_argument("--method", default="b2", choices=["b2", "b1"])
    ap.add_argument("--node-chunk", type=int, default=64)
    ap.add_argument("--n-sigma", type=int, default=60)
    ap.add_argument("--n-tau", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta, xi, sig, tau_myr, rhat_s, rhat_t, rhat_x, rhat_f, ndiv = load_bank(args.bank)
    n_have = xi.shape[0]
    log_age = np.asarray(meta["log_age_grid"])
    times_yr = 10.0**log_age
    truth_sigma = meta["truth_sigma"]
    truth_tau_myr = meta["truth_tau_myr"]

    print(f"bank {args.bank}: {n_have} galaxies, K={xi.shape[1]} draws, n_grid={xi.shape[2]}")
    print(
        f"interim R-hat across the bank -- sigma med {np.median(rhat_s):.2f} "
        f"max {np.max(rhat_s):.2f} | tau med {np.median(rhat_t):.2f} max {np.max(rhat_t):.2f} "
        f"| xi max {np.max(rhat_x):.3f} | divergences {int(ndiv.sum())}"
    )
    # R-hat is reported before the answer on purpose. A shared posterior built
    # from unconverged interim chains is precise and wrong, and the earlier
    # sweep railed to the prior bounds for exactly that reason.
    # The FIELD R-hat is the gate, not sigma's and certainly not xi's. The
    # estimator consumes m = L(sigma, tau) . xi, so chains agreeing on xi while
    # disagreeing on sigma still disagree on m -- and Posterior.rhat excludes
    # psd_xi by default, which is how "latents converged" became false comfort.
    if np.all(np.isfinite(rhat_f)):
        print(
            f"interim R-hat on the RECONSTRUCTED FIELD -- med {np.median(rhat_f):.2f} "
            f"max {np.max(rhat_f):.2f}  ({100 * float(np.mean(rhat_f > 1.05)):.0f}% above 1.05)"
        )
        frac_bad = float(np.mean(rhat_f > 1.05))
        gate = "field"
    else:
        print("  (bank predates the field R-hat; falling back to sigma's)")
        frac_bad = float(np.mean(rhat_s > 1.05))
        gate = "sigma"
    if frac_bad > 0.1:
        print(
            f"  WARNING: {100 * frac_bad:.0f}% of galaxies have R-hat({gate}) > 1.05. "
            "Pooling reduces variance, not bias: at large N this yields a TIGHTER "
            "wrong answer. Treat the intervals below as a measurement of the "
            "SAMPLER, not of the population."
        )

    if args.ns:
        n_values = [int(x) for x in args.ns.split(",")]
    else:
        n_values = [n for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= n_have]
    n_values = [n for n in n_values if n <= n_have]
    if len(n_values) < 3:
        raise SystemExit(
            f"Need at least 3 population sizes to fit a slope; bank has {n_have} galaxies."
        )

    grid = SharedGrid.uniform(
        sigma_bounds=tuple(meta["interim_sigma_bounds"]),
        tau_bounds_yr=(
            meta["interim_tau_bounds_myr"][0] * 1e6,
            meta["interim_tau_bounds_myr"][1] * 1e6,
        ),
        n_sigma=args.n_sigma,
        n_tau=args.n_tau,
    )

    rows = []
    print(f"\n{'N':>5}  {'sigma (68%)':>22}  {'tau Myr (68%)':>24}  {'ESS':>7}")
    for n in n_values:
        fields = centered_fields(xi[:n], sig[:n], tau_myr[:n] * 1e6, log_age)
        lp, ess = shared_log_posterior(
            fields, times_yr, grid, method=args.method, node_chunk=args.node_chunk
        )
        ci = credible_interval(np.asarray(lp), grid)
        s_lo, s_hi = ci["sigma_lower"], ci["sigma_upper"]
        t_lo, t_hi = ci["tau_lower_yr"] / 1e6, ci["tau_upper_yr"] / 1e6
        rows.append(
            {
                "n": n,
                "sigma_lo": s_lo,
                "sigma_hi": s_hi,
                "sigma_width": s_hi - s_lo,
                "sigma_covers": bool(s_lo <= truth_sigma <= s_hi),
                "tau_lo_myr": t_lo,
                "tau_hi_myr": t_hi,
                "tau_width_myr": t_hi - t_lo,
                "tau_covers": bool(t_lo <= truth_tau_myr <= t_hi),
                "ess_at_mode_min": float(np.min(np.asarray(ess.at_mode))),
            }
        )
        print(
            f"{n:5d}  {s_lo:8.3f}-{s_hi:<8.3f} {'OK' if rows[-1]['sigma_covers'] else 'miss':>4}  "
            f"{t_lo:9.1f}-{t_hi:<9.1f} {'OK' if rows[-1]['tau_covers'] else 'miss':>4}  "
            f"{rows[-1]['ess_at_mode_min']:7.1f}",
            flush=True,
        )

    bar = "=" * 72
    print(
        f"\n{bar}\nWIDTH SCALING — the gate  "
        f"(truth sigma={truth_sigma}, tau={truth_tau_myr} Myr)\n{bar}"
    )
    n_arr = np.array([r["n"] for r in rows], dtype=float)
    verdict = {}
    for label, key in (("sigma", "sigma_width"), ("tau", "tau_width_myr")):
        w = np.array([r[key] for r in rows], dtype=float)
        out = interval_width_scaling(w, n_arr)
        verdict[label] = out
        print(
            f"{label:6s}: slope {out['slope']:+.3f} +/- {out['slope_err']:.3f}  "
            f"excludes zero at 3 sigma: {out['excludes_zero_3sigma']}  "
            f"[{'PASS' if out['excludes_zero_3sigma'] else 'FAIL'}]"
        )
    print("\nA slope near -0.5 excluding zero means pooling works. A slope consistent")
    print("with zero means the posterior is prior-dominated, whatever the medians look like.")

    out_path = args.out or os.path.join(args.bank, f"scaling_{args.method}.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "bank": args.bank,
                "method": args.method,
                "n_galaxies_available": n_have,
                "rows": rows,
                "verdict": {k: dict(v) for k, v in verdict.items()},
                "interim_rhat_sigma_median": float(np.median(rhat_s)),
                "interim_rhat_sigma_max": float(np.max(rhat_s)),
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
