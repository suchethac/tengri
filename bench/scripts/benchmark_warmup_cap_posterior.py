#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Does a shallower warmup cap change the posterior, or only the clock?

``bench/reports/2026-08-31_fast_nuts.md`` Finding 9 measured
``warmup_max_num_doublings=5`` running **8.9x faster** than the uncapped control
on ``ctl-dpl`` seed 7, at a **10.6x** better seconds-per-effective-sample, with
a *better* max split-R-hat (1.0017 against 1.0051) and a *better* min ESS (257.3
against 215.3).

**That is not enough to adopt it, and the reason is one column.** The capped run
reports **20 divergences against the control's 1**. Divergences are the only
diagnostic in that row that indicates *bias* rather than inefficiency: R-hat
asks whether chains agree with each other and ESS asks how fast they decorrelate,
and neither can see a sampler that is efficiently exploring the wrong
distribution. This project has twice caught exactly that -- ``vi_fullrank`` at
1.6 dex with a monotone ELBO, ``pathfinder`` at 0.11x posterior width with
healthy-looking diagnostics -- so a 9x that quietly biases the posterior is a
known failure mode here, not a hypothetical one.

So the decisive measurement is **posterior agreement, per parameter**, and this
script is it. Both configurations run on the same model, the same mock, the same
seed and the same MAP warm start, **in one process**, so nothing but
``warmup_max_num_doublings`` differs. For every free parameter it reports:

* the marginal mean and sd under each configuration;
* the mean shift in units of the **control's** sd (the number that says whether
  a scientific conclusion would change);
* the sd ratio (the number that catches a collapsed or inflated width, which is
  how ``pathfinder`` failed);
* a two-sample comparison against the Monte Carlo error each run actually
  earned, ``sd / sqrt(ESS)``, rather than against ``sd / sqrt(n_draws)`` --
  autocorrelated draws do not supply ``n`` independent ones, and using the draw
  count would declare agreement that has not been demonstrated.

**Per parameter, never in aggregate.** #1436's rule -- "coverage has to be
enumerated by seam, not by 'a representative model'" -- applies for the same
reason: an aggregate hides the single direction that moved, and on this fixture
the candidates are named in advance. ``sfh_dpl_age_gyr`` is the control's worst
parameter and ``dust_tau_diff`` is the capped run's, so if the divergences
concentrate anywhere it is there.

It also splits divergences by **phase**. ``Posterior.diagnostics`` carries
``n_divergent`` over kept draws and, separately, ``warmup_divergence_frac`` for
the final warmup window. Divergences while a capped warmup is still adapting
mean something different from divergences during sampling under a step size that
warmup mis-tuned: the first is the adaptation working through a bad region, the
second is the adapted step size being wrong.

Usage::

    JAX_PLATFORMS=cpu python bench/scripts/benchmark_warmup_cap_posterior.py \\
        --notebook ctl-dpl --seed 7 --warmup-cap 5 \\
        --json bench/results/2026-08-31_warmup_cap_posterior.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _machine_load() -> dict:
    """Load average at the moment of measurement (see the report's Caveat 1)."""
    try:
        with open("/proc/loadavg") as fh:
            a, b, c = fh.read().split()[:3]
    except OSError:
        return {}
    return {"load1": float(a), "load5": float(b), "load15": float(c), "n_cpu": os.cpu_count()}


def run(notebook: str, seed: int, warmup_cap: int) -> dict:
    """Fit the same galaxy twice, differing only in the warmup tree cap."""
    import jax
    import numpy as np
    from benchmark_notebook_sampler import NOTEBOOKS

    import tengri
    from tengri import Data, ForwardModel, generate_mock
    from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    shipped = dict(cfg["shipped"])
    arms = {
        "control": dict(shipped),
        f"wcap={warmup_cap}": dict(shipped, warmup_max_num_doublings=int(warmup_cap)),
    }

    out: dict = {}
    for label, kwargs in arms.items():
        # A fresh ForwardModel per arm: adaptation caches are keyed on tuning
        # settings, and a fresh build also keeps the MAP seed identical per arm,
        # which is what makes the two posteriors comparable at all.
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        map_seed = forward.fit(
            data, method="map", key=k_fit, n_restarts=8, n_steps=800, verbose=False
        )
        t0 = time.perf_counter()
        post = forward.fit(
            data,
            key=k_fit,
            init_from=map_seed,
            n_chains=cfg["n_chains"],
            verbose=False,
            **kwargs,
        )
        wall = time.perf_counter() - t0
        d = dict(post.diagnostics)
        stats = {}
        for name, draws in post.samples.items():
            arr = np.asarray(draws).ravel()
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            ess = float(effective_sample_size(np.asarray(draws)))
            sd = float(np.std(arr, ddof=1))
            stats[name] = {
                "mean": float(np.mean(arr)),
                "sd": sd,
                "ess": ess,
                # MC error of the mean from the draws the chain ACTUALLY earned.
                # sd/sqrt(n_draws) would assume independence the chain does not
                # have and would manufacture agreement.
                "mcse": sd / max(np.sqrt(max(ess, 1e-12)), 1e-12),
            }
        out[label] = {
            "kwargs": {k: v for k, v in kwargs.items()},
            "wall_s": round(wall, 1),
            "n_divergent_sampling": d.get("n_divergent"),
            "warmup_divergence_frac_final_window": d.get("warmup_divergence_frac"),
            "step_size": d.get("step_size"),
            "tree_depth_mean": d.get("tree_depth_mean"),
            "frac_max_depth": d.get("frac_max_depth"),
            "max_num_doublings": d.get("max_num_doublings"),
            "warmup_max_num_doublings": d.get("warmup_max_num_doublings"),
            "n_samples": d.get("n_samples"),
            "n_chains": d.get("n_chains"),
            "params": stats,
        }

    ctrl_label, cap_label = "control", f"wcap={warmup_cap}"
    ctrl, cap = out[ctrl_label]["params"], out[cap_label]["params"]
    comparison = {}
    for name in sorted(set(ctrl) & set(cap)):
        c, w = ctrl[name], cap[name]
        # Combined MC error of the DIFFERENCE of two means. If |shift| exceeds a
        # few of these the two runs are not sampling the same distribution.
        joint = float(np.sqrt(c["mcse"] ** 2 + w["mcse"] ** 2))
        comparison[name] = {
            "control_mean": c["mean"],
            "cap_mean": w["mean"],
            "control_sd": c["sd"],
            "cap_sd": w["sd"],
            "shift_in_control_sd": (w["mean"] - c["mean"]) / c["sd"] if c["sd"] > 0 else None,
            "sd_ratio_cap_over_control": (w["sd"] / c["sd"]) if c["sd"] > 0 else None,
            "shift_in_joint_mcse": ((w["mean"] - c["mean"]) / joint) if joint > 0 else None,
            "control_ess": c["ess"],
            "cap_ess": w["ess"],
        }

    return {
        "probe": "warmup_cap_posterior",
        "notebook": notebook,
        "seed": seed,
        "warmup_cap": int(warmup_cap),
        "device": str(jax.devices()[0].platform),
        **_machine_load(),
        "arms": out,
        "comparison": comparison,
    }


def main(argv=None) -> int:
    from benchmark_notebook_sampler import NOTEBOOKS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default="ctl-dpl", choices=sorted(NOTEBOOKS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup-cap", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    row = run(args.notebook, args.seed, args.warmup_cap)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(row, fh, indent=2)

    print(f"\n{args.notebook} seed {args.seed}: control vs warmup cap {args.warmup_cap}")
    for label, arm in row["arms"].items():
        print(
            f"  {label:<10} wall {arm['wall_s']:>8.1f}s  step {arm['step_size']:.5f}  "
            f"depth {arm['tree_depth_mean']:.2f}  "
            f"div(sampling) {arm['n_divergent_sampling']}  "
            f"div(final warmup window) {arm['warmup_divergence_frac_final_window']}"
        )
    hdr = f"\n{'parameter':<28}{'shift/sd':>10}{'sd ratio':>10}{'shift/MCSE':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name, c in row["comparison"].items():
        print(
            f"{name:<28}{c['shift_in_control_sd']:>10.3f}"
            f"{c['sd_ratio_cap_over_control']:>10.3f}{c['shift_in_joint_mcse']:>12.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
