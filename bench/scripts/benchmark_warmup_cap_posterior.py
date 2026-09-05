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


def run(notebook: str, seed: int, warmup_cap, target_accepts=(), preconditions=()) -> dict:
    """Fit one galaxy under several adaptations, changing one thing each time.

    The ``target_accepts`` arms exist to rule out an alternative explanation for
    the whole of Finding 9. The depth cap works by making warmup conclude a
    LARGER step size -- but "adapt to a larger step size, run shallower trees,
    accept more divergences" is precisely what ``target_accept_rate`` does, and
    ``run_nuts`` has taken that parameter all along. If lowering the target
    reproduces the cap's step size, tree depth, speed and divergence count, then
    ``warmup_max_num_doublings`` is an indirect re-parameterization of a knob
    that already ships, and the honest finding is that nobody had tuned
    ``target_accept_rate``.

    0.65 and 0.6 are the natural points: 0.651 is ChEES's own default -- and
    deliberately not NUTS's 0.8 -- and 0.6 is the classic NUTS floor.
    """
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
    arms = {"control": dict(shipped)}
    if warmup_cap is not None:
        arms[f"wcap={warmup_cap}"] = dict(shipped, warmup_max_num_doublings=int(warmup_cap))
    for t in target_accepts:
        arms[f"target={t:g}"] = dict(shipped, target_accept_rate=float(t))
    # The recommended arm, and the one whose marginals had never been checked.
    # Every OTHER fast configuration measured on this fixture displaces
    # `sfh_dpl_beta` by the same amount -- wcap=5 at -0.230 sd / -3.08 MCSE and
    # target_accept_rate=0.65 at -0.225 sd / -3.07 MCSE, two mechanistically
    # distinct routes to a larger adapted step size agreeing to three
    # significant figures. That is a strong prior that ANY configuration
    # reaching a larger step will move it, and `precondition=0.5` reaches the
    # largest step of all (0.0387 against the control's 0.0050, ~7.7x). So the
    # honest expectation is a shift, and a report recommending this arm has to
    # say which way it fell.
    for a in preconditions:
        arms[f"precond={a:g}"] = dict(shipped, precondition=float(a))

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
        # ONE call over the whole samples dict, which is what the helper takes
        # (name -> array); it returns name -> {'ess': ..., 'tau_*': ...} and
        # silently drops parameters that never moved or are not 1-D. Calling it
        # per array raises, because an ndarray has no .items().
        ess_all = effective_sample_size({k: np.asarray(v) for k, v in post.samples.items()})
        stats = {}
        for name, draws in post.samples.items():
            arr = np.asarray(draws).ravel()
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            if name not in ess_all:
                continue
            ess = float(ess_all[name]["ess"])
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

    # Every non-control arm is compared against the control, so one process can
    # answer "is the cap a re-parameterization of the target" and "does either
    # move the posterior" with the same instrument.
    ctrl = out["control"]["params"]
    comparison = {}
    for label in [k for k in out if k != "control"]:
        # Never let a bug in the comparison destroy the arms. Each arm above is
        # a full NUTS fit costing minutes to hours; the comparison is
        # arithmetic. An earlier revision lost two completed fits to an
        # AttributeError raised after both had run, which is the expensive way
        # to find a one-line mistake.
        try:
            comparison[label] = _compare(ctrl, out[label]["params"])
        except Exception as exc:  # the arms matter; this arithmetic does not
            comparison[label] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "probe": "warmup_cap_posterior",
        "notebook": notebook,
        "seed": seed,
        "warmup_cap": warmup_cap,
        "target_accepts": list(target_accepts),
        "preconditions": list(preconditions),
        "device": str(jax.devices()[0].platform),
        **_machine_load(),
        "arms": out,
        "comparison": comparison,
    }


def _compare(ctrl: dict, other: dict) -> dict:
    """Per-parameter marginal agreement of ``other`` against ``ctrl``."""
    import numpy as np

    result = {}
    for name in sorted(set(ctrl) & set(other)):
        c, w = ctrl[name], other[name]
        # Combined MC error of the DIFFERENCE of two means. If |shift| exceeds a
        # few of these the two runs are not sampling the same distribution.
        joint = float(np.sqrt(c["mcse"] ** 2 + w["mcse"] ** 2))
        result[name] = {
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
    return result


def main(argv=None) -> int:
    from benchmark_notebook_sampler import NOTEBOOKS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default="ctl-dpl", choices=sorted(NOTEBOOKS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--warmup-cap",
        type=int,
        default=5,
        help="warmup tree-depth cap arm; pass -1 to omit the arm entirely",
    )
    ap.add_argument(
        "--precondition",
        type=float,
        nargs="*",
        default=[],
        metavar="ALPHA",
        help="add an analytic-metric arm per whitening strength. This is the "
        "configuration the report recommends, so its marginals must be checked "
        "against the control like any other fast arm.",
    )
    ap.add_argument(
        "--target-accept",
        type=float,
        nargs="*",
        default=[],
        metavar="RATE",
        help="add a lowered target_accept_rate arm per value. This is the control "
        "for Finding 9's mechanism: if lowering the target reproduces the depth "
        "cap's step size, depth, speed and divergences, the cap is an indirect "
        "re-parameterization of a knob that already ships.",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    row = run(
        args.notebook,
        args.seed,
        None if args.warmup_cap is not None and args.warmup_cap < 0 else args.warmup_cap,
        tuple(args.target_accept),
        tuple(args.precondition),
    )
    # Write BEFORE printing: a formatting error in the table below must not lose
    # hours of sampling either.
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(row, fh, indent=2)

    print(f"\n{args.notebook} seed {args.seed}")
    # The adapted step size is the quantity the mechanism is about, and unlike
    # every wall clock here it is contention-immune. It leads the table.
    hdr = (
        f"{'arm':<14}{'step size':>12}{'depth':>8}{'grad/draw':>11}"
        f"{'div(samp)':>11}{'div(warm)':>11}{'wall s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, arm in row["arms"].items():
        wf = arm["warmup_divergence_frac_final_window"]
        print(
            f"{label:<14}{arm['step_size']:>12.5f}{arm['tree_depth_mean']:>8.2f}"
            f"{2 ** arm['tree_depth_mean']:>11.1f}{arm['n_divergent_sampling']:>11}"
            f"{('n/a' if wf is None else format(wf, '.3f')):>11}{arm['wall_s']:>10.1f}"
        )
    for label, comp in row["comparison"].items():
        cols = f"{'parameter':<26}{'shift/sd':>10}{'sd ratio':>10}{'shift/MCSE':>12}"
        hdr2 = f"\n{label} vs control:  {cols}"
        print(hdr2)
        for name, c in comp.items():
            print(
                f"{'':<{len(label) + 16}}{name:<26}{c['shift_in_control_sd']:>10.3f}"
                f"{c['sd_ratio_cap_over_control']:>10.3f}{c['shift_in_joint_mcse']:>12.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
