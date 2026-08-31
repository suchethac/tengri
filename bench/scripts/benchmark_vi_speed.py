#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Cold and warm wall clock for the optimisation-based backends, against a NUTS reference.

Why cold and warm are reported separately, always
=================================================

``bench/reports/2026-08-30_mclmc_tuning.md`` measured that **75% of a NUTS fit on
a healthy fixture is XLA compilation** (189.4 s cold, 46.8 s warm), and that
MCLMC's structural win over NUTS is a 14x cheaper *graph*, not a faster clock.
Every backend in this file is shorter still, so the same question is sharper: a
method whose advantage is a shorter scan behind the same compile is not faster in
practice, and a single "wall time" column cannot tell the two apart. So the child
calls ``fit`` **twice in one process** with the same key, and the row carries
``cold``, ``warm`` and ``compile = cold - warm``.

That subtraction is the standard this repo already uses and it is a *lower bound*
on compile: anything cached between the two calls (an adaptation result, a model
cache entry) also lands in the difference. It is stated as "cold minus warm", not
as "XLA time".

Why the fidelity columns are not optional
=========================================

Every method here returns i.i.d. draws from a fitted Gaussian, so an effective
sample size computed from its output equals ``n_samples`` by construction and an
R-hat over its chains is 1.000 by construction. **Neither diagnostic can fail**,
which means neither is evidence. The only check available is agreement with a
sampler that does have diagnostics, so each row is scored against a reference
posterior on the same mock:

* ``z`` -- ``(mean_method - mean_ref)`` over the two estimates' own Monte-Carlo
  errors added in quadrature. The reference's MCSE uses its **ESS**, not its draw
  count; the VI methods' uses their draw count, which for i.i.d. draws is right.
* ``sd ratio`` -- ``sd_method / sd_ref``. VI families characteristically
  under-disperse, and a mean-field Gaussian on a tilted posterior reports the
  *conditional* width rather than the marginal one, so this column is the one
  that decides whether a fast row is usable.

The reference row is published with its own **min ESS beside its R-hat**, because
split-R-hat over two equally badly-mixed halves of one chain reads ~1.00
(``bench/reports/2026-08-31_catalog_preconditioning.md``).

Usage
=====

::

    # 1. the reference posterior (long, one per fixture, cached to JSON)
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_speed.py \\
        --notebook ctl-dpl --reference

    # 2. the speed table (one fit per subprocess, cold + warm in each)
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_speed.py \\
        --notebook ctl-dpl --seeds 3

Fixtures come from :data:`benchmark_notebook_sampler.NOTEBOOKS` so this file
declares no model of its own and cannot drift from the sampler campaign's
(``tools/check_harness_parity.py`` gates that registry, and a fixture added
without a ``parity=`` block fails it).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

import tengri  # noqa: E402
from benchmark_notebook_sampler import NOTEBOOKS  # noqa: E402
from tengri import Data, ForwardModel, generate_mock  # noqa: E402
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: Rows measured by default. ``precondition`` is threaded through the registry
#: capability, so a row asking for it on a backend that does not declare it is
#: refused by ``check_capabilities`` rather than silently dropped.
ROWS: dict[str, dict] = {
    "map": dict(method="map"),
    "laplace": dict(method="laplace"),
    "pathfinder": dict(method="pathfinder"),
    "pathfinder+precond": dict(method="pathfinder", precondition=True),
    # BlackJAX's own Pathfinder page warns that "bad initialization points may
    # improve L-BFGS convergence and Hessian estimation", which argues against the
    # MAP seed every other row here uses. ``map_seed=False`` starts from the
    # backend's own default init instead, so the claim is measured rather than
    # assumed either way.
    "pathfinder cold-init": dict(method="pathfinder", map_seed=False),
    "vi_meanfield": dict(method="vi_meanfield"),
    "vi_meanfield+precond": dict(method="vi_meanfield", precondition=True),
    "vi_fullrank": dict(method="vi_fullrank"),
    "vi_fullrank+precond": dict(method="vi_fullrank", precondition=True),
}

#: The Pathfinder-warm-start rows. Separate from :data:`ROWS` because they answer
#: a different question: not "is an approximate posterior faster than a sampler"
#: but "does swapping ``blackjax.window_adaptation`` for
#: ``blackjax.adaptation.pathfinder_adaptation`` buy back any of the warmup, which
#: is where a NUTS fit spends its wall clock".
#:
#: ``bench/reports/2026-04-22_pathfinder_vs_window_nuts.md`` measured this once, at
#: D=8, and found window adaptation ahead -- closing with the caveat that
#: Pathfinder "only pays off at high D (untested here)". Every measurement in this
#: project since has been D=3-9, so the caveat has never been tested and the D=8
#: verdict has never been re-measured on the current code. These rows re-measure
#: it, with the matched-warmup control the original report showed is mandatory: at
#: ``n_warmup=50`` its Pathfinder arm went **18x slower**, silently, because a
#: noisy inverse mass matrix drove NUTS to depth-10 trees. A warm-start row
#: without a matched-budget row beside it cannot tell a saving from that failure.
WARMSTART_ROWS: dict[str, dict] = {
    "nuts window 600": dict(method="mcmc_nuts", n_warmup=600, n_samples=600),
    "nuts pathfinder 600": dict(
        method="mcmc_nuts", n_warmup=600, n_samples=600, pathfinder_warmstart=True
    ),
    "nuts pathfinder 200": dict(
        method="mcmc_nuts", n_warmup=200, n_samples=600, pathfinder_warmstart=True
    ),
    "nuts window 600 +precond": dict(
        method="mcmc_nuts", n_warmup=600, n_samples=600, precondition=True
    ),
    "nuts pathfinder 600 +precond": dict(
        method="mcmc_nuts",
        n_warmup=600,
        n_samples=600,
        pathfinder_warmstart=True,
        precondition=True,
    ),
}

ROWS.update(WARMSTART_ROWS)


def build_fixture(nb: str, seed: int):
    """Fixture model, mock data and a MAP seed, exactly as the sampler harness builds them."""
    cfg = NOTEBOOKS[nb]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    return cfg, forward, data, key_fit, mock


def summarize(posterior, free_names) -> dict:
    """Per-parameter mean, sd, quantiles and MCSE. The only fidelity currency here."""
    out = {}
    samples = posterior.samples or {}
    ess = effective_sample_size({k: np.asarray(v) for k, v in samples.items() if k in free_names})
    for name in free_names:
        if name not in samples:
            continue
        v = np.asarray(samples[name]).ravel()
        n_eff = float(ess.get(name, {}).get("ess", np.nan))
        if not np.isfinite(n_eff) or n_eff <= 0:
            n_eff = float(v.size)
        sd = float(np.std(v, ddof=1))
        out[name] = {
            "mean": float(np.mean(v)),
            "sd": sd,
            "q16": float(np.quantile(v, 0.16)),
            "q50": float(np.quantile(v, 0.50)),
            "q84": float(np.quantile(v, 0.84)),
            "n_draws": int(v.size),
            "ess": n_eff,
            # MCSE on the MEAN. For i.i.d. VI draws ess == n and this is sd/sqrt(n);
            # for a chain it is sd/sqrt(ESS), which is the number that makes a
            # z-score against a correlated reference honest.
            "mcse": sd / np.sqrt(max(n_eff, 1.0)),
        }
    return out


def rss_gb() -> float:
    """Peak RSS of this process, in GB (Linux reports kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def child_row(nb: str, label: str, seed: int, out_path: str) -> None:
    """One row: build, MAP-seed, fit twice in this process, write JSON."""
    kwargs = dict(ROWS[label])
    use_map_seed = kwargs.pop("map_seed", True)
    cfg, forward, data, key_fit, _ = build_fixture(nb, seed)
    free_names = tuple(forward.spec.free_params)

    map_seed = (
        forward.fit(data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False)
        if use_map_seed
        else None
    )

    common = dict(key=key_fit, init_from=map_seed, verbose=False, **kwargs)
    if kwargs.get("method", "").startswith("mcmc"):
        common["n_chains"] = cfg["n_chains"]
    t0 = time.perf_counter()
    posterior = forward.fit(data, **common)
    cold = time.perf_counter() - t0

    t1 = time.perf_counter()
    forward.fit(data, **common)
    warm = time.perf_counter() - t1

    diag = dict(posterior.diagnostics or {})
    diag.pop("elbo_history", None)
    rhats = posterior.rhat() if posterior.samples else {}
    row = {
        "label": label,
        "notebook": nb,
        "seed": seed,
        "cold_s": cold,
        "warm_s": warm,
        "compile_s": cold - warm,
        "compile_frac": (cold - warm) / cold if cold > 0 else float("nan"),
        "rss_gb": rss_gb(),
        "free_params": list(free_names),
        "n_logdensity_evals": diag.get("n_logdensity_evals"),
        # Present only for the sampler rows. An R-hat over i.i.d. Gaussian draws
        # is 1.000 by construction, so the key is absent rather than filled with a
        # number that cannot fail -- see the module docstring.
        "max_rhat": (max(float(v) for v in rhats.values()) if rhats else None),
        "n_divergent": (None if "n_divergent" not in diag else int(diag["n_divergent"] or 0)),
        "tree_depth_mean": diag.get("tree_depth_mean"),
        "frac_max_depth": diag.get("frac_max_depth"),
        "step_size": diag.get("step_size"),
        "diagnostics": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
            for k, v in diag.items()
            if not isinstance(v, (list, dict, tuple, np.ndarray))
        },
        "summary": summarize(posterior, free_names) if posterior.samples else None,
        "params": {k: float(v) for k, v in (posterior.params or {}).items() if k in free_names},
    }
    Path(out_path).write_text(json.dumps(row))


def child_reference(nb: str, seed: int, out_path: str, n_warmup: int, n_samples: int) -> None:
    """The reference posterior: preconditioned NUTS, long, with R-hat and min ESS attached."""
    cfg, forward, data, key_fit, mock = build_fixture(nb, seed)
    free_names = tuple(forward.spec.free_params)
    map_seed = forward.fit(
        data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
    )
    t0 = time.perf_counter()
    posterior = forward.fit(
        data,
        key=key_fit,
        init_from=map_seed,
        method="mcmc_nuts",
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=cfg["n_chains"],
        precondition=True,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    summary = summarize(posterior, free_names)
    rhats = posterior.rhat()
    diag = posterior.diagnostics or {}
    Path(out_path).write_text(
        json.dumps(
            {
                "label": "reference: mcmc_nuts + precondition",
                "notebook": nb,
                "seed": seed,
                "wall_s": wall,
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "n_chains": cfg["n_chains"],
                "max_rhat": max(float(v) for v in rhats.values()) if rhats else float("nan"),
                "rhat": {k: float(v) for k, v in rhats.items()},
                "min_ess": min(v["ess"] for v in summary.values()) if summary else float("nan"),
                "n_divergent": int(diag.get("n_divergent") or 0),
                "tree_depth_mean": diag.get("tree_depth_mean"),
                "free_params": list(free_names),
                "summary": summary,
                "truth": {
                    k: float(v) for k, v in (mock.get("truth") or {}).items() if k in free_names
                },
            }
        )
    )


def _spawn(args_list: list[str], timeout: int) -> tuple[int, str]:
    """Run a child and report HOW it died, not just that it produced nothing.

    ``scripts/validate_backends_231.py`` recorded every childless death as
    ``SegfaultOrAbort`` without ever reading the return code, and that guess is
    what quarantined ``pathfinder`` for three months. A negative return code is a
    signal; a positive one is an exception. They are different bugs.
    """
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    try:
        p = subprocess.run(args_list, env=env, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    if p.returncode < 0:
        return p.returncode, f"KILLED by signal {-p.returncode}"
    tail = (p.stderr.strip().splitlines() or [""])[-1][:200]
    return p.returncode, tail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", choices=sorted(NOTEBOOKS), required=True)
    ap.add_argument("--only", default=None, help="comma-separated subset of the row labels")
    ap.add_argument("--seeds", type=int, default=1, help="consecutive seeds from the fixture's own")
    ap.add_argument("--reference", action="store_true", help="run the NUTS reference instead")
    ap.add_argument("--ref-warmup", type=int, default=1000)
    ap.add_argument("--ref-samples", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tag", default=None, help="suffix for the results JSON")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    base_seed = NOTEBOOKS[args.notebook]["seed"]
    tag = args.tag or args.notebook

    if args.reference:
        out = RESULTS / f"2026-08-31_vi_speed_reference_{tag}.json"
        rc, msg = _spawn(
            [
                sys.executable,
                __file__,
                "--child-reference",
                args.notebook,
                str(base_seed),
                str(out),
                str(args.ref_warmup),
                str(args.ref_samples),
            ],
            args.timeout,
        )
        print(f"reference rc={rc} {msg}")
        if out.exists():
            r = json.loads(out.read_text())
            print(
                f"  wall {r['wall_s']:.1f}s  max R-hat {r['max_rhat']:.4f}  "
                f"min ESS {r['min_ess']:.1f}  divergences {r['n_divergent']}"
            )
        return

    labels = [x.strip() for x in args.only.split(",")] if args.only else list(ROWS)
    unknown = [x for x in labels if x not in ROWS]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; known: {sorted(ROWS)}")

    out_json = RESULTS / f"2026-08-31_vi_speed_{tag}.json"
    rows = json.loads(out_json.read_text()) if out_json.exists() else []

    # flush=True on every row: a campaign is watched while it runs, and Python
    # buffers stdout to a pipe, so an unflushed table appears only at exit.
    print(
        f"{'row':<24}{'cold':>9}{'warm':>9}{'compile':>9}{'share':>8}{'RSS':>7}  notes",
        flush=True,
    )
    for i in range(args.seeds):
        seed = base_seed + i
        for label in labels:
            tmp = RESULTS / f".tmp_vi_{tag}_{label.replace('+', '_')}_{seed}.json"
            if tmp.exists():
                tmp.unlink()
            rc, msg = _spawn(
                [sys.executable, __file__, "--child", args.notebook, label, str(seed), str(tmp)],
                args.timeout,
            )
            if not tmp.exists():
                print(
                    f"{label:<24}{'--':>9}{'--':>9}{'--':>9}{'--':>8}{'--':>7}  rc={rc} {msg}",
                    flush=True,
                )
                rows.append({"label": label, "seed": seed, "status": "no_output", "rc": rc,
                             "detail": msg, "notebook": args.notebook})
                continue
            r = json.loads(tmp.read_text())
            tmp.unlink()
            rows.append(r)
            print(
                f"{label:<24}{r['cold_s']:>9.1f}{r['warm_s']:>9.1f}{r['compile_s']:>9.1f}"
                f"{100 * r['compile_frac']:>7.0f}%{r['rss_gb']:>7.1f}  seed {seed}",
                flush=True,
            )
            out_json.write_text(json.dumps(rows, indent=1))

    out_json.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_row(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) > 1 and sys.argv[1] == "--child-reference":
        child_reference(
            sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
        )
    else:
        main()
