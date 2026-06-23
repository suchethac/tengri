#!/usr/bin/env python
"""Throwaway gallery execution harness (Phase 1 of the gallery-refresh plan).

Runs every ``examples/**/plot_*.py`` as an isolated subprocess against the
current worktree's tengri, captures pass/error/skip + traceback tail, and
writes a JSON report. NOT meant to be committed.

Usage::

    JAX_PLATFORMS=cpu PYTHONPATH=$WT/src:$WT \
        /path/to/.venv/bin/python tools/run_gallery.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Mirror docs/conf.py ignore_pattern: heavy NUTS/VI/population/benchmark scripts
# that OOM or are too slow for an execution sweep. Matched against basename.
IGNORE = re.compile(
    r"plot_("
    r"population_scaling|"
    r"hierarchical|hierarchical_convergence|"
    r"prior_posterior_compare|"
    r"wrong_model_trap|"
    r"posterior_corner_dpl|"
    r"joint_photometry_line_fit|"
    r"galaxy_stack_1000|"
    r"vmap_population_throughput|"
    r"stochastic_sfh_ift_recovery|"
    r"waveprecomp_speedup|"
    r"real_data_fit|"
    r"dust_attenuation_recovery|"
    r"jit_cache_speedup|"
    r"waveprecomp_scaling"
    r")\.py$"
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


def run_one(path_str: str, timeout: int) -> dict:
    path = Path(path_str)
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env["MPLBACKEND"] = "Agg"  # never pop a window
    env["TENGRI_GALLERY_HARNESS"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"path": str(path.relative_to(REPO)), "status": "timeout",
                "secs": round(time.time() - t0, 1), "tail": ""}
    dt = round(time.time() - t0, 1)
    rel = str(path.relative_to(REPO))
    if proc.returncode == 0:
        return {"path": rel, "status": "pass", "secs": dt, "tail": ""}
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    # last exception line is the most informative
    exc_line = next((ln for ln in reversed(tail)
                     if re.match(r"^\w+(\.\w+)*(Error|Exception|Warning):", ln)), "")
    return {
        "path": rel,
        "status": "error",
        "secs": dt,
        "exc": exc_line,
        "tail": "\n".join(tail[-25:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default=str(REPO / "tools" / "gallery_report.json"))
    ap.add_argument("--filter", default="", help="only run paths containing this substring")
    args = ap.parse_args()

    scripts = sorted(
        p for p in EXAMPLES.rglob("plot_*.py")
        if not IGNORE.search(p.name) and (not args.filter or args.filter in str(p))
    )
    skipped = sorted(
        str(p.relative_to(REPO)) for p in EXAMPLES.rglob("plot_*.py")
        if IGNORE.search(p.name)
    )
    print(f"discovered {len(scripts)} runnable, {len(skipped)} ignore_pattern-skipped",
          flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, str(p), args.timeout): p for p in scripts}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            mark = {"pass": "ok", "error": "ERR", "timeout": "T/O"}.get(r["status"], "?")
            extra = f"  {r.get('exc', '')}" if r["status"] != "pass" else ""
            print(f"[{done}/{len(scripts)}] {mark} {r['path']} ({r['secs']}s){extra}",
                  flush=True)

    results.sort(key=lambda r: (r["status"] != "error", r["path"]))
    report = {
        "n_runnable": len(scripts),
        "n_pass": sum(r["status"] == "pass" for r in results),
        "n_error": sum(r["status"] == "error" for r in results),
        "n_timeout": sum(r["status"] == "timeout" for r in results),
        "skipped_ignore_pattern": skipped,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n== pass={report['n_pass']} error={report['n_error']} "
          f"timeout={report['n_timeout']} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
