# SPDX-License-Identifier: BSD-3-Clause
"""Execute the sphinx-gallery examples and fail on any error.

CI does not run the gallery, and that is not an oversight anyone made on
purpose — it falls out of how ``docs/conf.py`` builds ``_DO_NOT_EXECUTE``::

    _DO_NOT_EXECUTE = sorted(
        {
            p.stem.replace("sphx_glr_", "").rsplit("_", 1)[0]
            for p in _AUTO_EXAMPLES.glob("*/images/sphx_glr_plot_*_001.png")
        }
    )

The skip list is derived **from disk**: sphinx-gallery skips any example that
already has a committed figure. Every example has one. So the docs build
executes none of them — and ``docs.yml`` only runs on ``push: [main]`` anyway,
never on a pull request.

The consequence is not hypothetical. During #1145 an agent migrated five
examples to ``model.predict(p, wave=...)``, a keyword argument that does not
exist. They raise ``TypeError``. ``py_compile`` passes (it is a runtime error),
``ruff`` passes, the docs build never runs them, and the gallery would have
rendered the broken source next to a stale figure. They were caught only by
executing all of them by hand.

This script is that hand-execution, made permanent. It runs every example in a
SINGLE process — which is the whole reason it is affordable. Per-example cost is
~1.8 s warm; spawning a process each would add ~15-20 s of JAX import and
compile-cache load, turning 8 minutes into 40.

Skips are explicit and printed. A gate that quietly drops a third of its inputs
reads as "covered" when it is not.

Usage
-----
    python tools/run_gallery_examples.py                 # all runnable examples
    python tools/run_gallery_examples.py --list          # show what runs / skips
    python tools/run_gallery_examples.py --only agn/     # a subdirectory
    python tools/run_gallery_examples.py --changed-since origin/main
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import runpy
import subprocess
import sys
import time
import traceback
import warnings

REPO = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

# Examples that run inference. NUTS warmup peaks at 20+ GB (see
# docs/dev/notebook_orchestration_oom.md); they cannot run on a CI box, and that
# is exactly why their figures are the ones legitimately committed.
_RUNS_A_FIT = re.compile(
    r"""\.fit\(|Fitter\(|\.run\(\s*["'](?:nuts|vi|vi_native|mcmc|map|nss|pathfinder)["']""",
)


def _classify(path: pathlib.Path) -> str | None:
    """Return a skip reason, or None if the example should run."""
    src = path.read_text()
    if _RUNS_A_FIT.search(src):
        return "runs inference (20+ GB warmup)"
    return None


def _discover() -> tuple[list[pathlib.Path], list[tuple[pathlib.Path, str]]]:
    run, skip = [], []
    for path in sorted(EXAMPLES.rglob("plot_*.py")):
        reason = _classify(path)
        (skip.append((path, reason)) if reason else run.append(path))
    return run, skip


def _changed_since(ref: str) -> set[pathlib.Path]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD", "--", "examples/"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {REPO / line for line in out.split() if line.endswith(".py")}


def _run_one(path: pathlib.Path) -> tuple[bool, str]:
    """Execute one example in-process. Returns (ok, message)."""
    import matplotlib.pyplot as plt

    cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        with warnings.catch_warnings():
            # A DeprecationWarning from tengri means the example teaches an API we
            # are removing. That is a failure, not a nuisance.
            warnings.simplefilter("error", DeprecationWarning)
            warnings.filterwarnings("ignore", message=r".*trapz.*")  # numpy's own
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"jax.*")
            runpy.run_path(path.name, run_name="__main__")
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=6)
    finally:
        os.chdir(cwd)
        plt.close("all")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="show what runs and what skips, then exit")
    ap.add_argument("--only", default="", help="substring filter on the example path")
    ap.add_argument("--changed-since", default="", help="only examples changed vs this git ref")
    ap.add_argument("--shard", default="", metavar="I/N", help="run shard I of N (1-indexed)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    run, skip = _discover()

    if args.only:
        run = [p for p in run if args.only in str(p.relative_to(REPO))]
    if args.changed_since:
        changed = _changed_since(args.changed_since)
        run = [p for p in run if p in changed]
        print(f"restricting to examples changed vs {args.changed_since}: {len(run)}")
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= n:
            raise SystemExit(f"--shard {args.shard}: I must be in 1..N")
        # INTERLEAVED, not contiguous. Cost is wildly uneven and correlates with
        # the path: the Cue/nebular examples run ~10-20 s each and sort together,
        # so a block split would hand one shard every slow example and time out
        # while its siblings idle. Striding spreads them evenly.
        total = len(run)
        run = run[i - 1 :: n]
        print(f"shard {i}/{n}: {len(run)} of {total} examples (interleaved)")

    # Never silently drop coverage: say what is not being run, and why.
    print(f"gallery: {len(run)} to execute, {len(skip)} skipped")
    for path, reason in skip:
        print(f"  SKIP  {path.relative_to(REPO)}  — {reason}")
    if args.list:
        for path in run:
            print(f"  RUN   {path.relative_to(REPO)}")
        return 0
    if not run:
        print("nothing to execute")
        return 0

    print(flush=True)
    failures: list[tuple[pathlib.Path, str]] = []
    t0 = time.perf_counter()
    for i, path in enumerate(run, 1):
        rel = path.relative_to(REPO)
        t = time.perf_counter()
        ok, msg = _run_one(path)
        dt = time.perf_counter() - t
        print(f"  [{i:3d}/{len(run)}] {'ok  ' if ok else 'FAIL'} {dt:5.1f}s  {rel}", flush=True)
        if not ok:
            failures.append((rel, msg))

    total = time.perf_counter() - t0
    print(f"\ngallery: {len(run) - len(failures)}/{len(run)} passed in {total / 60:.1f} min")

    if failures:
        print(f"\n{'=' * 70}\n{len(failures)} example(s) FAILED\n{'=' * 70}")
        for rel, msg in failures:
            print(f"\n--- {rel} ---\n{msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
