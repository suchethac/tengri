"""Full-table VI scaling: powers-of-2 N × K-batch sweep, with convergence.

Runs MGVI (native_vi_linear) and geoVI (native_vi_nonlinear) on a 2D grid:
  N ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
  K ∈ {1, 2, 4, 8}   (forward_chunk_size)

Each (method, N, K) runs in a fresh subprocess for clean ΔRSS. Convergence
is verified via PopulationPosterior.diagnostics["n_iterations"]:
  * iter cap = 50 — high enough that the engines' kl_rtol=1e-2 early-stop
    is the regular exit. CONVERGED == iters_used < cap.
  * If a row hits the cap, retry once with cap=100.

Memory budget: <= 30 GB peak per worker. Per (method, K) column, once a
row exceeds budget the rest of that column is skipped (memory grows with
N for some configs; aborting saves time).

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py
    JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py --linear-only
    JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py --geovi-only
    JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py --ks 1,2,4
"""

from __future__ import annotations

import argparse
import json
import math as _math
import os
import sys
from pathlib import Path

from benchmark_population_native import spawn

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "vi_scaling_benchmark.json"
RESULTS_PATH_RICH = Path(__file__).resolve().parent.parent / "data" / "vi_scaling_benchmark_rich.json"
RESULTS_PATH_SPEC = Path(__file__).resolve().parent.parent / "data" / "vi_scaling_benchmark_spec.json"
RESULTS_PATH_JOINT = Path(__file__).resolve().parent.parent / "data" / "vi_scaling_benchmark_joint.json"

ALL_NS: tuple[int, ...] = (
    4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
)
DEFAULT_KS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
N_ITER_CAP = 50
N_ITER_RETRY_CAP = 100
N_SAMP = 6                # ×2 mirrored = 12 effective samples / iter
TIMEOUT = int(os.environ.get("VI_BENCHMARK_TIMEOUT", "2400"))
MEM_BUDGET_GB = 30.0


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'N':>5}  {'K':>2}  {'cold (s)':>9}  {'warm (s)':>9}"
        f"  {'compile~(s)':>12}  {'ΔRSS (GB)':>10}  {'iters':>9}  {'conv?':>5}  {'note':<14}"
    )
    print("  " + "-" * 96)


def _fmt_iters(row: dict) -> str:
    used = row.get("n_iters_used_warm", -1)
    cap = row.get("n_iters_max", -1)
    if used < 0 or cap < 0:
        return "  ?  / ?"
    return f"{used:>3}/{cap:<3}"


def _print_row(n: int, k: int, row: dict, note: str = "") -> None:
    err = row.get("error", "")
    if err:
        print(f"  {n:>5}  {k:>2}  ERROR: {err[:80]}")
        return
    wall = row.get("wall_s", -1)
    warm = row.get("wall_s_warm", -1)
    comp = row.get("compile_s_approx", -1)
    delta = row.get("rss_delta_gb", -1)
    iters = _fmt_iters(row)
    conv = "YES" if row.get("converged") else "NO"
    flag = "OOM-RISK" if delta > MEM_BUDGET_GB else note
    print(
        f"  {n:>5}  {k:>2}  {wall:>9.1f}  {warm:>9.1f}"
        f"  {comp:>12.1f}  {delta:>10.2f}  {iters:>9}  {conv:>5}  {flag:<14}"
    )


def _run_with_convergence(n: int, method: str, k: int, rich_obs: bool = False,
                          noise_frac: float = 0.10, spec_obs: bool = False,
                          joint_obs: bool = False) -> dict:
    """Run once at N_ITER_CAP; if it hit the cap, retry at N_ITER_RETRY_CAP."""
    row = spawn(n, method, N_ITER_CAP, N_SAMP, forward_chunk_size=k,
                compile_timeout=TIMEOUT, rich_obs=rich_obs, noise_frac=noise_frac,
                spec_obs=spec_obs, joint_obs=joint_obs)
    if row.get("error"):
        return row
    if not row.get("converged"):
        print(f"    (N={n} K={k} {method} did not converge at cap={N_ITER_CAP}; "
              f"retrying with cap={N_ITER_RETRY_CAP})", flush=True)
        row2 = spawn(n, method, N_ITER_RETRY_CAP, N_SAMP, forward_chunk_size=k,
                     compile_timeout=TIMEOUT * 2, rich_obs=rich_obs,
                     noise_frac=noise_frac, spec_obs=spec_obs,
                     joint_obs=joint_obs)
        if not row2.get("error"):
            return row2
    return row


_active_path: Path = RESULTS_PATH


def _save_rows(rows: list[dict]) -> None:
    """Write incremental results to the active JSON path."""
    _active_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _active_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, _active_path)


def _load_rows() -> list[dict]:
    """Load existing rows for idempotent resume from the active path."""
    if not _active_path.exists():
        return []
    try:
        with _active_path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _have_cell(rows: list[dict], method: str, n: int, k: int) -> dict | None:
    """Return existing converged row for this (method, N, K) or None."""
    for r in rows:
        if (r.get("method") == method
                and r.get("n_gal") == n
                and r.get("forward_chunk_size") == k
                and not r.get("error")
                and r.get("converged", False)):
            return r
    return None


def _drop_cell(rows: list[dict], method: str, n: int, k: int) -> None:
    """Remove any existing entry for (method, N, K) so a forced run replaces it."""
    keep = [
        r for r in rows
        if not (r.get("method") == method
                and r.get("n_gal") == n
                and r.get("forward_chunk_size") == k)
    ]
    rows.clear()
    rows.extend(keep)


def run_method(method: str, ns: tuple[int, ...], ks: tuple[int, ...], label: str,
               all_rows: list[dict], force: bool = False,
               rich_obs: bool = False, noise_frac: float = 0.10,
               spec_obs: bool = False, joint_obs: bool = False) -> None:
    """Sweep (K outer, N inner). One column per K so memory aborts are local."""
    for k in ks:
        _print_header(
            f"{label} ({method}) — K={k}, N={list(ns)}, "
            f"n_samples={N_SAMP} (×2 mirrored), iter cap={N_ITER_CAP}"
        )
        for n in ns:
            if n % k != 0:
                print(f"  {n:>5}  {k:>2}  (skipped: {n} % {k} != 0)")
                continue
            if force:
                _drop_cell(all_rows, method, n, k)
            else:
                cached = _have_cell(all_rows, method, n, k)
                if cached is not None:
                    _print_row(n, k, cached, note="cached")
                    continue
            print(f"  Running N={n}/K={k} {method}...", flush=True)
            row = _run_with_convergence(n, method, k, rich_obs=rich_obs,
                                        noise_frac=noise_frac, spec_obs=spec_obs,
                                        joint_obs=joint_obs)
            row["method"] = method
            all_rows.append(row)
            _save_rows(all_rows)
            delta = row.get("rss_delta_gb", -1)
            err = row.get("error", "")
            if delta > MEM_BUDGET_GB:
                print(f"    -> ΔRSS {delta:.1f} GB > {MEM_BUDGET_GB} GB; "
                      f"aborting K={k} column at N>={n}.", flush=True)
                _print_row(n, k, row)
                break
            if err:
                # Worker died (TIMEOUT, OOM-kill, segfault, ...). Abort the K
                # column so we don't trigger the same crash at larger N.
                print(f"    -> worker error; aborting K={k} column at N>={n}.",
                      flush=True)
                _print_row(n, k, row)
                break
            _print_row(n, k, row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-only", action="store_true")
    ap.add_argument("--geovi-only", action="store_true")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                    help="Comma-separated K values (default: power-of-2 sweep).")
    ap.add_argument("--snap-ks", type=int, default=None, metavar="BUCKET",
                    help="Snap each K to the nearest power of BUCKET (e.g. "
                         "--snap-ks 4 collapses {1,2,4,8} → {1,4,4,4} so the "
                         "compile cache is shared across K values whose "
                         "scientific signal you don't care about). Off by "
                         "default — persistent cache amortizes most repeat "
                         "compile cost without losing K=2 / K=8 information.")
    ap.add_argument("--ns", default=None,
                    help="Comma-separated N override.")
    ap.add_argument("--force", action="store_true",
                    help="Rerun all selected (method, N, K) cells, overwriting "
                         "any existing JSON entries for them.")
    ap.add_argument("--rich-obs", action="store_true",
                    help="Use 10-band photometry (FUV/NUV+SDSS+JHKs) instead "
                         "of SDSS-only. Writes to vi_scaling_benchmark_rich.json.")
    ap.add_argument("--spec-obs", action="store_true",
                    help="Use spectroscopy (3000–7500 Å rest, R≈500) covering "
                         "Hα/Hβ/[OIII]/4000Å break. Writes to "
                         "vi_scaling_benchmark_spec.json. Mutually exclusive "
                         "with --rich-obs (spec wins).")
    ap.add_argument("--joint-obs", action="store_true",
                    help="Joint rich photometry + emission-line luminosities "
                         "(Hα, Hβ, [OIII]_5007, [OII]_3727). Writes to "
                         "vi_scaling_benchmark_joint.json. Implies --rich-obs; "
                         "mutually exclusive with --spec-obs.")
    ap.add_argument("--noise-frac", type=float, default=0.10,
                    help="Fractional photometric/spectroscopic noise (default 0.10).")
    args = ap.parse_args()
    global _active_path
    if args.spec_obs:
        _active_path = RESULTS_PATH_SPEC
    elif args.joint_obs:
        _active_path = RESULTS_PATH_JOINT
    elif args.rich_obs:
        _active_path = RESULTS_PATH_RICH
    else:
        _active_path = RESULTS_PATH

    ks = tuple(int(x) for x in args.ks.split(","))
    if args.snap_ks is not None:
        bucket = max(2, int(args.snap_ks))
        snapped: list[int] = []
        for k in ks:
            # Round k down to the nearest power of `bucket` (≥1).
            level = 0 if k <= 1 else _math.floor(_math.log(k, bucket))
            snapped.append(max(1, bucket**level))
        ks = tuple(dict.fromkeys(snapped))  # dedupe, preserve order
        print(f"K-snap: bucket={bucket} → {list(ks)}")
    ns = ALL_NS if args.ns is None else tuple(int(x) for x in args.ns.split(","))

    print("=" * 96)
    if args.spec_obs:
        obs_label = "spec (3000–7500 Å, R≈500)"
    elif args.joint_obs:
        obs_label = "joint (rich-10band + 4 emission lines)"
    elif args.rich_obs:
        obs_label = "rich (10-band)"
    else:
        obs_label = "SDSS-only (5-band)"
    print(f"VI scaling: powers-of-2 × K-sweep   N={list(ns)}   K={list(ks)}   "
          f"obs={obs_label}   noise={args.noise_frac:.2g}   "
          f"mem budget={MEM_BUDGET_GB} GB")
    print("=" * 96)

    all_rows: list[dict] = _load_rows()
    if all_rows:
        print(f"Loaded {len(all_rows)} cached rows from {_active_path}; "
              "they will be skipped (idempotent resume).")
    if not args.geovi_only:
        run_method("native_vi_linear", ns, ks, "MGVI", all_rows,
                   force=args.force, rich_obs=args.rich_obs,
                   noise_frac=args.noise_frac, spec_obs=args.spec_obs,
                   joint_obs=args.joint_obs)
    if not args.linear_only:
        run_method("native_vi_nonlinear", ns, ks, "geoVI", all_rows,
                   force=args.force, rich_obs=args.rich_obs,
                   noise_frac=args.noise_frac, spec_obs=args.spec_obs,
                   joint_obs=args.joint_obs)
    print(f"\nResults written to {_active_path}")


if __name__ == "__main__":
    sys.exit(main())
