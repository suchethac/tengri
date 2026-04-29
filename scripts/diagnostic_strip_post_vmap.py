"""Diagnostic strip: 4 cells re-run after vectorized MAP-init refactor.

Compares wall-time and σ_PSD posterior to the existing baseline in
`data/vi_scaling_benchmark.json` for the same (N, K, method) cells.
Output: analysis/diagnostic_strip_post_vmap.json with side-by-side numbers.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Reuse the existing benchmark's spawn() so we get the watchdog + JSON shape.
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_population_native import spawn  # noqa: E402

CELLS = [
    {"n_gal": 1024, "method": "native_vi_linear", "K": 1},
    {"n_gal": 1024, "method": "native_vi_linear", "K": 64},
    {"n_gal": 8192, "method": "native_vi_linear", "K": 1},
    {"n_gal": 8192, "method": "native_vi_linear", "K": 64},
]

N_ITER = 15
N_SAMP = 3

BASELINE_PATH = Path("data/vi_scaling_benchmark.json")
OUT_PATH = Path("analysis/diagnostic_strip_post_vmap.json")


def find_baseline_cell(rows, n_gal: int, method: str, K: int) -> dict | None:
    for r in rows:
        if (r.get("n_gal") == n_gal
                and r.get("method") == method
                and r.get("forward_chunk_size") == K):
            return r
    return None


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if BASELINE_PATH.exists():
        baseline_rows = json.loads(BASELINE_PATH.read_text())
    else:
        print(f"WARN: no baseline at {BASELINE_PATH}; only post-vmap rows reported")
        baseline_rows = []

    results = []
    for cell in CELLS:
        n_gal = cell["n_gal"]
        method = cell["method"]
        K = cell["K"]
        print(f"\n=== N={n_gal} K={K} {method} ===", flush=True)

        baseline = find_baseline_cell(baseline_rows, n_gal, method, K)
        if baseline is None:
            print(f"  no baseline cell — running post-vmap only")
        else:
            print(f"  baseline wall_s = {baseline.get('wall_s'):.1f}, "
                  f"σ median = {baseline.get('psd_sigma_summary', {}).get('median', '?')}")

        t0 = time.time()
        try:
            row = spawn(
                n_gal=n_gal, method=method,
                n_iterations=N_ITER, n_samples=N_SAMP,
                forward_chunk_size=K, compile_timeout=2400,
            )
        except Exception as exc:
            row = {"error": repr(exc), "wall_s": -1.0}
        wall_outer = time.time() - t0

        post = {
            "cell": cell, "n_iterations": N_ITER, "n_samples": N_SAMP,
            "wall_s": row.get("wall_s"),
            "wall_s_warm": row.get("wall_s_warm"),
            "rss_delta_gb": row.get("rss_delta_gb"),
            "n_iters_used_warm": row.get("n_iters_used_warm"),
            "converged": row.get("converged"),
            "psd_sigma_summary": row.get("psd_sigma_summary"),
            "psd_tau_summary": row.get("psd_tau_summary"),
            "error": row.get("error"),
            "wall_outer_s": wall_outer,
        }
        if baseline is not None:
            post["baseline_wall_s"] = baseline.get("wall_s")
            post["baseline_psd_sigma_summary"] = baseline.get("psd_sigma_summary")
            post["baseline_psd_tau_summary"] = baseline.get("psd_tau_summary")
            if (baseline.get("wall_s") and post["wall_s"]
                    and post["wall_s"] > 0):
                post["wall_speedup"] = baseline["wall_s"] / post["wall_s"]

        results.append(post)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  post-vmap wall_s = {post['wall_s']}")
        if post.get("wall_speedup"):
            print(f"  speedup vs baseline: {post['wall_speedup']:.2f}x")

    print(f"\nWrote {len(results)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
