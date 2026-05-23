"""Targeted retry of backends that failed for fixable reasons.

- mcmc_raytrace: harness used wrong kwarg name (n_samples→n_steps)
- mcmc (auto): NUTS missing context normalisation — bug fixed in src
- native_vi_*: default n_seeds=5 explodes compile; now pinned to 1
- mcmc_nuts: dense_mass→dense_mass_matrix kwarg name (fixed)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

SCRIPT = Path(__file__).parent / "validate_backends_231.py"
TMP = Path("/tmp/validate_231_retry")
TMP.mkdir(exist_ok=True)

TARGETS = [
    ("mcmc_nuts", "dpl", 300),
    ("mcmc_raytrace", "dpl", 300),
    ("mcmc", "dpl", 300),
    ("native_vi_linear", "dpl", 300),
    ("native_vi_nonlinear", "dpl", 300),
    ("mcmc_raytrace", "dense_basis", 300),
    ("native_vi_linear", "dense_basis", 300),
    ("native_vi_nonlinear", "dense_basis", 300),
]


def main():
    out = []
    for backend, variant, timeout in TARGETS:
        out_json = TMP / f"{backend}_{variant}.json"
        if out_json.exists():
            out_json.unlink()
        cmd = [sys.executable, str(SCRIPT), "--child", backend, variant, str(out_json)]
        print(f"  -> {backend:25s} {variant:14s}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, timeout=timeout, check=False)
            if out_json.exists():
                r = json.loads(out_json.read_text())
            else:
                r = {"backend": backend, "variant": variant,
                     "status": "crashed_no_output"}
        except subprocess.TimeoutExpired:
            r = {"backend": backend, "variant": variant,
                 "status": "timeout", "wall_s": timeout}
        out.append(r)
        if r["status"] == "ok":
            print(f"   cold={r['cold_s']:6.1f}s warm={r.get('warm_s',0):.1f}s rss={r['rss_gb_peak']:.2f}GB",
                  flush=True)
        else:
            print(f"   FAIL[{r['status']}] {r.get('error_type','')} "
                  f"{(r.get('error_msg') or '')[:80]}", flush=True)
    Path("scripts/_backend_validation_retry.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
