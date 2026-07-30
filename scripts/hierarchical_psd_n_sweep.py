# SPDX-License-Identifier: BSD-3-Clause
"""N-sweep for hierarchical PSD recovery — the measurement that decides the claim.

The companion paper's section 4.3 asserts the shared-PSD credible intervals
"shrink approximately as 1/sqrt(N)". A later measurement found they did not
shrink at all across an 8192-fold data increase, which is what a prior-dominated
posterior looks like. This script settles it: fit at several N, regress
log(interval width) on log(N), and report whether the slope excludes zero.

The operative half of the criterion is EXCLUDING ZERO, not matching -0.5. A flat
slope means pooling is not happening, whatever the point estimates look like.

Run with the worktree on the path (the editable install points at the MAIN
checkout, which does not carry tengri.inference.population)::

  PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_n_sweep.py
"""

from __future__ import annotations

import json
import time

import numpy as np

# Import the single-N driver so the sweep cannot drift from the run it sweeps.
from scripts.hierarchical_psd_recovery_run import (
    TRUTH_SIGMA,
    TRUTH_TAU_MYR,
    run_recovery,
)

from tengri.inference.population.diagnostics import interval_width_scaling

# Sized to what this machine finishes. Two earlier sweeps were OOM-killed
# at N=16; a 4x range in N is enough to fit a slope.
N_VALUES = (4, 8, 12, 16)
OUT_JSON = "n_sweep_results.json"


def main():
    import tengri

    print("tengri:", tengri.__file__)
    if "worktrees/hierarchical-psd-spec" not in tengri.__file__:
        raise SystemExit(
            "WRONG CHECKOUT — set PYTHONPATH=<worktree>/src. The editable install "
            "resolves to the main checkout, which has no inference.population."
        )

    print(f"\nN-sweep over {N_VALUES}, truths sigma={TRUTH_SIGMA}, tau={TRUTH_TAU_MYR} Myr\n")

    rows = []
    for n in N_VALUES:
        t0 = time.time()
        r = run_recovery(n_galaxies=n)
        dt = time.time() - t0
        s_w = r["sigma_84"] - r["sigma_16"]
        t_w = r["tau_84"] - r["tau_16"]
        rows.append(
            {
                "n": n,
                "sigma_med": r["sigma_med"],
                "sigma_width": s_w,
                "tau_med": r["tau_med"],
                "tau_width": t_w,
                "wall_s": dt,
            }
        )
        print(
            f"N={n:3d}  sigma {r['sigma_med']:.3f} (width {s_w:.3f})  "
            f"tau {r['tau_med']:.1f} (width {t_w:.1f})  {dt:.0f}s",
            flush=True,
        )
        # Persist after EVERY N. Two earlier sweeps were OOM-killed partway and
        # lost every completed point, because results were only written at the
        # end. A long run on a machine that can kill it must checkpoint.
        with open(OUT_JSON, "w") as fh:
            json.dump({"rows": rows, "verdict": None}, fh, indent=2)

    n_arr = np.array([r["n"] for r in rows], dtype=float)
    print("\n" + "=" * 72)
    print("WIDTH SCALING — the gate")
    print("=" * 72)

    verdict = {}
    for label, key, truth in (
        ("sigma", "sigma_width", TRUTH_SIGMA),
        ("tau", "tau_width", TRUTH_TAU_MYR),
    ):
        w = np.array([r[key] for r in rows], dtype=float)
        out = interval_width_scaling(w, n_arr)
        verdict[label] = out
        status = "PASS" if out["excludes_zero_3sigma"] else "FAIL"
        print(
            f"{label:6s}: slope {out['slope']:+.3f} +/- {out['slope_err']:.3f}  "
            f"excludes zero at 3 sigma: {out['excludes_zero_3sigma']}  [{status}]"
        )
        print(f"        widths {np.array2string(w, precision=3)} at N={N_VALUES}")

    print(
        "\nInterpretation: a slope near -0.5 excluding zero means the intervals\n"
        "shrink with N and pooling works. A slope consistent with zero means the\n"
        "posterior is prior-dominated and the paper's claim does not hold."
    )

    with open(OUT_JSON, "w") as fh:
        json.dump(
            {"rows": rows, "verdict": {k: dict(v) for k, v in verdict.items()}}, fh, indent=2
        )
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
