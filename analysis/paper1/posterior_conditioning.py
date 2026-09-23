# SPDX-License-Identifier: BSD-3-Clause
"""What does a diagonal mass matrix leave behind on this posterior?

Three NUTS attempts on the joint mock failed to reach the basin, and after the
third the remaining untouched lever was the metric: ``dense_mass_matrix`` was
``False`` on all of them. ``divergence_geometry`` had already read the ta085
divergences as *diffuse* -- no single coordinate singled out, the trouble in
the directions between them -- but that verdict comes from where the divergent
draws sit, which is indirect. This measures the geometry itself, from the draws
a finished run already wrote, with no model build and no refit.

The argument is one line of linear algebra. A diagonal mass matrix rescales
each coordinate by its own standard deviation and does nothing else, so what it
leaves behind is exactly the **correlation** matrix. Its condition number is
therefore the conditioning HMC still faces after diagonal preconditioning, and
a dense metric is worth its cost precisely when that number is large.

Why the number matters more than it looks: the step size is limited by the
posterior's *tightest* direction while the distance to traverse is set by its
*widest*, so the trajectory length a sampler needs grows like the square root
of the condition number. A condition number of 400 is a factor of twenty in
leapfrog steps per draw, which is wall clock, not elegance.

Measured on the mock (2026-09-23): 464 on ta085 and 385 on ta095, with no
pairwise correlation above 0.76 and only two above 0.7. That gap is the point.
The conditioning is not one bad pair a reparameterization would fix; it is many
moderate correlations compounding across 36 dimensions, which is the "diffuse"
verdict restated in a form that names a remedy.

**A badly mixed chain cannot estimate this.** A p-by-p correlation matrix needs
more than p independent draws to exist at all and several times p to be stable,
so the effective sample size is a precondition rather than a footnote, and this
refuses below :data:`MIN_ESS_PER_PARAM` rather than reporting a number built on
too little.

CLI::

    python -m paper1.posterior_conditioning POSTERIOR.npz [--top 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Effective samples required per free parameter before a correlation matrix is
#: worth computing. A p-by-p covariance is singular with fewer than p draws and
#: merely unstable for a while after; five per parameter is the conventional
#: floor for an estimate anyone should act on. ta085 reaches 14.4 and is
#: reported; ta095, at 118 effective samples over 36 parameters, reaches 3.3 and
#: is refused -- which is correct, and is why the two runs are not equally good
#: evidence even though both produced 2400 draws.
MIN_ESS_PER_PARAM = 5.0


def conditioning(path: Path, top: int = 8) -> dict:
    """Condition number of the posterior correlation matrix, and what drives it."""
    with np.load(path, allow_pickle=True) as handle:
        data = {k: handle[k] for k in handle.files}

    if "free_params" not in data:
        raise SystemExit(f"{path} records no free_params, so its columns are unnamed")
    names = [str(x) for x in data["free_params"]]

    columns = []
    kept = []
    for name in names:
        if name not in data:
            continue
        col = np.asarray(data[name], dtype=float).ravel()
        columns.append(col)
        kept.append(name)
    if not columns:
        raise SystemExit(f"{path} holds none of the parameters it names in free_params")

    draws = np.column_stack(columns)
    draws = draws[np.all(np.isfinite(draws), axis=1)]

    # A parameter that never moved has no correlation with anything, and
    # including it makes the matrix singular for a reason that is not geometry.
    spread = draws.std(axis=0)
    moving = spread > 0
    frozen = [n for n, m in zip(kept, moving) if not m]
    draws = draws[:, moving]
    kept = [n for n, m in zip(kept, moving) if m]

    n_param = len(kept)
    ess_min = _ess_min(data, kept)
    if ess_min is not None and ess_min / n_param < MIN_ESS_PER_PARAM:
        raise SystemExit(
            f"REFUSED: {ess_min:.1f} effective samples over {n_param} parameters is "
            f"{ess_min / n_param:.1f} per parameter, under the {MIN_ESS_PER_PARAM} "
            "floor. A correlation matrix estimated from this would be noise "
            "wearing a condition number, and the number is the whole output. "
            "Measure it on a run that mixed."
        )

    corr = np.corrcoef(draws, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    cond = float(eigenvalues.max() / max(eigenvalues.min(), 1e-300))

    off = corr - np.eye(n_param)
    pairs = [
        (float(off[a, b]), kept[a], kept[b]) for a in range(n_param) for b in range(a + 1, n_param)
    ]
    pairs.sort(key=lambda p: abs(p[0]), reverse=True)

    return {
        "posterior": str(path),
        "n_draws": int(draws.shape[0]),
        "n_param": n_param,
        "frozen": frozen,
        "ess_min": ess_min,
        "condition_number": cond,
        "trajectory_factor": float(np.sqrt(cond)),
        "top_pairs": [{"r": r, "a": a, "b": b} for r, a, b in pairs[:top]],
        "n_above_0.7": sum(1 for r, _, _ in pairs if abs(r) > 0.7),
        "n_above_0.9": sum(1 for r, _, _ in pairs if abs(r) > 0.9),
    }


def _ess_min(data: dict, names: list[str]) -> float | None:
    """Smallest per-parameter effective sample size the file records, if any."""
    found = [float(data[f"ess_{n}"]) for n in names if f"ess_{n}" in data]
    return min(found) if found else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("posterior", type=Path)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    report = conditioning(args.posterior, args.top)

    print(f"{report['n_draws']} draws, {report['n_param']} moving parameters")
    if report["frozen"]:
        print(f"  frozen, excluded: {report['frozen']}")
    if report["ess_min"] is not None:
        print(
            f"  ess_min {report['ess_min']:.1f}"
            f"  ({report['ess_min'] / report['n_param']:.1f} per parameter)"
        )
    print(f"\ncorrelation-matrix condition number: {report['condition_number']:,.1f}")
    print("  this is what a diagonal mass matrix leaves behind")
    print(f"  trajectory length scales as its square root: {report['trajectory_factor']:.1f}x")
    print(f"\ntop correlated pairs (of {report['n_param'] * (report['n_param'] - 1) // 2}):")
    for pair in report["top_pairs"]:
        print(f"  {pair['r']:+.3f}  {pair['a']}  x  {pair['b']}")
    print(f"\n|r| > 0.7: {report['n_above_0.7']}    |r| > 0.9: {report['n_above_0.9']}")
    if report["n_above_0.9"] == 0 and report["condition_number"] > 100:
        print(
            "  No single pair is extreme while the matrix is badly conditioned, so\n"
            "  this is many moderate correlations compounding rather than one pair a\n"
            "  reparameterization would fix. A dense metric addresses the former."
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
