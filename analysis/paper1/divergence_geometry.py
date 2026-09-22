# SPDX-License-Identifier: BSD-3-Clause
"""Where do a fit's divergent transitions sit in parameter space?

A divergence count says a posterior has a geometry problem. It does not say
which kind, and the two kinds have different remedies.

A **funnel** concentrates divergent draws in one coordinate's neck: that
parameter's divergent draws sit far out in its own distribution, typically two
standard deviations or more from the median, while the others look ordinary.
The fix is that parameter -- reparameterize it, bound it, or fix it.

**Curvature or correlation** spreads them out. No single coordinate looks
unusual because the problem is in the directions *between* coordinates, which
a diagonal mass matrix cannot represent at all. The fix is the metric, or a
smaller step, and hunting for a guilty parameter will not find one.

So this reports, per parameter, how far the divergent draws sit from the bulk
in units of that parameter's own spread. The shape of the ranking is the
answer, not any single row.

Measured on the refused 11 h run (``mock_joint_nuts_ta085_20260922``, 79
divergences in 2400 draws): the largest offset is ``neb_dig_frac`` at -0.69,
nothing else past 0.44, and ``neb_dig_frac`` sits mid-range in a 0-1 fraction
rather than against a bound. Diffuse, therefore, and not a funnel.

CLI::

    python -m paper1.divergence_geometry [--posterior NPZ] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_POSTERIOR = RESULTS / "mock_joint_mcmc_nuts.npz"
DEFAULT_OUT = RESULTS / "divergence_geometry.json"

#: At or beyond this many standard deviations, divergent draws are sitting in
#: one coordinate's tail rather than scattered through the bulk. Two sigma is
#: the conventional reading of "in the neck"; below it the ranking is flat
#: enough that no parameter is being singled out.
FUNNEL_SIGMA = 2.0


def offsets(posterior, free_names: list[str]) -> list[dict]:
    """Per parameter, the divergent draws' median offset in units of its spread.

    A parameter whose draws never move has no spread to measure against, and
    dividing by it would manufacture an infinite offset out of a constant. It
    is reported with ``offset_sd`` of ``None`` instead, which says "no scale"
    rather than "enormously off".
    """
    rows = []
    for name in free_names:
        key = f"divergent_{name}"
        if key not in posterior:
            continue
        allv = np.asarray(posterior[name], dtype=float)
        divv = np.asarray(posterior[key], dtype=float)
        if divv.size == 0:
            continue
        sd = float(allv.std())
        median_all = float(np.median(allv))
        median_div = float(np.median(divv))
        rows.append(
            {
                "parameter": name,
                "median_all": median_all,
                "median_divergent": median_div,
                "sd": sd,
                "offset_sd": ((median_div - median_all) / sd) if sd > 0 else None,
            }
        )
    rows.sort(key=lambda r: abs(r["offset_sd"] or 0.0), reverse=True)
    return rows


def verdict(rows: list[dict], n_divergent: int) -> dict:
    """Funnel or diffuse, or a refusal to say."""
    if n_divergent == 0:
        return {
            "shape": None,
            "note": (
                "no divergent transitions, so there is no geometry to locate. "
                "This is not a diffuse result; it is an absent one."
            ),
        }
    scored = [r for r in rows if r["offset_sd"] is not None]
    if not scored:
        return {
            "shape": None,
            "note": "no parameter has a spread to measure an offset against",
        }
    worst = scored[0]
    if abs(worst["offset_sd"]) >= FUNNEL_SIGMA:
        return {
            "shape": "funnel",
            "parameter": worst["parameter"],
            "offset_sd": worst["offset_sd"],
            "note": (
                f"divergent draws concentrate in {worst['parameter']} at "
                f"{worst['offset_sd']:+.2f} sd. The remedy is that parameter, "
                "not the step size."
            ),
        }
    return {
        "shape": "diffuse",
        "parameter": worst["parameter"],
        "offset_sd": worst["offset_sd"],
        "note": (
            f"no parameter is singled out -- the largest offset is "
            f"{worst['parameter']} at {worst['offset_sd']:+.2f} sd, short of "
            f"{FUNNEL_SIGMA}. The problem is in the directions between "
            "coordinates, which a diagonal mass matrix cannot represent, or "
            "the step is simply too long. Looking for a guilty parameter will "
            "not find one."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if not args.posterior.is_file():
        raise SystemExit(f"no posterior at {args.posterior}")

    with np.load(args.posterior, allow_pickle=True) as handle:
        posterior = {k: handle[k] for k in handle.files}

    if "free_params" not in posterior:
        raise SystemExit(f"{args.posterior} records no free_params, so its columns are unnamed")
    free_names = [str(x) for x in posterior["free_params"]]

    if "divergent_mask" not in posterior:
        raise SystemExit(
            f"{args.posterior} carries no divergent_mask. Absent is not zero: a "
            "run that never recorded which draws diverged cannot be said to "
            "have had none."
        )
    n_divergent = int(np.asarray(posterior["divergent_mask"], dtype=bool).sum())

    rows = offsets(posterior, free_names)
    shape = verdict(rows, n_divergent)

    print(f"divergent draws: {n_divergent} of {len(posterior[free_names[0]])}")
    print(f"\n{'parameter':<28}{'offset/sd':>11}{'median all':>13}{'median div':>13}")
    print("-" * 65)
    for row in rows[: args.top]:
        off = "n/a" if row["offset_sd"] is None else f"{row['offset_sd']:+.2f}"
        print(
            f"{row['parameter']:<28}{off:>11}{row['median_all']:>13.4f}"
            f"{row['median_divergent']:>13.4f}"
        )
    print(f"\nshape: {shape['shape']}\n{shape['note']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"n_divergent": n_divergent, "parameters": rows, "verdict": shape},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
