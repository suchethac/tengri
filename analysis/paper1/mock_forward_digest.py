# SPDX-License-Identifier: BSD-3-Clause
"""Emit the mock's forward prediction at truth, for comparing two source trees.

The pin is a frozen branch, so "is it safe to move it forward?" is a question
that recurs and that opinion cannot answer. This writes the quantities the
paper's figure is built from -- both channels, at the mock's own truth -- so
the same script run against two trees gives a diff in the units that matter.

Run it once per tree and diff the two files::

    PYTHONPATH=<tree>/src:<pin>/analysis python -m paper1.mock_forward_digest a.npz
    PYTHONPATH=<other>/src:<pin>/analysis python -m paper1.mock_forward_digest b.npz
    python -m paper1.mock_forward_digest --compare a.npz b.npz

A tree snapshot comes from ``git archive <ref> src``; outside the repo the data
locator cannot walk to ``data/``, so set ``TENGRI_DATA_DIR``. Set
``TENGRI_DISABLE_PRECOMP_CACHE=1`` on both arms: the cache is content-hashed,
but a stale entry is the one failure mode that would make two trees look
identical when they are not.

**Both channels, deliberately.** The spectrum is 1500 of the ~1516 data points.
A photometry-only digest would call a tree "identical" while saying nothing
about 99% of what the fit is fit to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TRUTH = Path(__file__).with_name("results") / "mock_joint_truth.npz"


def digest() -> dict:
    """The mock's forward prediction at truth: photometry and spectrum."""
    from paper1.verify_mock_listing import (
        MOCK_FILTERS,
        SSP_NAME as _SSP,
        build_joint_observation,
        build_mock_model,
    )

    import tengri

    if not TRUTH.is_file():
        raise SystemExit(f"no truth at {TRUTH}; run fig_mock_joint_infer first")
    saved = np.load(TRUTH, allow_pickle=True)
    params = {str(n): float(v) for n, v in zip(saved["free_params"], saved["truth_values"])}

    model = build_mock_model(tengri.load_ssp(_SSP), build_joint_observation())
    return {
        "photometry": np.asarray(model.predict_photometry(params), dtype=np.float64),
        "spectrum": np.asarray(model.predict(params).spectrum(), dtype=np.float64),
        "filters": np.array(MOCK_FILTERS),
        "free_params": np.array(sorted(model.spec.free_params)),
        "tengri_path": np.array(tengri.__file__),
    }


def compare(a: Path, b: Path) -> int:
    """Diff two digests in units of the mock's own noise."""
    left, right = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    saved = np.load(TRUTH, allow_pickle=True)

    if list(left["free_params"]) != list(right["free_params"]):
        raise SystemExit(
            "the two trees declare different free parameters, so a per-band diff "
            "would compare two different models"
        )

    print(f"A: {left['tengri_path']}\nB: {right['tengri_path']}\n")
    worst = 0.0
    for channel, sigma_key in (("photometry", "phot_sig"), ("spectrum", "spec_sig")):
        lo, hi = np.asarray(left[channel]), np.asarray(right[channel])
        if lo.shape != hi.shape:
            raise SystemExit(f"{channel}: shapes differ, {lo.shape} vs {hi.shape}")
        sigma = np.asarray(saved[sigma_key])
        in_sigma = (hi - lo) / sigma
        identical = int((hi == lo).sum())
        print(
            f"{channel:<11} {identical}/{lo.size} bit-identical  "
            f"max|shift| {np.nanmax(np.abs(in_sigma)):.4f} sigma  "
            f"chi2 {float(np.nansum(in_sigma**2)):.4f}"
        )
        worst = max(worst, float(np.nanmax(np.abs(in_sigma))))
        if channel == "photometry":
            for i, band in enumerate(left["filters"]):
                if hi[i] != lo[i]:
                    rel = (hi[i] - lo[i]) / lo[i] if lo[i] else float("nan")
                    print(f"  {band!s:<14}{rel:>10.3%}{in_sigma[i]:>10.4f} sigma")
    print(f"\nworst shift across both channels: {worst:.4f} sigma")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, nargs="?", help="where to write this tree's digest")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare)
    if args.out is None:
        parser.error("give an output path, or --compare A B")
    np.savez(args.out, **digest())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
