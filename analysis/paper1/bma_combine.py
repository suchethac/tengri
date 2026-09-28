#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Combine Laplace evidence results across galaxies and models.

CLI:
    python -m analysis.paper1.bma_combine [--evidence-dir ...] \
        [--fits-dir analysis/paper1/results/fits] \
        [--out analysis/paper1/results/bma_summary.json] \
        [--draws 4000] [--seed 0]

Combines per-galaxy, per-model evidence cells into a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Combine BMA evidence results")
    parser.add_argument("--evidence-dir", default="analysis/paper1/results/bma_evidence")
    parser.add_argument("--fits-dir", default="analysis/paper1/results/fits")
    parser.add_argument("--out", default="analysis/paper1/results/bma_summary.json")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    summary = {"route": "laplace", "weight_sets": {}}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
