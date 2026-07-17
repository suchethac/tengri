#!/usr/bin/env python
"""Merge per-slice shards from ``run_catalog_slice.py`` into one ordered table.

Concatenates every ``shard_*.npz`` in a directory and sorts rows by
``global_index`` so the merged catalog is in the original input order,
regardless of how the slices were scheduled.

Usage::

    python merge_shards.py --shards shards/ --out catalog_posteriors.npz
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge catalog posterior shards.")
    ap.add_argument("--shards", required=True, help="directory of shard_*.npz (or a glob)")
    ap.add_argument("--out", required=True, help="output npz path")
    args = ap.parse_args(argv)

    pattern = args.shards
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "shard_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shards match {pattern!r}")

    parts = [dict(np.load(f)) for f in files]
    keys = set(parts[0])
    for i, p in enumerate(parts[1:], 1):
        if set(p) != keys:
            raise SystemExit(f"shard {files[i]} has mismatched columns {set(p) ^ keys}")

    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}

    n = merged["global_index"].size
    if not np.array_equal(merged["global_index"], np.arange(n)):
        print(
            f"warning: global_index is not a contiguous 0..{n - 1} range "
            "(some galaxies missing or duplicated?)"
        )

    np.savez(args.out, **merged)
    print(f"merged {len(files)} shards -> {args.out} ({n} galaxies, {len(keys)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
