#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does a results directory hold the configurations this repository declares?

A results directory records which parameters were sampled but not which model
was meant. When a configuration is redefined, the old cells keep their old
model and their old file names, so `13097_IV.npz` goes on being read as
Configuration IV by anything that trusts the name. Nothing raises, and every
per-configuration statement drawn from it is mislabeled.

This measures it. The family a cell actually sampled is recovered from its own
parameter names -- `sfh_dir_log_total_mass`, `sfh_dir_z_0`, ... share the
prefix `sfh_dir_` -- and compared against the family `configs.CONFIGS` declares
for that configuration. The evidence is the cell's own content, so the check
holds for archives written before anything recorded provenance.

The prefix table is static so that consumers need not import tengri (the
figure scripts that read only NPZs stay free of JAX); it is checked against
the live SFH registry by
`analysis/paper1/tests/test_cell_provenance.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SFH registry name -> the parameter prefix its free parameters carry.
# Verified against tengri.list_sfh_models() by the test named above.
SFH_PREFIX_BY_TYPE: dict[str, str] = {
    "const": "sfh_const_",
    "continuity": "sfh_cont_",
    "declining_exp": "sfh_declining_exp_",
    "delayed": "sfh_delayed_",
    "dirichlet": "sfh_dir_",
    "dpl": "sfh_dpl_",
    "lnorm": "sfh_lnorm_",
}

# Arrays under an `sfh_` name that describe the reconstructed history rather
# than a sampled parameter, so they carry no family information.
_NON_PARAMETER_SFH_KEYS = frozenset(
    {
        "sfh_lookback_time_yr",
        "sfh_sfr_median",
        "sfh_sfr_p16",
        "sfh_sfr_p84",
    }
)


@dataclass(frozen=True)
class Mismatch:
    """One configuration whose cells did not sample the declared family."""

    config: str
    declared_type: str
    declared_prefix: str
    found_prefixes: tuple[str, ...]
    n_cells: int

    def describe(self) -> str:
        found = ", ".join(self.found_prefixes) or "nothing"
        return (
            f"Configuration {self.config}: declared {self.declared_type} "
            f"({self.declared_prefix}), {self.n_cells} cell(s) sampled {found}"
        )


def sampled_sfh_prefix(npz_keys) -> str | None:
    """The SFH parameter prefix a cell sampled, from its own parameter names.

    Returns None when the cell records no sampled SFH parameter at all, which
    is a different condition from sampling the wrong one and is reported
    separately by the caller.
    """
    names = [
        key for key in npz_keys if key.startswith("sfh_") and key not in _NON_PARAMETER_SFH_KEYS
    ]
    if not names:
        return None

    # A known family is matched outright. This is the only branch that is
    # correct for a configuration sampling a single SFH parameter, where the
    # common prefix below is the whole parameter name and carries no boundary.
    known = [
        prefix
        for prefix in SFH_PREFIX_BY_TYPE.values()
        if all(name.startswith(prefix) for name in names)
    ]
    if known:
        return max(known, key=len)

    # An unregistered family still gets an answer, but only from two or more
    # names: the longest common prefix cut back to a component boundary.
    # Taking the second underscore-token instead would read
    # `sfh_declining_exp_tau_gyr` as the family `declining`.
    if len(names) < 2:
        return None
    common = names[0]
    for name in names[1:]:
        limit = min(len(common), len(name))
        cut = 0
        while cut < limit and common[cut] == name[cut]:
            cut += 1
        common = common[:cut]
    boundary = common.rfind("_")
    if boundary <= len("sfh"):
        return None
    return common[: boundary + 1]


def audit(results_dir: Path, configs: dict) -> tuple[list[Mismatch], list[str]]:
    """Compare every cell in results_dir against the declared configurations.

    Returns (mismatches, notes). `notes` carries conditions that are not a
    mismatch but that a caller should still surface: configurations declaring
    an SFH type this module does not know, and cells with no sampled SFH
    parameter.
    """
    found: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    notes: list[str] = []

    for npz_path in sorted(Path(results_dir).glob("*_*.npz")):
        config = npz_path.stem.partition("_")[2]
        if config not in configs:
            continue
        with np.load(npz_path, allow_pickle=True) as npz:
            prefix = sampled_sfh_prefix(npz.files)
        counts[config] = counts.get(config, 0) + 1
        if prefix is None:
            notes.append(f"{npz_path.stem}: no sampled SFH parameter recorded")
            continue
        found.setdefault(config, set()).add(prefix)

    # A directory with no cells is not a directory that matches: reporting OK
    # for it turns "the grid has not started" into "the grid is correct", which
    # is the reading that costs the most to be wrong about.
    if not counts:
        notes.append(f"{Path(results_dir).name}: no cells read, so nothing was verified")

    mismatches: list[Mismatch] = []
    for config in configs:
        declared_type = configs[config].get("sfh_type")
        if declared_type is None:
            notes.append(f"Configuration {config}: no sfh_type declared in CONFIGS")
            continue
        declared_prefix = SFH_PREFIX_BY_TYPE.get(declared_type)
        if declared_prefix is None:
            notes.append(
                f"Configuration {config}: declared SFH type {declared_type!r} is not "
                "in SFH_PREFIX_BY_TYPE; cannot verify"
            )
            continue
        prefixes = found.get(config)
        if not prefixes:
            continue
        if prefixes != {declared_prefix}:
            mismatches.append(
                Mismatch(
                    config=config,
                    declared_type=declared_type,
                    declared_prefix=declared_prefix,
                    found_prefixes=tuple(sorted(prefixes)),
                    n_cells=counts.get(config, 0),
                )
            )
    return mismatches, notes


def banner(results_dir: Path, mismatches: list[Mismatch], notes: list[str]) -> str | None:
    """A block to print, or None when the directory matches the declarations."""
    if not mismatches and not notes:
        return None
    lines = ["=" * 78]
    if mismatches:
        lines.append("CONFIGURATION LABELS DO NOT MATCH configs.py")
        lines.append(f"  in {Path(results_dir).name}")
        lines.extend(f"  {m.describe()}" for m in mismatches)
        lines.append("  Per-configuration statements from this directory are mislabeled.")
    for note in notes:
        lines.append(f"  note: {note}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs import CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)

    mismatches, notes = audit(args.results_dir, CONFIGS)
    text = banner(args.results_dir, mismatches, notes)
    if text is None:
        print(f"OK: {args.results_dir} matches the declared configurations")
        return 0
    print(text)
    payload = {
        "results_dir": str(args.results_dir),
        "mismatches": [m.describe() for m in mismatches],
        "notes": notes,
    }
    print(json.dumps(payload, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
