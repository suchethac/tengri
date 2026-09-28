#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Generate mismatch table for X-like configurations vs external codes.

Reads metadata from config_metadata.py and writes a table comparing
each X-like configuration's choices to Pacifici et al. (2023) Table 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Read metadata (no JAX import)
sys.path.insert(0, str(Path(__file__).parent))
from config_metadata import XLIKE_CONFIGS


def _table_row_latex(key: str, metadata: dict) -> str:
    """Format one row of the LaTeX table.

    Parameters
    ----------
    key : str
        Configuration key.
    metadata : dict
        Configuration metadata dict.

    Returns
    -------
    str
        LaTeX table row.
    """
    code = metadata["code"]
    parity = "Yes" if metadata["parity_check"] else "No"
    n_mismatches = len(metadata.get("mismatches", []))
    mismatches = r"\newline ".join(f"\\textbullet~{m}" for m in metadata.get("mismatches", [])[:3])
    if n_mismatches > 3:
        mismatches += r"\newline " + r"\textbullet~" + f"({n_mismatches - 3} more)"
    return f"{code:<18} & {parity:<3} & {mismatches}\\\\\n"


def write_latex_table(output_file: Path) -> None:
    """Write AASTeX tabular table.

    Parameters
    ----------
    output_file : Path
        Output .tex file.
    """
    header = r"""\begin{deluxetable}{lcp{10cm}}
\tablehead{Code & Parity & Mismatches vs Paper 1 Implementation}
\startdata
"""
    footer = r"""\enddata
\end{deluxetable}
"""
    with open(output_file, "w") as f:
        f.write(header)
        for key in sorted(XLIKE_CONFIGS.keys()):
            f.write(_table_row_latex(key, XLIKE_CONFIGS[key]))
        f.write(footer)


def write_json(output_file: Path) -> None:
    """Write JSON summary.

    Parameters
    ----------
    output_file : Path
        Output .json file.
    """
    summary = {}
    for key in sorted(XLIKE_CONFIGS.keys()):
        cfg = XLIKE_CONFIGS[key]
        summary[key] = {
            "code": cfg["code"],
            "parity_check": cfg["parity_check"],
            "n_mismatches": len(cfg.get("mismatches", [])),
            "mismatches": cfg.get("mismatches", []),
        }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)


def print_summary() -> None:
    """Print table to stdout.

    Each X-like configuration's code name, parity status, and mismatch count.
    """
    print("\nX-like configuration summary:\n")
    for key in sorted(XLIKE_CONFIGS.keys()):
        cfg = XLIKE_CONFIGS[key]
        parity = "Yes" if cfg["parity_check"] else "No"
        n_mismatches = len(cfg.get("mismatches", []))
        print(f"{key:20} {cfg['code']:15} parity={parity:3} mismatches={n_mismatches}")
        for m in cfg.get("mismatches", [])[:2]:
            print(f"  - {m}")
        if n_mismatches > 2:
            print(f"  ... {n_mismatches - 2} more")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate X-like configuration mismatch table")
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Write JSON summary to this file",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        help="Write LaTeX table to this file",
    )
    args = parser.parse_args()

    print_summary()

    if args.out_json:
        print(f"Writing JSON to {args.out_json}")
        write_json(args.out_json)

    if args.out_tex:
        print(f"Writing LaTeX to {args.out_tex}")
        write_latex_table(args.out_tex)
