#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Generate mismatch tables for X-like configurations vs external codes.

Reads metadata from config_metadata.py and writes two AASTeX deluxetables:
(A) Configuration choices: side-by-side Pacifici et al. (2023) Table 1 vs tengri choices
(B) Where the X-like configurations differ: detailed mismatches with sources
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Read metadata (no JAX import)
sys.path.insert(0, str(Path(__file__).parent))
from config_metadata import XLIKE_CONFIGS


def _escape_latex(text: str) -> str:
    """Escape LaTeX-special characters in text.

    Parameters
    ----------
    text : str
        Text to escape.

    Returns
    -------
    str
        LaTeX-escaped text.
    """
    # Order matters: backslash first, then other special chars
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    text = text.replace("$", r"\$")
    text = text.replace("%", r"\%")
    text = text.replace("^", r"\textasciicircum{}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("_", r"\_")
    return text


def _format_config_row(
    code_name: str, parity_status: str, table1_row: dict, config_text: dict
) -> tuple[str, list[str]]:
    """Format configuration comparison row for Table A.

    Parameters
    ----------
    code_name : str
        Code name (e.g., "CIGALE").
    parity_status : str
        "Yes" or "No".
    table1_row : dict
        Pacifici et al. (2023) Table 1 row (sampler, sfh, ssp, nebular, dust_att, dust_em, agn).
    config_text : dict
        Tengri configuration fields (sfh, library, nebular, attenuation, dust_ir).

    Returns
    -------
    tuple
        (latex_header_row, list of component rows as strings)
    """
    components = [
        "Star formation history",
        "Stellar library",
        "Nebular emission",
        "Dust attenuation",
        "Dust emission",
    ]
    component_keys = ["sfh", "library", "nebular", "attenuation", "dust_ir"]
    pacifici_keys = ["sfh", "ssp", "nebular", "dust_att", "dust_em"]

    rows = []
    for i, (comp, cfg_key, p_key) in enumerate(zip(components, component_keys, pacifici_keys)):
        pacifici_val = table1_row.get(p_key, "---")
        config_val = _escape_latex(config_text.get(cfg_key, "---"))

        if i == 0:
            # First row includes the code name and parity check
            row = (
                f"{_escape_latex(code_name):<12} & "
                f"{parity_status:<3} & "
                f"{comp:<25} & "
                f"{_escape_latex(pacifici_val):<18} & "
                f"{config_val}\\\\"
            )
        else:
            # Subsequent rows are component-only
            row = (
                f"{'':12} & {'':3} & {comp:<25} & "
                f"{_escape_latex(pacifici_val):<18} & "
                f"{config_val}\\\\"
            )
        rows.append(row)

    return rows


def write_latex_tables(output_file: Path) -> None:
    """Write two AASTeX deluxetables (configuration choices and differences).

    Parameters
    ----------
    output_file : Path
        Output .tex file.
    """
    with open(output_file, "w") as f:
        # =====================================================================
        # TABLE A: Configuration choices
        # =====================================================================
        f.write(r"\begin{deluxetable*}{llp{3.5cm}p{3.5cm}p{3.5cm}}")
        f.write("\n")
        f.write(
            r"\tablecaption{X-like configuration choices. Each row shows "
            r"Pacifici et al. (2023) Table 1 specifications beside tengri's "
            r"X-like configuration. BEAGLE and Dense Basis have no reproduction "
            r"notebook parity check.}"
        )
        f.write("\n")
        f.write(
            r"\tablehead{Code & Parity & Component & "
            r"Pacifici et al. (2023) & tengri X-like}"
        )
        f.write("\n")
        f.write(r"\startdata")
        f.write("\n")

        for key in sorted(XLIKE_CONFIGS.keys()):
            cfg = XLIKE_CONFIGS[key]
            code = cfg["code"]
            parity = "Yes" if cfg["parity_check"] else "No"

            fiducial_table1 = cfg.get("fiducial_table1", {})
            config_text = {
                "sfh": cfg.get("sfh", "---"),
                "library": cfg.get("library", "---"),
                "nebular": cfg.get("nebular", "---"),
                "attenuation": cfg.get("attenuation", "---"),
                "dust_ir": cfg.get("dust_ir", "---"),
            }

            rows = _format_config_row(code, parity, fiducial_table1, config_text)
            for row in rows:
                f.write(row)
                f.write("\n")

        f.write(r"\enddata")
        f.write("\n")
        f.write(r"\label{tab:xlike_choices}")
        f.write("\n")
        f.write(r"\end{deluxetable*}")
        f.write("\n\n")

        # =====================================================================
        # TABLE B: Where the X-like configurations differ
        # =====================================================================
        f.write(r"\begin{deluxetable*}{lp{6cm}l}")
        f.write("\n")
        f.write(
            r"\tablecaption{Mismatches between tengri's X-like configurations and their "
            r"corresponding external codes. Each row lists one difference and its source. "
            r"BEAGLE and Dense Basis have no reproduction notebook parity check.}"
        )
        f.write("\n")
        f.write(r"\tablehead{Code & Difference & Source}")
        f.write("\n")
        f.write(r"\startdata")
        f.write("\n")

        for key in sorted(XLIKE_CONFIGS.keys()):
            cfg = XLIKE_CONFIGS[key]
            code = cfg["code"]
            mismatches_text = cfg.get("mismatches_text", [])
            sources = cfg.get("mismatch_sources", [])

            for i, mismatch in enumerate(mismatches_text):
                source = sources[i] if i < len(sources) else "(no source)"
                # Truncate source to file name only for brevity in table
                if "/" in source:
                    # Extract filename from path before colon
                    file_part = source.split(":")[0]
                    source_display = file_part.split("/")[-1]
                else:
                    source_display = source.split(":")[0]

                if i == 0:
                    # First row includes the code name
                    row = (
                        f"{_escape_latex(code):<12} & "
                        f"{_escape_latex(mismatch):<40} & "
                        f"{_escape_latex(source_display)}\\\\"
                    )
                else:
                    # Subsequent rows are mismatch-only
                    row = (
                        f"{'':12} & "
                        f"{_escape_latex(mismatch):<40} & "
                        f"{_escape_latex(source_display)}\\\\"
                    )
                f.write(row)
                f.write("\n")

        f.write(r"\enddata")
        f.write("\n")
        f.write(r"\label{tab:xlike_differences}")
        f.write("\n")
        f.write(r"\end{deluxetable*}")
        f.write("\n")


def write_json(output_file: Path) -> None:
    """Write JSON summary with both mismatches and astronomy-language mismatches_text.

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
            "mismatches_text": cfg.get("mismatches_text", []),
            "mismatch_sources": cfg.get("mismatch_sources", []),
        }
        if "notes" in cfg:
            summary[key]["notes"] = cfg["notes"]
        if "notes_sources" in cfg:
            summary[key]["notes_sources"] = cfg["notes_sources"]
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)


def print_summary() -> None:
    """Print table to stdout.

    Each X-like configuration's code name, parity status, mismatches, and sources.
    """
    print("\nX-like configuration summary:\n")
    for key in sorted(XLIKE_CONFIGS.keys()):
        cfg = XLIKE_CONFIGS[key]
        parity = "Yes" if cfg["parity_check"] else "No"
        n_mismatches = len(cfg.get("mismatches", []))
        print(f"{key:20} {cfg['code']:15} parity={parity:3} mismatches={n_mismatches}")
        mismatches_text = cfg.get("mismatches_text", [])
        sources = cfg.get("mismatch_sources", [])
        for i, m in enumerate(mismatches_text[:2]):
            source = sources[i] if i < len(sources) else "(no source)"
            print(f"  - {m}")
            print(f"    source: {source}")
        if n_mismatches > 2:
            print(f"  ... {n_mismatches - 2} more")
        if "notes" in cfg:
            print("  Notes:")
            notes = cfg.get("notes", [])
            notes_sources = cfg.get("notes_sources", [])
            for i, note in enumerate(notes):
                source = notes_sources[i] if i < len(notes_sources) else "(no source)"
                print(f"    - {note}")
                print(f"      source: {source}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate X-like configuration mismatch tables")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/xlike_mismatch_table.json"),
        help="Write JSON summary to this file (default: results/xlike_mismatch_table.json)",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("tables/xlike_mismatch_table.tex"),
        help="Write LaTeX table to this file (default: tables/xlike_mismatch_table.tex)",
    )
    args = parser.parse_args()

    print_summary()

    if args.out_json:
        print(f"Writing JSON to {args.out_json}")
        write_json(args.out_json)

    if args.out_tex:
        print(f"Writing LaTeX to {args.out_tex}")
        write_latex_tables(args.out_tex)
