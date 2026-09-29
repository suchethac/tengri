#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Generate mismatch tables for X-like configurations vs external codes.

Reads metadata from config_metadata.py and writes two booktabs tables (table*):
(A) Configuration choices: side-by-side Pacifici et al. (2023) Table 1 vs tengri choices
(B) Where the X-like configurations differ: detailed mismatches with sources
"""

from __future__ import annotations

import argparse
import json
import re
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


# Reader-facing wording for the tengri column of Table A, keyed by the raw metadata
# value other code reads. A value without an entry raises, so a new configuration
# cannot reach the paper in shorthand.
_TENGRI_DISPLAY = {
    "sfh": {
        "delayed-tau": r"Delayed-$\tau$",
        "double power law": "Double power law",
        "continuity, 7 bins": "Continuity, seven age bins",
        "dense basis quantiles": r"Dense Basis quantile form \citep{Iyer_2019}",
    },
    "library": {
        "BC03 2003 STELIB Chabrier": (
            r"\citet{Bruzual_2003} with STELIB spectra, \citet{Chabrier_2003} IMF"
        ),
        "FSPS MIST/MILES": (
            r"FSPS with MIST isochrones and MILES spectra, \citet{Chabrier_2003} IMF"
        ),
    },
    "nebular": {"Cue": r"\textsc{Cue} \citep{Li_2025}"},
    "attenuation": {
        "Calzetti, 1-comp": r"\citet{Calzetti_2000}, single screen",
        "Calzetti, 2-comp": r"\citet{Calzetti_2000}, birth-cloud and diffuse screens",
        "Leitherer+02, 2-comp": r"\citet{Leitherer_2002}, birth-cloud and diffuse screens",
        "Charlot+Fall 2000, 2-comp": r"\citet{Charlot_2000}, birth-cloud and diffuse screens",
    },
    "dust_ir": {
        "Draine+2007": r"\citet{Draine_2007}",
        "Dale+2014 CIGALE": r"\citet{Dale_2014}, CIGALE template set",
        "None": "None",
    },
}

# (pattern, LaTeX form, plain form) for the Source column of Table B. The exact
# file:line stays in the JSON; a reader of the paper gets a citation or a data product.
_REPRO_CODE = {"cigale": "CIGALE", "prospector": "Prospector", "bagpipes": "BAGPIPES"}
_REPRO_PATTERN = re.compile(r"^reproduction/(\w+)/")
_PAPER_SECTION = re.compile(r"^paper Section (\S+)$")
_SOURCE_RULES = (
    (
        r"^Pacifici et al\. \(2023\) Table 1",
        r"\citet{Pacifici_2023}, Table~1",
        "Pacifici et al. (2023), Table 1",
    ),
    (
        r"^art_sedfitting/code_outputs/BEAGLE_summary_catalogue_z1\.fits",
        r"BEAGLE workshop catalog (\texttt{tauV\_eff} and \texttt{mu} outputs) "
        r"\citep{Pacifici_2023}",
        "BEAGLE workshop catalog (tauV_eff and mu outputs), Pacifici et al. (2023)",
    ),
    (
        r"^art_sedfitting/code_outputs/header",
        r"Workshop catalog (column list) \citep{Pacifici_2023}",
        "workshop catalog (column list), Pacifici et al. (2023)",
    ),
    (
        r"^pcigale/",
        r"CIGALE source code \citep{Boquien_2019}",
        "CIGALE source code (Boquien et al. 2019)",
    ),
    (
        r"^bagpipes/",
        r"BAGPIPES source code \citep{Carnall_2018}",
        "BAGPIPES source code (Carnall et al. 2018)",
    ),
    (
        r"^prospect/",
        r"Prospector source code \citep{Johnson_2021b}",
        "Prospector source code (Johnson et al. 2021)",
    ),
    (
        r"dense_basis/priors\.py",
        r"Dense Basis source \citep{Iyer_2019}",
        "Dense Basis source (Iyer et al. 2019)",
    ),
    (r"^(analysis/paper1/|src/tengri/)", "this work", "this work"),
)


def _source_part(part: str) -> tuple[str, str]:
    """Map one raw source (a file:line or a citation) to (LaTeX, plain) reader wording."""
    part = part.strip()
    m = _REPRO_PATTERN.match(part)
    if m and m.group(1) in _REPRO_CODE:
        text = f"tengri reproduction notebook ({_REPRO_CODE[m.group(1)]})"
        return text, text
    m = _PAPER_SECTION.match(part)
    if m:
        return rf"Section~\ref{{{m.group(1)}}}", f"Section {m.group(1)} of the paper"
    for pattern, latex, plain in _SOURCE_RULES:
        if re.search(pattern, part):
            return latex, plain
    raise ValueError(f"No reader-facing form for source {part!r}; add a rule to _SOURCE_RULES")


def source_display(raw: str) -> tuple[str, str]:
    """Return (LaTeX, plain) reader-facing wording for a raw source string.

    Parameters
    ----------
    raw : str
        Raw source, possibly several ``;``-separated parts.

    Returns
    -------
    tuple of str
        The LaTeX cell text and its plain-text equivalent, duplicates removed.
    """
    pairs = []
    for part in raw.split(";"):
        pair = _source_part(part)
        if pair not in pairs:
            pairs.append(pair)
    return "; ".join(p[0] for p in pairs), "; ".join(p[1] for p in pairs)


def display_code(code: str) -> str:
    """Return the reader-facing code name (``Dense_Basis`` -> ``Dense Basis``)."""
    return code.replace("_", " ")


def tengri_display(field: str, raw: str) -> str:
    """Return the journal wording (LaTeX) of one tengri configuration value."""
    try:
        return _TENGRI_DISPLAY[field][raw]
    except KeyError:
        raise ValueError(
            f"No reader-facing wording for {field}={raw!r}; add it to _TENGRI_DISPLAY"
        ) from None


def _prose(text: str) -> str:
    """Escape prose for LaTeX and typeset the optical-depth and reddening symbols."""
    out = _escape_latex(text)
    for raw, math in (
        (r"(tau\_V, mu)", r"($\tau_V$, $\mu$)"),
        (r"tau\_V", r"$\tau_V$"),
        ("E(B-V)", r"$E(B{-}V)$"),
    ):
        out = out.replace(raw, math)
    return out


_PARITY_NOTE = (
    "Parity is Yes where a reproduction notebook checks the configuration against the "
    "external code; BEAGLE and Dense Basis have none."
)
# AASTeX 6.3.1 breaks every p/m/b column type (its table tools redefine the array
# preamble parser; "! Extra \or." at \begin{tabular}, deluxetable or not), so wrapped
# cells are \parbox[t] in plain l columns.
_TABLE_A_COLUMNS = r"@{}lllll@{}"
_TABLE_B_COLUMNS = r"@{}lll@{}"
_WIDTH_TENGRI = "5.4cm"
_WIDTH_OPTIONS = "3.4cm"
_WIDTH_DIFFERENCE = "9.6cm"
_WIDTH_SOURCE = "4.6cm"


def _wrap(text: str, width: str) -> str:
    """Return ``text`` as a top-aligned ragged-right paragraph cell of the given width."""
    return rf"\parbox[t]{{{width}}}{{\raggedright {text}}}"


_ROW_END = r" \\"
# (component label, tengri metadata key, Pacifici Table 1 key)
_COMPONENTS = (
    ("Star formation history", "sfh", "sfh"),
    ("Stellar library", "library", "ssp"),
    ("Nebular emission", "nebular", "nebular"),
    ("Dust attenuation", "attenuation", "dust_att"),
    ("Dust emission", "dust_ir", "dust_em"),
)


def _table_a_rows() -> list[str]:
    """Return the body rows of Table A, one block of five components per code."""
    lines = []
    for n, key in enumerate(sorted(XLIKE_CONFIGS)):
        cfg = XLIKE_CONFIGS[key]
        table1 = cfg.get("fiducial_table1", {})
        if n:
            lines.append(r"\midrule")
        for i, (comp, cfg_key, p_key) in enumerate(_COMPONENTS):
            if i == 0:
                head = f"{_escape_latex(display_code(cfg['code']))} & "
                head += "Yes" if cfg["parity_check"] else "No"
            else:
                head = " & "
            lines.append(
                f"{head} & {comp} & {_escape_latex(table1.get(p_key, '---'))} & "
                f"{_wrap(tengri_display(cfg_key, cfg[cfg_key]), _WIDTH_TENGRI)}{_ROW_END}"
            )
    return lines


def _table_b_rows() -> list[str]:
    """Return the body rows of Table B, one row per mismatch."""
    lines = []
    for n, key in enumerate(sorted(XLIKE_CONFIGS)):
        cfg = XLIKE_CONFIGS[key]
        if n:
            lines.append(r"\midrule")
        sources = cfg.get("mismatch_sources", [])
        for i, text in enumerate(cfg.get("mismatches_text", [])):
            code = _escape_latex(display_code(cfg["code"])) if i == 0 else ""
            src = source_display(sources[i])[0] if i < len(sources) else "this work"
            lines.append(
                f"{code} & {_wrap(_prose(text), _WIDTH_DIFFERENCE)} & "
                f"{_wrap(src, _WIDTH_SOURCE)}{_ROW_END}"
            )
    return lines


def render_latex() -> str:
    """Return both tables as ``table*`` blocks in the paper's house style."""
    table_a = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Configuration choices of the X-like models beside the options each code "
        r"offers, as listed in Table~1 of \citet{Pacifici_2023}. That table does not record "
        r"the configuration of the workshop runs. ``C'' is Cloudy with no version year. "
        + _PARITY_NOTE
        + "}",
        r"\label{tab:xlike_choices}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + _TABLE_A_COLUMNS + "}",
        r"\toprule",
        r"Code & Parity & Component & "
        + _wrap(r"Options in each code (\citealp[Table~1]{Pacifici_2023})", _WIDTH_OPTIONS)
        + r" & \textsc{tengri} X-like"
        + _ROW_END,
        r"\midrule",
        *_table_a_rows(),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    table_b = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Differences between each X-like configuration and the external code "
        r"it stands in for, with the source of each statement. " + _PARITY_NOTE + "}",
        r"\label{tab:xlike_differences}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + _TABLE_B_COLUMNS + "}",
        r"\toprule",
        "Code & Difference & Source" + _ROW_END,
        r"\midrule",
        *_table_b_rows(),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(table_a) + "\n\n" + "\n".join(table_b) + "\n"


def write_latex_tables(output_file: Path) -> None:
    """Write the configuration-choices and differences tables.

    The tables use ``table*`` with ``booktabs`` like the paper's own tables.
    AASTeX 6.3.1 cannot take ``p{}`` column types in any table: its column parser
    fails with a run of ``Extra \\or`` errors and no PDF is produced.

    Parameters
    ----------
    output_file : Path
        Output .tex file.
    """
    output_file.write_text(render_latex())


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
            "display_code": display_code(cfg["code"]),
            "mismatch_sources_display": [
                source_display(src)[1] for src in cfg.get("mismatch_sources", [])
            ],
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
