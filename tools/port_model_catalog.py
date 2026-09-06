#!/usr/bin/env python
r"""
Port the tengri Model Catalog from LaTeX to MyST Markdown.

Reads the LaTeX sources from a catalog directory, runs pandoc with
citeproc, and post-processes the output to create MyST Markdown pages.

Usage:
    python port_model_catalog.py [--catalog DIR] [--out DIR] [--pandoc PATH]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversionStats:
    """Track conversion statistics per page."""

    page_name: str
    source_equations: int = 0
    output_equations: int = 0
    labels_written: int = 0
    references_rewritten: int = 0
    dashes_replaced: int = 0
    aliases: list[tuple[str, str]] = field(default_factory=list)
    spans_unwrapped: int = 0
    raw_word_count: int = 0  # word count from pandoc raw output
    final_word_count: int = 0  # word count after post-processing


def extract_macros(ms_tex_path: Path) -> str:
    r"""Extract newcommand/renewcommand/providecommand lines from 0-ms.tex."""
    with open(ms_tex_path) as f:
        content = f.read()

    lines = []
    for line in content.split("\n"):
        if re.search(r"\\(?:new|renew|provide)command", line):
            lines.append(line)

    return "\n".join(lines) + "\n"


def run_pandoc(
    pandoc_path: str, catalog_dir: Path, part_tex: str, macro_text: str, bibliography_path: Path
) -> str:
    """Run pandoc to convert LaTeX to Markdown."""
    with open(catalog_dir / part_tex) as f:
        input_text = macro_text + f.read()

    cmd = [
        pandoc_path,
        "-f",
        "latex",
        "-t",
        (
            "markdown+tex_math_dollars-raw_tex-raw_html-grid_tables"
            "-multiline_tables-simple_tables+pipe_tables"
        ),
        "--wrap=none",
        "--markdown-headings=atx",
        "--citeproc",
        f"--bibliography={bibliography_path}",
    ]

    try:
        result = subprocess.run(cmd, input=input_text, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc for {part_tex}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def process_equation_blocks(text: str, stats: ConversionStats) -> str:
    r"""
    Process $$ ... $$ blocks: put label on closing $$ line, no separate target line.
    Track aliases for later rewriting.
    """
    result = []
    pos = 0
    # Track equation aliases: maps old name -> first label in block
    aliases: dict[str, str] = {}

    while True:
        start = text.find("$$", pos)
        if start == -1:
            result.append(text[pos:])
            break

        end = text.find("$$", start + 2)
        if end == -1:
            result.append(text[pos:])
            break

        result.append(text[pos:start])

        content = text[start + 2 : end]

        # Look for ALL equation labels (for multi-label blocks)
        labels = re.findall(r"\\label\{eq:([^}]+)\}", content)

        if labels:
            # Use the first label as the primary one
            primary_label = labels[0]
            myst_label = primary_label.replace(":", "-").replace("_", "-")

            # Track aliases for other labels in this block
            for label in labels[1:]:
                myst_alias = label.replace(":", "-").replace("_", "-")
                aliases[myst_alias] = myst_label
                stats.aliases.append((myst_alias, myst_label))

            # Remove all labels from content
            content = re.sub(r"\\label\{eq:[^}]+\}", "", content).strip()

            # Strip equation environment markers
            content = re.sub(r"\\begin\{equation\*?\}", "", content)
            content = re.sub(r"\\end\{equation\*?\}", "", content)

            # Convert align to aligned
            content = re.sub(r"\\begin\{align\*?\}", r"\\begin{aligned}", content)
            content = re.sub(r"\\end\{align\*?\}", r"\\end{aligned}", content)

            content = content.strip()
            stats.labels_written += 1

            # Put label on closing $$ line: $$ (eq-NAME)
            # Ensure it's on its own line with nothing after (add newline after label)
            result.append(f"$${content}\n$$ (eq-{myst_label})\n")
        else:
            # Strip equation environment markers even without labels
            content = re.sub(r"\\begin\{equation\*?\}", "", content)
            content = re.sub(r"\\end\{equation\*?\}", "", content)
            content = re.sub(r"\\begin\{align\*?\}", r"\\begin{aligned}", content)
            content = re.sub(r"\\end\{align\*?\}", r"\\end{aligned}", content)
            content = content.strip()

            result.append(f"$${content}$$")

        pos = end + 2

    return "".join(result)


def process_headings_with_labels(text: str) -> str:
    r"""Convert ## Title {#app:NAME} to (app-NAME)= target line + heading."""
    heading_pattern = r"^(#{1,6}) (.+?)\s*\{#app:([^}]+)\}$"

    def replace_heading(match):
        hashes = match.group(1)
        title = match.group(2)
        label_name = match.group(3)

        myst_label = label_name.replace(":", "-").replace("_", "-")
        return f"(app-{myst_label})=\n\n{hashes} {title}"

    text = re.sub(heading_pattern, replace_heading, text, flags=re.MULTILINE)
    return text


def process_table_labels(text: str) -> str:
    r"""Convert ::: {#tab:NAME} to (tab-NAME)= target line."""

    def replace_table_label(match):
        label_name = match.group(1)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        return f"(tab-{myst_label})="

    text = re.sub(r"^::: \{#tab:([^}]+)\}$", replace_table_label, text, flags=re.MULTILINE)
    text = re.sub(r"(\(tab-[^=]*\)=)\n([^(\n])", r"\1\n\n\2", text)

    return text


def process_references(text: str, stats: ConversionStats) -> str:
    r"""Convert pandoc reference spans to MyST roles."""
    # Match references by looking for the URL pattern first
    # Then backtrack to find the opening bracket

    # Handle equation references with specific URL pattern: ](#eq:NAME){...}
    # The link text is everything from [ to ], but we need to be careful with escaped brackets
    # For equations, link text is typically [\[eq:name\]] with escaped brackets
    eq_ref_pattern = r"\[\\\[([^\\]*)\\\]\]\(#(eq):([^)]+)\)\{[^}]*\}"

    def replace_eq_ref(match):
        label_name = match.group(3)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        stats.references_rewritten += 1
        return "{eq}`eq-" + myst_label + "`"

    text = re.sub(eq_ref_pattern, replace_eq_ref, text)

    # Handle table references: [TEXT](#tab:NAME){...} -> {ref}`TEXT <tab-NAME>`
    tab_ref_pattern = r"\[([^\]]*)\]\(#(tab):([^)]+)\)\{[^}]*\}"

    def replace_tab_ref(match):
        link_text = match.group(1)
        label_name = match.group(3)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        stats.references_rewritten += 1
        return "{ref}`" + link_text + " <tab-" + myst_label + ">`"

    text = re.sub(tab_ref_pattern, replace_tab_ref, text)

    # Handle app references with escaped brackets: §[\[app:NAME\]](...) -> {ref}`app-NAME`
    app_ref_escaped = r"§\s*\[\\\[([^\\]*)\\\]\]\(#(app):([^)]+)\)\{[^}]*\}"

    def replace_app_escaped(match):
        label_name = match.group(3)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        stats.references_rewritten += 1
        return "{ref}`app-" + myst_label + "`"

    text = re.sub(app_ref_escaped, replace_app_escaped, text)

    # Handle app references with § prefix (simple text): §[TEXT](#app:NAME){...} -> {ref}`app-NAME`
    app_ref_pattern = r"§\s*\[([^\]]*)\]\(#(app):([^)]+)\)\{[^}]*\}"

    def replace_app_ref(match):
        label_name = match.group(3)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        stats.references_rewritten += 1
        return "{ref}`app-" + myst_label + "`"

    text = re.sub(app_ref_pattern, replace_app_ref, text)

    # Handle app without § prefix with escaped brackets: [\[app:NAME\]](...) -> {ref}
    app_ref_no_sect_escaped = r"\[\\\[([^\\]*)\\\]\]\(#(app):([^)]+)\)\{[^}]*\}"

    def replace_app_no_sect_escaped(match):
        label_name = match.group(3)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        stats.references_rewritten += 1
        return "{ref}`app-" + myst_label + "`"

    text = re.sub(app_ref_no_sect_escaped, replace_app_no_sect_escaped, text)

    # Handle app references without § prefix (simple text): [...](...) -> {ref}`app-NAME`
    app_ref_no_sect = r"\[([^\]]*)\]\(#(app):([^)]+)\)\{[^}]*\}"

    def replace_app_no_sect(match):
        label_name = match.group(3)
        myst_label = label_name.replace(":", "-").replace("_", "-")
        stats.references_rewritten += 1
        return "{ref}`app-" + myst_label + "`"

    text = re.sub(app_ref_no_sect, replace_app_no_sect, text)

    return text


def process_dashes(text: str, stats: ConversionStats) -> str:
    r"""Replace em dash U+2014 and literal ' --- ' with ', '."""
    count = 0
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        # Skip table alignment rows (start with |)
        if line.startswith("|"):
            result_lines.append(line)
            continue

        # Count and replace em dashes (U+2014)
        if "\u2014" in line:
            count += line.count("\u2014")
            line = line.replace("\u2014", ", ")

        # Count and replace literal ' --- ' (no spaces)
        if " --- " in line:
            count += line.count(" --- ")
            line = line.replace(" --- ", ", ")

        # Replace --- (three hyphens) not preceded/followed by spaces
        if "---" in line:
            count += line.count("---")
            line = line.replace("---", ", ")

        result_lines.append(line)

    text = "\n".join(result_lines)

    # Clean up doubled commas
    while ",," in text:
        text = text.replace(",,", ",")

    text = text.replace(",,.", ".")
    text = text.replace(",.", ".")

    stats.dashes_replaced = count
    return text


def process_div_fences(text: str) -> str:
    r"""Remove pandoc div fence lines. Convert {#refs} to ## References heading.

    Removes any line matching ^:{3,}(\s*\{[^}]*\})?\s*$ (3+ colons, optional
    whitespace, optional braced attributes, optional trailing whitespace).
    Convert the top-level {#refs} fence to ## References heading.
    """
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        # Match bare fences: 3+ colons, optionally followed by whitespace only
        if re.match(r"^:{3,}(\s*(\{[^}]*\}|[A-Za-z][\w-]*))?\s*$", line):
            # If this is the top-level refs fence opening, output the heading instead
            if "{#refs" in line:
                result_lines.append("## References")
                result_lines.append("")
            # Otherwise skip fence lines entirely (bibliography entry boundaries)
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def ensure_equation_spacing(text: str) -> str:
    r"""Ensure blank lines before opening $$ and after closing $$ (eq-NAME)."""
    lines = text.split("\n")
    result_lines = []

    for i, line in enumerate(lines):
        # Check if this line starts with $$
        if line.startswith("$$"):
            # If not the first line and previous line is not blank, add blank line
            if result_lines and result_lines[-1].strip():
                result_lines.append("")
            result_lines.append(line)
        else:
            result_lines.append(line)
            # Check if we just added a closing $$ (eq-NAME) line
            if (
                re.match(r"^\$\$ \(eq-[a-z0-9-]+\)$", line)
                and i + 1 < len(lines)
                and lines[i + 1].strip()
            ):
                result_lines.append("")

    return "\n".join(result_lines)


def process_remnants(text: str, stats: ConversionStats, page_name: str) -> str:
    r"""Remove LaTeX remnants and unwrap spans."""
    # Unwrap [text]{...}
    span_pattern = r"\[([^\]]+)\]\{[^}]*\}"
    matches = re.findall(span_pattern, text)
    for _ in matches:
        stats.spans_unwrapped += 1
    text = re.sub(span_pattern, r"\1", text)

    # Remove Pandoc raw HTML markers {=html} and their preceding content
    # These are Pandoc's way of marking raw HTML blocks; we strip the markers
    text = re.sub(r"`<!-- -->`\{=html\}", "", text)
    text = text.replace("{=html}", "")

    # Remove custom commands
    text = re.sub(r"\\confirm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\fix\{([^}]*)\}", r"\1", text)
    text = text.replace("\\noindent", "")
    text = text.replace("\\centering", "")

    # Check for unprocessed LaTeX outside math
    parts = text.split("$$")
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if re.search(r"\\begin\{(?!equation|align|aligned|cases)[^}]+\}", part):
                match = re.search(r"\\begin\{(?!equation|align|aligned|cases)([^}]+)\}", part)
                raise ValueError(f"Unprocessed LaTeX in {page_name}: \\begin{{{match.group(1)}}}")
            if re.search(r"\\end\{(?!equation|align|aligned|cases)[^}]+\}", part):
                match = re.search(r"\\end\{(?!equation|align|aligned|cases)([^}]+)\}", part)
                raise ValueError(f"Unprocessed LaTeX in {page_name}: \\end{{{match.group(1)}}}")

    return text


def count_source_equations(catalog_dir: Path, part_tex: str) -> int:
    """Count display equations in LaTeX source."""
    with open(catalog_dir / part_tex) as f:
        content = f.read()

    count = 0
    count += len(re.findall(r"\\begin\{equation", content))
    count += len(re.findall(r"\\begin\{align", content))

    return count


def create_index_md(parts: list[str]) -> str:
    """Create index.md for model_reference."""
    toctree_entries = "\n".join(parts)

    return f"""# Model Reference

The tengri Model Catalog describes the physical components and their
mathematical formulations that are implemented in the code.

The tables in [components](../components.md) hold the live registry of every model
variant and its configuration grammar; this section provides the physics
and equations for each model block. These pages are the
canonical reference, the catalog is maintained alongside the code
repository and rendered in this documentation, the same content as the
versioned document.

The complete tengri Model Catalog is also published as a standalone
versioned document.

```{{toctree}}
:maxdepth: 2

{toctree_entries}
```
"""


def port_catalog(catalog_dir: Path, out_dir: Path, pandoc_path: str) -> None:
    """Main conversion function."""
    catalog_dir = Path(catalog_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        ("1-introduction.tex", "introduction"),
        ("2-sps.tex", "sps"),
        ("3-metallicity.tex", "metallicity"),
        ("4-sfh.tex", "sfh"),
        ("5-dust.tex", "dust"),
        ("6-nebular.tex", "nebular"),
        ("7-agn.tex", "agn"),
        ("8-xray-radio.tex", "xray_radio"),
        ("9-igm.tex", "igm"),
        ("10-observation.tex", "observation"),
        ("11-computation.tex", "computation"),
        ("12-parameters.tex", "parameters"),
    ]

    bibliography_path = catalog_dir / "99-references.bib"

    print("Extracting macros from 0-ms.tex...")
    macros = extract_macros(catalog_dir / "0-ms.tex")

    all_stats = []
    all_labels: dict[str, str] = {}
    all_references: set[str] = set()
    all_aliases: list[tuple[str, str]] = []

    print("Converting LaTeX to Markdown...")
    for source_file, page_name in parts:
        print(f"  Processing {page_name}...")
        stats = ConversionStats(page_name=page_name)

        stats.source_equations = count_source_equations(catalog_dir, source_file)

        md_text = run_pandoc(pandoc_path, catalog_dir, source_file, macros, bibliography_path)

        # Count words in raw pandoc output (before post-processing), without the
        # div fence lines that the post-processor deletes by design
        fence_free = re.sub(r"(?m)^:{3,}(\s*(\{[^}]*\}|[A-Za-z][\w-]*))?\s*$", "", md_text)
        stats.raw_word_count = len(fence_free.split())

        stats.output_equations = len(re.findall(r"\$\$", md_text)) // 2

        # Post-process in order
        md_text = process_equation_blocks(md_text, stats)

        # Collect aliases from this page
        all_aliases.extend(stats.aliases)

        md_text = process_headings_with_labels(md_text)
        md_text = process_table_labels(md_text)
        md_text = process_references(md_text, stats)
        md_text = process_dashes(md_text, stats)
        md_text = process_div_fences(md_text)
        md_text = ensure_equation_spacing(md_text)
        md_text = process_remnants(md_text, stats, page_name)

        # Count words in final output
        stats.final_word_count = len(md_text.split())

        # Check word retention (must be at least 97%)
        if stats.raw_word_count > 0:
            retention = stats.final_word_count / stats.raw_word_count
            if retention < 0.97:
                loss = 100 * (1 - retention)
                print(f"  ERROR: {page_name} lost {loss:.1f}% of words", file=sys.stderr)
                msg = f"Raw: {stats.raw_word_count} → Final: {stats.final_word_count}"
                print(f"    {msg}", file=sys.stderr)
                sys.exit(1)

        # Extract labels
        for match in re.finditer(r"\$\$ \(eq-([a-z0-9-]+)\)$", md_text, re.MULTILINE):
            label = "eq-" + match.group(1)
            all_labels[label] = page_name

        for match in re.finditer(r"\(tab-([a-z0-9-]+)\)=$", md_text, re.MULTILINE):
            label = "tab-" + match.group(1)
            all_labels[label] = page_name

        for match in re.finditer(r"\(app-([a-z0-9-]+)\)=$", md_text, re.MULTILINE):
            label = "app-" + match.group(1)
            all_labels[label] = page_name

        # Extract references
        for match in re.finditer(r"\{eq\}`([^`]+)`", md_text):
            all_references.add(match.group(1))

        for match in re.finditer(r"\{ref\}`([^`<]+)<([^>]+)>`", md_text):
            all_references.add(match.group(2))

        for match in re.finditer(r"\{ref\}`(app-[^`]+)`", md_text):
            all_references.add(match.group(1))

        out_file = out_dir / f"{page_name}.md"
        with open(out_file, "w") as f:
            f.write(md_text)

        all_stats.append(stats)

    # Check cross-references
    print("\nChecking cross-references...")
    undefined_refs = all_references - set(all_labels.keys())
    if undefined_refs:
        print("ERROR: Undefined references:", file=sys.stderr)
        for ref in sorted(undefined_refs):
            print(f"  {ref}", file=sys.stderr)
        sys.exit(1)

    # Write index.md
    page_names = [name for _, name in parts]
    index_content = create_index_md(page_names)
    with open(out_dir / "index.md", "w") as f:
        f.write(index_content)

    # Print report
    print("\n" + "=" * 70)
    print("CONVERSION REPORT")
    print("=" * 70)

    print("\nPER-PAGE WORD COUNTS (raw → final):")
    for stats in all_stats:
        if stats.raw_word_count > 0:
            retention = 100 * stats.final_word_count / stats.raw_word_count
        else:
            retention = 100
        msg = f"{stats.page_name}: {stats.raw_word_count} → {stats.final_word_count}"
        print(f"  {msg} ({retention:.1f}%)")

    for stats in all_stats:
        print(f"\n{stats.page_name}:")
        print(f"  Source equations: {stats.source_equations}")
        print(f"  Output $$ blocks: {stats.output_equations}")
        print(f"  Labels written: {stats.labels_written}")
        print(f"  References rewritten: {stats.references_rewritten}")
        print(f"  Dashes replaced: {stats.dashes_replaced}")
        print(f"  Spans unwrapped: {stats.spans_unwrapped}")

    if all_aliases:
        print("\n" + "=" * 70)
        print("EQUATION ALIASES (maps old name -> primary label)")
        print("=" * 70)
        for old_name, primary in sorted(set(all_aliases)):
            print(f"  {old_name} -> {primary}")

    print("\n" + "=" * 70)
    print("Output written to:", out_dir)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Port tengri Model Catalog from LaTeX to MyST Markdown"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=os.environ.get("TENGRI_MODEL_CATALOG_DIR"),
        help=(
            "Path to the Model Catalog LaTeX directory (default: the "
            "TENGRI_MODEL_CATALOG_DIR environment variable)"
        ),
    )
    parser.add_argument("--out", type=str, default="docs/model_reference", help="Output directory")
    parser.add_argument(
        "--pandoc",
        type=str,
        default=shutil.which("pandoc") or "pandoc",
        help="Path to the pandoc executable (default: the one on PATH)",
    )

    args = parser.parse_args()
    if not args.catalog:
        parser.error("--catalog is required (or set TENGRI_MODEL_CATALOG_DIR)")

    port_catalog(Path(args.catalog), Path(args.out), args.pandoc)


if __name__ == "__main__":
    main()
