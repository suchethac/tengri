#!/usr/bin/env python3
"""CI guard: dust group grammar validation across the entire repository.

Validates dust build-group dicts in docs/, reproduction/, bench/, scripts/ and
anywhere else outside src/ and tests/. Two breaking dust changes (#1989 explicit
laws, #2000 dust= retirement) left 88 broken blocks that are not caught by
existing gates because no gate reaches dict VALUES in docs/notebooks/markdown.

This tool scans:
- All .py files in the tree
- Code cells (type "code") in .ipynb files
- Fenced ```python blocks in .md and .rst files

Rules enforced:
1. dust= at all → violation "retired group"
2. emission key nested inside dust_attenuation dict → violation "retired nesting"
3. dust_attenuation with type == 'two_component': exactly one of {law} or
   {law_bc AND law_diff}. Flag one half of the pair, flag neither, flag both.
4. type == 'single_component': law required, law_bc/law_diff forbidden
5. type == 'wg00': no law keys allowed
6. type == 'none'/'off': skip entirely
7. law values that are string literals must be in the law table
8. Non-literal law values (variables, f-strings) are skipped silently

Files in an allowlist (file-level entries) are excluded entirely, styled after
tools/.numeric_guards_ledger (one relative path + reason per line).

Allowlisted files include:
- docs/dev/archive/ — retired blocks
- docs/internal/plans/ and docs/internal/specs/ — issue discussions (6 blocks)
- docs/dev/api_migration_v0.x.md — quotes retired spelling while explaining migration
- docs/dev/sed-model-components.md — references to old API

Exit code 0 when grammar is valid; 1 otherwise, listing each violation.
Supports --list mode to print all findings without failing.

Dependencies: standard library only. AST rather than grep: a dict's keys routinely
sit on different lines from its initializer, and a line-based scan misses many.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Dust attenuation laws (copied from src/tengri/components/dust/laws/_registry.py
# to keep this tool stdlib-only and CI-runnable without tengri imports).
DUST_LAWS = frozenset({
    "calzetti",
    "cardelli",
    "conroy2010",
    "d03_mwrv31",
    "hd23_mwrv31",
    "kriek_conroy",
    "leitherer02",
    "li08",
    "lmc",
    "narayanan_z",
    "noll09",
    "power_law",
    "prevot_smc",
    "reddy15",
    "salim",
    "salim_sbl18",
    "smc",
    "tea",
    "vw07_bc",
    "vw07_diff",
    "wd01_mwrv31",
    "wd01_smcbar",
})


def _load_allowlist() -> dict[str, str]:
    """Load allowlist from .dust_grammar_allowlist file."""
    allowlist_path = REPO_ROOT / "tools" / ".dust_grammar_allowlist"
    allowlist: dict[str, str] = {}
    if allowlist_path.exists():
        lines = allowlist_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " -> " in line:
                path, reason = line.split(" -> ", 1)
                allowlist[path.strip()] = reason.strip()
    return allowlist


# Files that legitimately contain old dust grammar while explaining the migration.
# Loaded from tools/.dust_grammar_allowlist file.
ALLOWLIST = _load_allowlist()


def relpath(path: Path) -> str:
    """Return path relative to repo root, posix-style."""
    return path.relative_to(REPO_ROOT).as_posix()


def is_allowlisted(path: Path) -> bool:
    """Check if a file is in the allowlist (dir or file prefix match)."""
    rel = relpath(path)
    for allowed_path in ALLOWLIST:
        if rel == allowed_path or rel.startswith(allowed_path + "/"):
            return True
    return False


def extract_python_code_cells_from_ipynb(path: Path) -> list[tuple[int, str]]:
    """Extract code cells from a Jupyter notebook.

    Returns list of (cell_index, source_text) tuples.
    Raises SyntaxError if source doesn't parse.
    """
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []

    cells = []
    cell_index = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source
            if source_text.strip():
                cells.append((cell_index, source_text))
            cell_index += 1
    return cells


def extract_fenced_python_blocks(path: Path) -> list[tuple[int, str]]:
    """Extract ```python fenced code blocks from markdown or rst.

    Returns list of (line_offset, block_text) tuples where line_offset is the
    line number where the fence begins (1-indexed, suitable for reporting).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    blocks = []
    i = 0
    block_num = 0
    while i < len(lines):
        line = lines[i]
        # Match ```python or ```{python} (common in rst)
        if line.strip().startswith("```") and ("python" in line):
            # Found opening fence
            start_line = i + 1  # 1-indexed for reporting
            i += 1
            block_lines = []
            # Collect lines until closing fence
            while i < len(lines):
                if lines[i].strip().startswith("```"):
                    # Found closing fence
                    break
                block_lines.append(lines[i])
                i += 1
            if block_lines:
                block_text = "\n".join(block_lines)
                blocks.append((start_line, block_text))
            block_num += 1
        i += 1

    return blocks


def scan_ast_for_dust_groups(tree: ast.AST) -> list[tuple[int, str, dict]]:
    """Find dust group violations in an AST tree.

    Looks for:
    - dust=... keyword arguments (both spellings of dict-key notation)
    - dust_attenuation=... and dust_emission=...

    Returns list of (line_number, violation_type, details) tuples.
    violation_type is one of:
        "dust_retired": dust= found (retired)
        "emission_in_attenuation": nested emission key (retired)
        "two_component_law_mismatch": two_component with wrong law key combination
        "single_component_law_missing": single_component without law
        "single_component_law_forbidden": single_component with law_bc/law_diff
        "unknown_law_string": law value is a known invalid string literal
    """
    violations = []

    for node in ast.walk(tree):
        # Check Call nodes for keyword arguments: dust=, dust_attenuation=, dust_emission=
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "dust" and isinstance(kw.value, ast.Dict):
                    # dust= is retired
                    violations.append((node.lineno, "dust_retired", {}))
                elif (
                    kw.arg in ("dust_attenuation", "dust_emission")
                    and isinstance(kw.value, ast.Dict)
                ):
                    v = _check_dust_dict(kw.value, kw.arg)
                    if v:
                        violations.append((node.lineno, v[0], v[1]))

        # Check Dict nodes with string keys: "dust", "dust_attenuation", "dust_emission"
        elif isinstance(node, ast.Dict):
            string_keys = {
                k.value: i
                for i, k in enumerate(node.keys)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }

            if "dust" in string_keys:
                violations.append((node.lineno, "dust_retired", {}))

            if "dust_attenuation" in string_keys:
                idx = string_keys["dust_attenuation"]
                if idx < len(node.values):
                    val = node.values[idx]
                    if isinstance(val, ast.Dict):
                        v = _check_dust_dict(val, "dust_attenuation")
                        if v:
                            violations.append((node.lineno, v[0], v[1]))

    return violations


def _check_dust_dict(
    dict_node: ast.Dict, parent_key: str
) -> tuple[str, dict] | None:
    """Check a dust_attenuation or dust_emission dict for violations.

    Returns (violation_type, details) or None if valid.
    """
    string_keys = {
        k.value: i
        for i, k in enumerate(dict_node.keys)
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }

    # Check for nested emission (old dust_attenuation={'emission': ...})
    if parent_key == "dust_attenuation" and "emission" in string_keys:
        return ("emission_in_attenuation", {})

    # Get the type value
    type_val = None
    if "type" in string_keys:
        idx = string_keys["type"]
        if idx < len(dict_node.values):
            type_node = dict_node.values[idx]
            if isinstance(type_node, ast.Constant) and isinstance(type_node.value, str):
                type_val = type_node.value

    # Skip none and off types
    if type_val in ("none", "off"):
        return None

    # Check wg00 type
    if type_val == "wg00":
        # wg00 should not have law keys
        law_keys = {"law", "law_bc", "law_diff"}
        if law_keys & set(string_keys):
            forbidden = law_keys & set(string_keys)
            return ("wg00_with_law", {"keys": forbidden})
        return None

    # Check single_component type
    if type_val == "single_component":
        has_law = "law" in string_keys
        has_bc_diff = "law_bc" in string_keys or "law_diff" in string_keys
        if not has_law:
            return ("single_component_law_missing", {})
        if has_bc_diff:
            return ("single_component_law_forbidden", {"keys": {"law_bc", "law_diff"}})
        # Validate law value if it's a string literal
        if has_law:
            idx = string_keys["law"]
            if idx < len(dict_node.values):
                law_node = dict_node.values[idx]
                if isinstance(law_node, ast.Constant) and isinstance(law_node.value, str):
                    law_val = law_node.value
                    if law_val not in DUST_LAWS:
                        return ("unknown_law_string", {"law": law_val})
        return None

    # Check two_component type
    if type_val == "two_component":
        has_law = "law" in string_keys
        has_bc = "law_bc" in string_keys
        has_diff = "law_diff" in string_keys
        has_bc_diff = has_bc or has_diff

        # Exactly one of {law} or {law_bc AND law_diff} is required
        if has_law and has_bc_diff:
            # Both forms present - violation
            return ("two_component_both_forms", {"keys": {"law", "law_bc", "law_diff"}})
        elif has_bc and not has_diff:
            # One half of the pair
            return ("two_component_incomplete_pair", {"keys": {"law_bc"}})
        elif has_diff and not has_bc:
            # One half of the pair
            return ("two_component_incomplete_pair", {"keys": {"law_diff"}})
        elif not has_law and not has_bc_diff:
            # Neither form
            return ("two_component_law_missing", {})

        # Validate law values if they're string literals
        for law_key in ["law", "law_bc", "law_diff"]:
            if law_key in string_keys:
                idx = string_keys[law_key]
                if idx < len(dict_node.values):
                    law_node = dict_node.values[idx]
                    if isinstance(law_node, ast.Constant) and isinstance(law_node.value, str):
                        law_val = law_node.value
                        if law_val not in DUST_LAWS:
                            return ("unknown_law_string", {"law": law_val})

        return None

    # For other types (dust emission types, etc.), just validate law if present
    if "law" in string_keys:
        idx = string_keys["law"]
        if idx < len(dict_node.values):
            law_node = dict_node.values[idx]
            if isinstance(law_node, ast.Constant) and isinstance(law_node.value, str):
                law_val = law_node.value
                if law_val not in DUST_LAWS:
                    return ("unknown_law_string", {"law": law_val})

    return None


def scan_python_file(path: Path) -> list[tuple[int, str, dict]]:
    """Scan a .py file for dust group violations."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    return scan_ast_for_dust_groups(tree)


def scan_notebook_file(path: Path) -> list[tuple[int, str, dict]]:
    """Scan code cells in an .ipynb file."""
    violations = []
    cells = extract_python_code_cells_from_ipynb(path)
    for cell_idx, source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Silently skip unparseable cells
            continue
        cell_violations = scan_ast_for_dust_groups(tree)
        # Augment with cell info so we can report it
        for _lineno, vtype, details in cell_violations:
            # Report cell index as the source (line numbers within cells are less useful)
            violations.append((cell_idx, vtype, details))
    return violations


def scan_markdown_file(path: Path) -> list[tuple[int, str, dict]]:
    """Scan fenced ```python blocks in .md or .rst files."""
    violations = []
    blocks = extract_fenced_python_blocks(path)
    for line_offset, source in blocks:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Silently skip unparseable blocks
            continue
        block_violations = scan_ast_for_dust_groups(tree)
        for _lineno, vtype, details in block_violations:
            # Report using the fence's line offset
            violations.append((line_offset, vtype, details))
    return violations


def format_violation(file_path: Path, lineno: int, vtype: str, details: dict) -> str:
    """Format a violation message."""
    rel = relpath(file_path)

    messages = {
        "dust_retired": f"{rel}:{lineno}  dust= (retired group)",
        "emission_in_attenuation": (
            f"{rel}:{lineno}  dust_attenuation with nested 'emission' key (retired)"
        ),
        "two_component_law_missing": (
            f"{rel}:{lineno}  dust_attenuation type='two_component' missing law "
            f"({{'law'}} or {{'law_bc', 'law_diff'}})"
        ),
        "two_component_incomplete_pair": (
            f"{rel}:{lineno}  dust_attenuation type='two_component' with "
            "incomplete per-screen law pair"
        ),
        "two_component_both_forms": (
            f"{rel}:{lineno}  dust_attenuation type='two_component' with both "
            "'law' and 'law_bc'/'law_diff'"
        ),
        "single_component_law_missing": (
            f"{rel}:{lineno}  dust_attenuation type='single_component' "
            "missing required 'law' key"
        ),
        "single_component_law_forbidden": (
            f"{rel}:{lineno}  dust_attenuation type='single_component' "
            "with forbidden 'law_bc'/'law_diff'"
        ),
        "unknown_law_string": (
            f"{rel}:{lineno}  dust law '{details.get('law', '?')}' "
            "not in valid law set"
        ),
        "wg00_with_law": (
            f"{rel}:{lineno}  dust_attenuation type='wg00' "
            "with forbidden law keys"
        ),
    }
    return messages.get(vtype, f"{rel}:{lineno}  {vtype}")


def main(list_only: bool = False) -> int:
    """Run the guard. Return 0 if valid, 1 if violations found."""
    violations = []
    seen_allowlist = set()

    # Build paths to skip (in repo root)
    skip_dirs = {
        REPO_ROOT / ".git",
        REPO_ROOT / ".venv",
        REPO_ROOT / "node_modules",
        REPO_ROOT / ".claude",
        REPO_ROOT / "build",
        REPO_ROOT / "dist",
    }

    # Walk all files in the repo
    for path in sorted(REPO_ROOT.rglob("*")):
        # Skip directories
        if path.is_dir():
            continue

        # Skip files in excluded directories or in __pycache__
        if any(str(path).startswith(str(skip_dir)) for skip_dir in skip_dirs):
            continue
        if "__pycache__" in path.parts:
            continue

        # Skip src/ and tests/ — they were swept in #2000
        if "src/tengri" in path.as_posix() or "tests/" in path.as_posix():
            continue

        # Check allowlist (before scanning, for efficiency)
        if is_allowlisted(path):
            seen_allowlist.add(relpath(path))
            continue

        # Process by file type
        if path.suffix == ".py":
            path_violations = scan_python_file(path)
        elif path.suffix == ".ipynb":
            path_violations = scan_notebook_file(path)
        elif path.suffix in (".md", ".rst"):
            path_violations = scan_markdown_file(path)
        else:
            continue

        for lineno, vtype, details in path_violations:
            msg = format_violation(path, lineno, vtype, details)
            violations.append(msg)

    if violations:
        if not list_only:
            print("Dust group grammar violations:\n", file=sys.stderr)
        for violation in sorted(violations):
            print(f"  {violation}", file=sys.stderr)
        if not list_only:
            print(f"\n{len(violations)} violation(s) found.", file=sys.stderr)
        return 1 if not list_only else 0

    # Success message
    print(
        f"OK: all dust group grammars are valid. {len(seen_allowlist)} file(s) allowlisted."
    )
    return 0


if __name__ == "__main__":
    list_only = "--list" in sys.argv
    sys.exit(main(list_only=list_only))
