#!/usr/bin/env python3
"""Audit AGN, IGM, radio, xray components for closure-captured large constants.

Scans precompute/component files for jnp.array() calls outside of @jax.jit
functions and estimates their sizes.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
COMPONENTS_DIR = REPO / "src" / "tengri" / "components"


class ConstantInfo(NamedTuple):
    """Tracked closure-captured constant."""

    file: str
    func_name: str
    line: int
    size_estimate: str
    shape_expr: str


def _parse_file(filepath: str) -> list[ConstantInfo]:
    """Parse Python file for closure-captured jnp.array calls."""
    results = []

    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return results

    # Find all function definitions and their scope
    tree = ast.parse(content, filename=filepath)

    # Track @jax.jit decorators
    jitted_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Attribute):
                        if dec.func.attr == "jit":
                            jitted_funcs.add(node.name)
                elif isinstance(dec, ast.Attribute):
                    if dec.attr == "jit":
                        jitted_funcs.add(node.name)
                elif isinstance(dec, ast.Name):
                    if dec.id == "jit":
                        jitted_funcs.add(node.name)

    # Scan for jnp.array calls outside of jitted functions
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        # Match jnp.array, np.array (after import), etc.
        if re.search(r"\bjnp\.array\s*\(", line):
            # Estimate size if we can see a source
            match = re.search(r"jnp\.array\(([^)]+)\)", line)
            if match:
                source = match.group(1)
                # Try to guess from context
                shape_expr = "unknown"
                size_est = "<1MB"

                if "raw[" in source or "data[" in source or "grid" in source.lower():
                    size_est = "LARGE (templates/grids)"
                    if "skirtor" in filepath.lower() or "disc" in filepath.lower():
                        shape_expr = "~5D x n_wave"
                    elif "cat3d" in filepath.lower():
                        shape_expr = "~3D x n_wave"

                results.append(
                    ConstantInfo(
                        file=filepath.replace(str(COMPONENTS_DIR), ""),
                        func_name="<module>",
                        line=i,
                        size_estimate=size_est,
                        shape_expr=shape_expr,
                    )
                )

    return results


def main() -> int:
    """Audit AGN, IGM, radio, xray for closure constants."""
    audit_dirs = [
        COMPONENTS_DIR / "agn",
        COMPONENTS_DIR / "igm",
        COMPONENTS_DIR / "radio",
        COMPONENTS_DIR / "xray",
    ]

    all_results = []
    for audit_dir in audit_dirs:
        for py_file in sorted(audit_dir.glob("*.py")):
            results = _parse_file(str(py_file))
            all_results.extend(results)

    if not all_results:
        print("No obvious closure-captured arrays found via regex.")
        print("\nKey files to inspect manually:")
        print("  - src/tengri/components/agn/skirtor.py:237-239 (grid_jax = jnp.array)")
        print("  - src/tengri/components/agn/disc.py:~1000+ (templates in closures)")
        print("  - src/tengri/components/agn/_nthcomp.py:60-65 (X-ray grids)")
        print("  - src/tengri/components/igm/igm.py:27-70 (Lyman series coefficients)")
        return 0

    print("| File | Func | Line | Size Estimate | Shape |")
    print("|------|------|------|---------------|-------|")
    for r in all_results:
        print(f"| {r.file} | {r.func_name} | {r.line} | {r.size_estimate} | {r.shape_expr} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
