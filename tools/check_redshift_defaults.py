#!/usr/bin/env python3
"""CI guard: redshift defaults must use the canonical accessor.

Motivation
----------
A `params.get("redshift", default)` with a numeric default puts a galaxy at
the default redshift when ``redshift`` is absent. Since a Fixed redshift is
legitimate (it is not required to be present in the params dict), the default
is semantically invisible: it silently activates when the caller did not expect
it to.

The issue is not theoretical. With ``redshift=Fixed(0.5)`` and a params dict that
correctly omits it, bare ``params.get("redshift", 0.0)`` computes the flux at
10 pc instead of z = 0.5 — ~16 orders of magnitude off, with no warning.

Solution: use :func:`tengri.parameters.resolve.require_redshift(params, where)`
instead, which raises ``KeyError`` when redshift is absent. Every params dict
reaching a point that needs redshift has already passed one of two boundaries:

  * :class:`~tengri.forward.prediction.Prediction`, which sets ``_params =
    resolve_fixed_params(model, params)``, or
  * the forward pipeline, which merges ``{**fixed_values, **params}`` before
    any component runs.

So the default is unreachable — which makes it dangerous. A default that
cannot be reached is not a safety net; it is a silencer for the one condition
worth hearing about.

Exception: some callsites legitimately carry a fallback. Two kinds:

  1. ``ref_params.get("redshift", ...)`` — caller-supplied reference params,
     not guaranteed to include Fixed values. Document the reference z choice.
  2. ``fixed_values.get("redshift", ...)`` — reading from ``spec.get_fixed_values()``,
     which omits a redshift parameter when it is FREE. Raising breaks every
     free-redshift model.

Usage
-----
    python tools/check_redshift_defaults.py

Exit code 0 if no unsafe patterns are found; 1 with violations listed otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "src" / "tengri"

#: Files and patterns that are deliberately safe. Each entry is:
#: (file_path_relative, line_pattern_substring) → reason
SAFE_PATTERNS: dict[tuple[str, str], str] = {
    # Nebular precompute reads caller-supplied reference params, not guaranteed
    # to include Fixed values. The reference z is documented in the function.
    (
        "components/nebular/line_precompute.py",
        "ref_params.get",
    ): "caller-supplied reference params; documented reference z",
    (
        "components/nebular/nebular_grid_precompute.py",
        "ref_params.get",
    ): "caller-supplied reference params; documented reference z",
    # Likelihood correctly chains params → fixed_values → 0.0, with the
    # fallback only reached when redshift is not in either place.
    (
        "inference/likelihood.py",
        'params.get("redshift", fixed_values.get',
    ): "correctly chains params → fixed_values before 0.0 fallback",
    # Docstring illustrations and assertions, not live code paths
    (
        "presets/param_presets.py",
        'params.get("redshift")',
    ): "docstring assertion, not a live default path",
    # Documentation and examples in resolve.py itself
    ("parameters/resolve.py", 'params.get("redshift"'): "documentation, not a live code path",
    # Documentation of the problematic pattern in prediction.py
    (
        "forward/prediction.py",
        'params.get("redshift", 0.0)',
    ): "docstring documentation of the problematic pattern in #1097/#1124/#1127",
}


def _is_comment_or_docstring(path: Path, lineno: int) -> bool:
    """Check if a line is in a comment or docstring."""
    text = path.read_text()
    lines = text.splitlines()
    if lineno > len(lines) or lineno < 1:
        return False

    line = lines[lineno - 1]
    # Skip comment lines
    if line.strip().startswith("#"):
        return True

    # Skip lines that contain docstring triple quotes
    return '"""' in line or "'''" in line


def _is_redshift_get_pattern(code_line: str) -> bool:
    """Check if a line contains a redshift .get() call with a numeric default."""
    return bool(
        re.search(
            r'\.get\s*\(\s*["\']redshift["\']\s*,\s*[0-9\-\.e]+',
            code_line,
        )
    )


def _get_pattern_context(path: Path, lineno: int) -> str:
    """Return the code line for context."""
    try:
        lines = path.read_text().splitlines()
        if 0 < lineno <= len(lines):
            return lines[lineno - 1].strip()
        return ""
    except Exception:
        return ""


def main() -> int:
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            if not _is_redshift_get_pattern(line):
                continue

            # Skip comments and docstrings
            if _is_comment_or_docstring(path, lineno):
                continue

            rel = path.relative_to(ROOT)
            rel_str = str(rel)

            # Check if this is a deliberately-safe pattern
            is_safe = False
            for (safe_file, safe_pattern), _reason in SAFE_PATTERNS.items():
                if safe_file in rel_str and safe_pattern in line:
                    is_safe = True
                    break

            if is_safe:
                continue

            context = _get_pattern_context(path, lineno)
            violations.append((rel_str, lineno, context))

    if not violations:
        print("check_redshift_defaults: OK — no unsafe redshift defaults found.")
        return 0

    print(
        f"check_redshift_defaults: {len(violations)} redshift default(s) "
        "should use require_redshift() instead\n"
    )
    for rel, lineno, context in violations:
        print(f"  {rel}:{lineno}")
        print(f"      {context}")
        print()
    print(
        "A redshift default in params.get() is unreachable in live code paths "
        "and silences the one condition worth hearing about. Use "
        "require_redshift(params, where) instead, which raises KeyError when "
        "redshift is absent.\n"
        "  from tengri.parameters.resolve import require_redshift\n"
        "  z = require_redshift(params, 'forward/sed_model.py:3075')\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
