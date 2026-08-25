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

Why this parses instead of grepping
-----------------------------------
The previous version matched one regex against one line at a time::

    r'\\.get\\s*\\(\\s*["\\']redshift["\\']\\s*,\\s*[0-9\\-\\.e]+'

Measured, it could not see:

* **Any call wrapped across lines.** ``params.get(\\n "redshift",\\n 0.0,\\n)``
  matches on no single line. Ruff's formatter *produces* that wrapping at line
  length 99, so a compliant long line escaped the guard automatically.
* ``params.get("redshift", _DEFAULT_Z)`` — the default had to be a numeric
  literal, so a named constant or attribute passed.
* ``params.pop("redshift", 0.0)`` and ``params.setdefault("redshift", 0.0)``,
  which carry the identical hazard.

Parsing also removed three of the six allowlist entries. They existed only to
suppress matches inside *docstrings* — the old ``_is_comment_or_docstring``
heuristic tested whether a line contained ``\"\"\"``, which is true of the
delimiter line and false of every line between the delimiters. An AST does not
see docstring text at all.

What counts as a hazardous default
----------------------------------
A numeric literal, or a bare name/attribute (``_DEFAULT_Z``, ``self._z0``) that
could be one. A *computed* default is not flagged: ``float(redshift)`` sets an
actual redshift rather than substituting for a missing one, and
``fixed_values.get("redshift", ...)`` is a legitimate chain whose own inner
call is judged on its own.

Exception: two receivers legitimately carry a fallback, and are named rather
than allowlisted by file.

  1. ``ref_params`` — caller-supplied reference params, not guaranteed to
     include Fixed values. The reference z is documented at the callsite.
  2. ``fixed_values`` — read from ``spec.get_fixed_values()``, which omits a
     redshift parameter when it is FREE. Raising would break every
     free-redshift model.

Both are expressed as a rule about the receiver, not as a (file, substring)
suppression, so a new callsite of the same legitimate kind needs no edit here
and a new callsite of the dangerous kind cannot be smuggled in beside one.

Usage
-----
    python tools/check_redshift_defaults.py

Exit code 0 if no unsafe patterns are found; 1 with violations listed otherwise.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "tengri"

#: Dict methods that take (key, default) and silently substitute the default.
_HAZARDOUS_METHODS = frozenset({"get", "pop", "setdefault"})

#: Receivers whose fallback is legitimate. See the module docstring.
_EXEMPT_RECEIVERS = frozenset({"ref_params", "fixed_values"})


def _receiver_name(node: ast.expr) -> str:
    """Best-effort name of the object a method is called on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_hazardous_default(node: ast.expr) -> bool:
    """True for a default that could silently stand in for a real redshift.

    A literal number, or a bare name/attribute that could hold one. A call is
    not flagged: it computes a value from context rather than substituting a
    constant, and if it is itself a hazardous lookup it is visited separately.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_hazardous_default(node.operand)
    return isinstance(node, (ast.Name, ast.Attribute))


def find_violations(source: str) -> list[tuple[int, str]]:
    """Return (lineno, rendered call) for every unsafe redshift default."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in _HAZARDOUS_METHODS:
            continue
        if len(node.args) != 2:
            continue

        key = node.args[0]
        if not (isinstance(key, ast.Constant) and key.value == "redshift"):
            continue

        if _receiver_name(fn.value) in _EXEMPT_RECEIVERS:
            continue
        if not _is_hazardous_default(node.args[1]):
            continue

        found.append((node.lineno, ast.unparse(node)))

    return found


def main() -> int:
    violations: list[tuple[str, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        for lineno, rendered in find_violations(path.read_text(encoding="utf-8")):
            violations.append((rel, lineno, rendered))

    if not violations:
        print("check_redshift_defaults: OK — no unsafe redshift defaults found.")
        return 0

    print(
        f"check_redshift_defaults: {len(violations)} redshift default(s) "
        "should use require_redshift() instead\n"
    )
    for rel, lineno, rendered in violations:
        print(f"  {rel}:{lineno}")
        print(f"      {rendered}")
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
