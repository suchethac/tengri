#!/usr/bin/env python3
"""CI guard: a signature default must lie inside its parameter's declared prior.

ADR-0011 makes the prior object a parameter's single source of truth. A model
function that repeats the value as a literal — ``agn_log_lbol: float = 45.0`` —
holds a second copy, and the two drift. This guard catches the drift that
matters: a default *outside* the declared prior's support.

Such a default is wrong twice over. No fit can reach it, because the sampler is
confined to the prior, so it is dead on the inference path and only fires for
someone calling the model function directly. And it is usually the tell of a
unit confusion: nine AGN entry points shipped ``agn_log_lbol=45.0`` — the
``log10(erg/s)`` magnitude — against a declaration in ``log10(L/L_sun)``, so a
bare call returned an SED ~1e33 too luminous (3.8e78 erg/s, some 30 dex past
the brightest quasar ever observed) and overflowed float32 besides. Half that
family was repaired in #1200, half survived to #1560 because the fix pinned
instances rather than the rule.

The guard deliberately does *not* flag a default that merely differs from the
declared default while staying in support — ``dust_tau_bc=0.0`` in a mock
generator is a deliberate "no dust" choice, not a bug.

Usage
-----
    python tools/check_param_defaults.py

Exit code 0 if every default is in support; 1 with the violations listed
otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tengri.parameters.registry import registry

SRC = ROOT / "src" / "tengri"

#: Parameters whose in-signature default is intentionally outside the declared
#: prior. Each entry needs a reason; prefer fixing the default over adding one.
ALLOWLIST: dict[str, str] = {}


def _support(name: str) -> tuple[float, float] | None:
    """Return the declared ``(low, high)`` support, or None if not bounded."""
    record = registry().get(name)
    prior = getattr(record, "prior", None) if record else None
    if prior is None:
        return None
    low = getattr(prior, "low", getattr(prior, "lo", None))
    high = getattr(prior, "high", getattr(prior, "hi", None))
    if low is None or high is None:
        return None
    return float(low), float(high)


def _numeric_defaults(tree: ast.AST):
    """Yield ``(param_name, value, lineno, func_name)`` for numeric defaults."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        positional = args.posonlyargs + args.args
        paired = list(
            zip(positional[len(positional) - len(args.defaults) :], args.defaults, strict=False)
        )
        paired += [
            (kw, default)
            for kw, default in zip(args.kwonlyargs, args.kw_defaults, strict=False)
            if default is not None
        ]
        for arg, default in paired:
            if not isinstance(default, ast.Constant):
                continue
            value = default.value
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            yield arg.arg, float(value), default.lineno, node.name


def main() -> int:
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            print(f"ERROR: cannot parse {path.relative_to(ROOT)}: {exc}")
            return 1
        for name, value, lineno, func in _numeric_defaults(tree):
            if name in ALLOWLIST:
                continue
            support = _support(name)
            if support is None:
                continue
            low, high = support
            if low <= value <= high:
                continue
            violations.append((path.relative_to(ROOT), lineno, func, name, value, low, high))

    if not violations:
        print("check_param_defaults: OK — every signature default is inside its declared prior.")
        return 0

    print(f"check_param_defaults: {len(violations)} default(s) outside the declared prior\n")
    for rel, lineno, func, name, value, low, high in violations:
        print(f"  {rel}:{lineno}  {func}()")
        print(f"      {name} = {value:g}  outside declared support [{low:g}, {high:g}]")
    print(
        "\nA default outside the prior cannot be reached by any fit, and usually means "
        "the value is in the wrong units.\nRead the default off the declaration instead "
        "of repeating it:\n"
        "    from tengri.protocols.component import declared_default\n"
        "    DEFAULT_X = declared_default(PARAMS, 'x')\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
