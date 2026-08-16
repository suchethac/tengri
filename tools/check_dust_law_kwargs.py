#!/usr/bin/env python3
"""CI guard: dust-law shape parameters reach a law by splat, never by hand.

Three separate defects in 2026-08 were one defect written three times, and all
three are the same *shape* of call rather than the same physics:

- the photometry LUT in ``two_component.py`` bound ``n_slope=`` at eight sites
  and dropped delta, the bump and Rv, so a free bump moved the full-grid screen
  and not the LUT the likelihood reads (#1833, fixed in ``60b16eb8a``);
- the spectroscopy pixel block built ``n_slope`` from
  ``params.get("dust_slope", -0.7)``, so a spectroscopic fit reddened its pixels
  with a different curve from the model it was fitting (#1856);
- ``emission_helpers.attenuate_emission`` has no ``dust_delta`` or ``dust_Rv``
  parameter at all, so lines and continuum sit on different curves whenever
  either is free (#1858).

A survey of the law-evaluation call sites sorts on that shape and not on
anything semantic: **every site that splats a resolved parameter dict is
correct, and every site that names its kwargs by hand has drifted or will.**
Hand-binding is what makes a site silently partial — it passes the parameters
whoever wrote it remembered, and a parameter added later reaches every splatted
site and no hand-rolled one.

So this guard enforces the shape. Build the dict with
``resolve_bc_diff_law_params`` (or the caller's own resolver) and splat it.

Dependencies: standard library only. The ``lint`` job installs ruff and nothing
else, so this must not import ``yaml`` or ``tengri``. AST rather than grep: a
call's keywords routinely sit on different lines from its callee, and a
line-based scan of this same tree reports a fraction of the real sites.

Usage
-----
    python tools/check_dust_law_kwargs.py

Exit code 0 when every law evaluation splats; 1 otherwise, listing each
hand-bound site.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "tengri"

# Shape parameters of an attenuation law. Naming any of these as an explicit
# keyword is the hand-binding this guard exists to catch.
SHAPE_KWARGS = frozenset(
    {
        "n_slope",
        "dust_slope",
        "dust_bump_strength",
        "dust_delta",
        "dust_Rv",
        "dust_tea_scatter",
    }
)

# Calls that merely *name* these as data — building a parameter mapping — rather
# than evaluating a law with them. `Parameters(dust_slope=...)` declares a
# parameter; it does not bind a curve.
DATA_CONSTRUCTORS = frozenset({"Parameters", "ParamSpec", "dict", "replace"})

# Files where naming a shape parameter explicitly IS the correct thing.
EXEMPT_FILES: dict[str, str] = {
    "components/dust/attenuation.py": (
        "the law definitions themselves; several compose each other "
        "(kriek_conroy on calzetti, narayanan_z on its own base) and must name "
        "the parameters they are defining"
    ),
}

# Individual sites that are deliberately hand-bound. Each needs a written
# reason — a bare 'we know' entry here is how the fourth instance ships.
ALLOWLIST: dict[str, str] = {
    "forward/sed_model.py::cardelli": (
        "Milky Way FOREGROUND extinction, not the galaxy's attenuation law: "
        "`dust_Rv` here is `spec.foreground_rv`, a separate field with its own "
        "physical meaning. There is no resolved galaxy-law dict to splat, and "
        "splatting one would be wrong"
    ),
    "components/dust/two_component.py::law_bc_fn": (
        "#1856, OPEN — the spectroscopy pixel block. Remove this entry when "
        "that PR merges; it is the second of the three 2026-08 instances"
    ),
    "components/dust/two_component.py::law_diff_fn": ("#1856, OPEN — same block as above"),
    "forward/sed_model.py::attenuate_emission": (
        "#1858, OPEN — `attenuate_emission` has no `dust_delta`/`dust_Rv` "
        "parameter, so the caller cannot splat even if it wanted to. Fixing the "
        "signature is the issue; remove this entry with it"
    ),
    "analysis/simulate.py::two_component_dust": (
        "mock-generation path: binds `n_slope=` beside a `**dust_kwargs` splat. "
        "Same shape as the real defects and worth folding into the #1858 sweep, "
        "but it drives `analysis/simulate.py` fixtures rather than a fit"
    ),
}


def relpath(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def callee_name(node: ast.Call) -> str:
    """Best-effort name for whatever is being called."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Call):  # resolve_dust_law("calzetti")(wave, ...)
        return callee_name(func)
    return "<expr>"


def hand_bound_kwargs(node: ast.Call) -> set[str]:
    """Shape parameters this call names explicitly."""
    return {kw.arg for kw in node.keywords if kw.arg in SHAPE_KWARGS}


def scan(path: Path) -> list[tuple[int, str, set[str]]]:
    """Hand-bound law evaluations in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = callee_name(node)
        if name in DATA_CONSTRUCTORS:
            continue
        bound = hand_bound_kwargs(node)
        if bound:
            found.append((node.lineno, name, bound))
    return found


def main() -> int:
    if not SRC.is_dir():
        print(f"ERROR: cannot read {SRC}", file=sys.stderr)
        return 1

    violations: list[str] = []
    seen_allowlist: set[str] = set()

    for path in sorted(SRC.rglob("*.py")):
        rel = relpath(path)
        if rel in EXEMPT_FILES:
            continue
        for lineno, name, bound in scan(path):
            key = f"{rel}::{name}"
            if key in ALLOWLIST:
                seen_allowlist.add(key)
                continue
            named = ", ".join(sorted(bound))
            violations.append(
                f"{rel}:{lineno}  {name}(..., {named}=...)\n"
                "      Hand-bound shape parameters. Build the dict with\n"
                "      `resolve_bc_diff_law_params` and splat it, so a parameter\n"
                "      added later reaches this site too."
            )

    for key in sorted(set(ALLOWLIST) - seen_allowlist):
        violations.append(
            f"stale allowlist entry `{key}` in {Path(__file__).name}: the site is\n"
            "      gone. Drop the entry so the next real one is not hidden behind it."
        )

    if violations:
        print("Dust-law shape parameters bound by hand:\n", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}\n", file=sys.stderr)
        return 1

    print(
        f"OK: every dust-law evaluation splats its parameters "
        f"({len(ALLOWLIST)} documented exceptions)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
