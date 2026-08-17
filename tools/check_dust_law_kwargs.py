#!/usr/bin/env python3
"""CI guard: dust-law shape parameters reach a law by splat, never by hand.

Three separate defects in 2026-08 were one defect written three times, and all
three are the same *shape* of call rather than the same physics:

- the photometry LUT in ``two_component.py`` bound ``n_slope=`` at eight sites
  and dropped delta, the bump and Rv, so a free bump moved the full-grid screen
  and not the LUT the likelihood reads (#1833, fixed in ``60b16eb8a``);
- the spectroscopy pixel block built ``n_slope`` from
  ``params.get("dust_slope", -0.7)``, so a spectroscopic fit reddened its pixels
  with a different curve from the model it was fitting (#1856, fixed in
  ``db75b3fa6``);
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

Two forms of the same rule, because a kwargs-only check has an obvious way out —
fill a dict in by hand and splat *that*, and every call site reads as correct:

1. no call may name a shape parameter as an explicit keyword;
2. no dict literal may carry one as a key, outside the files that legitimately
   *enumerate* parameters (registries, priors, name-translation tables).

Rule 2 is not hypothetical. ``attenuate_emission`` splats honestly and is still
wrong, because the dict it splats is built from a signature that has no
``dust_delta`` or ``dust_Rv`` to offer (#1858). Rule 1 catches that only at its
callers; rule 2 catches it where it lives.

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

# Files that legitimately write shape parameters as dict *keys*: registries,
# prior declarations and name-translation tables all enumerate the parameters
# rather than evaluating a law with them. Applies to the dict-literal check
# only; these files are still checked for hand-bound call kwargs.
DECLARATION_FILES: dict[str, str] = {
    "components/dust/laws/_registry.py": "declares which parameters each law accepts",
    "components/dust/priors.py": "declares the priors, keyed by parameter name",
    "parameters/translate.py": "maps external parameter names onto tengri's",
    "presets/synthesizer.py": "a preset's parameter values, not a law evaluation",
}

# Individual sites that are deliberately hand-bound. Each needs a written
# reason — a bare 'we know' entry here is how the fourth instance ships.
#
# Entries are meant to leave. The two `two_component.py` ones were dropped when
# #1856 merged as `db75b3fa6`, and the stale-entry check below is what said so:
# it failed on the rebase before any human reread the file. That is the point of
# expiring an exception rather than recording it.
ALLOWLIST: dict[str, str] = {
    "forward/sed_model.py::cardelli": (
        "Milky Way FOREGROUND extinction, not the galaxy's attenuation law: "
        "`dust_Rv` here is `spec.foreground_rv`, a separate field with its own "
        "physical meaning. There is no resolved galaxy-law dict to splat, and "
        "splatting one would be wrong"
    ),
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

# Hand-built parameter dicts that are then splatted. Splatting a dict you filled
# in by hand is the same defect wearing the correct shape, so the kwargs check
# above cannot see it; this is where #1858 actually lives.
DICT_ALLOWLIST: dict[str, str] = {
    "forward/emission_helpers.py": (
        "#1858, OPEN — `attenuate_emission` builds `dust_kw` from its own two "
        "parameters and splats that. The splat is honest; the SIGNATURE is "
        "partial, so `dust_delta`/`dust_Rv` cannot be threaded at all. This is "
        "the defect's real location; its callers in `sed_model.py` are the "
        "symptom. Remove this entry with the signature fix"
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


def scan_dict_literals(path: Path) -> list[tuple[int, set[str]]]:
    """Dict literals in one file that carry shape parameters as keys."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        named = keys & SHAPE_KWARGS
        if named:
            found.append((node.lineno, named))
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

    seen_dicts: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = relpath(path)
        if rel in EXEMPT_FILES or rel in DECLARATION_FILES:
            continue
        for lineno, named in scan_dict_literals(path):
            if rel in DICT_ALLOWLIST:
                seen_dicts.add(rel)
                continue
            listed = ", ".join(sorted(named))
            violations.append(
                f"{rel}:{lineno}  {{{listed}: ...}}\n"
                "      A hand-built law-parameter dict. Splatting a dict you filled in\n"
                "      by hand wears the right shape and carries the wrong contents —\n"
                "      whatever its author remembered. Get it from\n"
                "      `resolve_bc_diff_law_params`, or add this file to\n"
                "      DECLARATION_FILES if it enumerates parameters rather than\n"
                "      evaluating a law with them."
            )

    for key in sorted(set(ALLOWLIST) - seen_allowlist):
        violations.append(
            f"stale allowlist entry `{key}` in {Path(__file__).name}: the site is\n"
            "      gone. Drop the entry so the next real one is not hidden behind it."
        )

    for key in sorted(set(DICT_ALLOWLIST) - seen_dicts):
        violations.append(
            f"stale DICT_ALLOWLIST entry `{key}` in {Path(__file__).name}: the\n"
            "      hand-built dict is gone. Drop the entry."
        )

    for key in sorted(DECLARATION_FILES):
        path = SRC / key
        if not path.exists() or not scan_dict_literals(path):
            violations.append(
                f"stale DECLARATION_FILES entry `{key}` in {Path(__file__).name}:\n"
                "      the file no longer declares shape parameters as dict keys, so the\n"
                "      exemption now covers nothing and would silently excuse the next\n"
                "      real one. Drop the entry."
            )

    if violations:
        print("Dust-law shape parameters bound by hand:\n", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}\n", file=sys.stderr)
        return 1

    n_exempt = len(EXEMPT_FILES) + len(DECLARATION_FILES)
    n_dicts = len(DICT_ALLOWLIST)
    print(
        f"OK: every dust-law evaluation takes its parameters from a resolved dict "
        f"({len(ALLOWLIST)} call sites and {n_dicts} hand-built "
        f"dict{'' if n_dicts == 1 else 's'} allowlisted, {n_exempt} files exempt)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
