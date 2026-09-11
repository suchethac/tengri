#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: dust-law shape parameters reach a law by splat, never by hand, and
no law declares a ``**kwargs`` catch-all.

Two rules with one subject. The first three sections below are about *callers*
handing a law the wrong parameters; the last is about the *law* accepting
whatever it is handed, which is what let the caller defects stay silent.

Three separate defects in 2026-08 were one defect written three times, and all
three are the same *shape* of call rather than the same physics:

- the photometry LUT in ``two_component.py`` bound ``n_slope=`` at eight sites
  and dropped delta, the bump and Rv, so a free bump moved the full-grid screen
  and not the LUT the likelihood reads (#1833, fixed in ``60b16eb8a``);
- the spectroscopy pixel block built ``n_slope`` from
  ``params.get("dust_slope", -0.7)``, so a spectroscopic fit reddened its pixels
  with a different curve from the model it was fitting (#1856, fixed in
  ``db75b3fa6``);
- ``emission_helpers.attenuate_emission`` had no ``dust_delta`` or ``dust_Rv``
  parameter at all, so lines and continuum sat on different curves whenever
  either was free (#1858, fixed in #2223 by deleting the function and routing
  its one caller through the dust component's own ``attenuate_line_catalog``).

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

A third rule, added with #2185:

3. no ``@register_dust_law`` function may declare ``**kwargs``.

A fourth rule, added with #2339 to catch cross-validation mismatches:

2b. no subscript-assigned dict may carry one as a key (a hand-built dict that
   is then splatted: ``d = {}; d["dust_slope"] = ...`` is the same defect as
   rule 2);
4. every law call site across src/, tests/, bench/, examples/, analysis/ must
   use correct parameter spellings.

Every law used to. That catch-all is what made a wrong splat survivable and
therefore invisible: ``def calzetti(wavelength, **_kwargs)`` *accepts*
``dust_Rv`` while fixing R_V = 4.05 in the polynomial, so the grammar declared
the parameter free, the sampler explored it, and the curve never moved. Four
laws (``noll09``, ``salim_sbl18``, ``tea``, ``narayanan_z``) took ``slope``
beside the ``delta`` they actually read — an inert parameter three characters
from the live one. Measured across the 22 registered laws, 72 (law, per-screen
key) pairs were bit-identical to omitting the key.

With the catch-all gone the signature is the contract: callers narrow to what
the law declares (``select_law_kwargs``) and refuse a key no law in play reads
(``reject_unread_law_kwargs``), and a parameter offered to a law that cannot use
it is a ``TypeError`` at the boundary rather than a flat posterior.

Rule 2 was not hypothetical. ``attenuate_emission`` splatted honestly and was
still wrong, because the dict it splatted was built from a signature that had
no ``dust_delta`` or ``dust_Rv`` to offer (#1858). Rule 1 catches that only at
its callers; rule 2 catches it where it lives.

Rule 4 was added when the 2026-09 nightly crossval red showed a test calling
``cardelli(..., Rv=)`` and ``power_law(..., n=)`` — parameter spellings no law
declared. Rules 1-3 walk src/ only; a test calling a law with the wrong spelling
stayed silent. Rule 4 walks all trees that could call laws. A call asserted to
raise (inside ``pytest.raises(...)``) tests the law's refusal and is excluded.

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
}

# Hand-built parameter dicts that are then splatted. Splatting a dict you filled
# in by hand is the same defect wearing the correct shape, so the kwargs check
# above cannot see it; this is where #1858 actually lived.
DICT_ALLOWLIST: dict[str, str] = {}

# Individual law call sites (rule 4: spelling) that deliberately use deprecated
# or alternative spellings. Each needs a written reason and an expiration date.
#
# Format: "<relpath>::<callee>::<keyword>" -> reason
SPELLING_ALLOWLIST: dict[str, str] = {
    "tests/contract/test_dust_slope_kwarg_alias.py::power_law::n_slope": (
        "the alias test calls the deprecated spelling on purpose; expires with the alias in v1.0"
    ),
    "tests/contract/test_dust_slope_kwarg_alias.py::conroy2010::n_slope": (
        "the alias test calls the deprecated spelling on purpose; expires with the alias in v1.0"
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


def scan_law_catchalls(path: Path) -> list[tuple[int, str]]:
    """Registered dust laws in one file that declare a ``**kwargs`` catch-all."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        registered = any(
            callee_name(dec) == "register_dust_law"
            for dec in node.decorator_list
            if isinstance(dec, ast.Call)
        )
        if registered and node.args.kwarg is not None:
            found.append((node.lineno, node.name))
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


def scan_subscript_assignments(path: Path) -> list[tuple[int, str]]:
    """Subscript-assigned dict keys in one file that are shape parameters.

    Rule 2b: detect d["dust_slope"] = ... patterns (hand-built dicts splatted).
    Returns list of (lineno, key) tuples for each violation.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Check if target is a subscript: d["key"] = value
        if not isinstance(node.targets[0], ast.Subscript):
            continue
        subscript = node.targets[0]
        # Only look at string constant subscripts
        if not isinstance(subscript.slice, ast.Constant):
            continue
        if not isinstance(subscript.slice.value, str):
            continue
        key = subscript.slice.value
        if key in SHAPE_KWARGS:
            found.append((node.lineno, key))
    return found


def build_law_params(attenuation_path: Path) -> dict[str, set[str]]:
    """Build table of law names -> parameter names from attenuation.py.

    Walks @register_dust_law functions and extracts their explicit parameters.
    Also includes public wrappers that accept **law_params.
    """
    law_params: dict[str, set[str]] = {}

    try:
        tree = ast.parse(attenuation_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Check if this is a @register_dust_law function
        registered = any(
            callee_name(dec) == "register_dust_law"
            for dec in node.decorator_list
            if isinstance(dec, ast.Call)
        )
        if registered:
            # Extract parameter names from the function signature
            params = set()
            for arg in node.args.args:
                params.add(arg.arg)
            # Exclude 'self' if present
            params.discard("self")
            law_params[node.name] = params

    # Also check for public wrapper functions with **law_params catch-all
    # These accept any law parameter via the catch-all
    wrapper_names = {
        "two_component_dust",
        "two_component_dust_separable",
        "single_component_dust",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in wrapper_names:
            continue
        # Extract explicit parameters (excluding **law_params splat)
        params = set()
        for arg in node.args.args:
            params.add(arg.arg)
        # Add SHAPE_KWARGS as these are accepted via **law_params
        params.update(SHAPE_KWARGS)
        params.discard("self")
        law_params[node.name] = params

    return law_params


def _is_inside_raises(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Check if a node is inside a pytest.raises(...) or .raises(...) context."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.With):
            # Check if this With statement has a raises() call
            for item in current.items:
                if isinstance(item.context_expr, ast.Call):
                    ctx_callee = callee_name(item.context_expr)
                    if ctx_callee == "raises" or ctx_callee.endswith(".raises"):
                        return True
    return False


def scan_call_spellings(path: Path, law_params: dict[str, set[str]]) -> list[tuple[int, str, str]]:
    """Rule 4: check law call sites for correct parameter spellings.

    Excludes calls inside pytest.raises(...) blocks (negative tests).
    Returns list of (lineno, callee_name, bad_keyword) for each violation.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    # Build parent map for all nodes
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = callee_name(node)
        if name not in law_params:
            continue

        # Skip calls inside pytest.raises(...) blocks
        if _is_inside_raises(node, parents):
            continue

        # Check each explicit keyword argument
        accepted = law_params[name]
        for kw in node.keywords:
            # Skip **kwargs splats
            if kw.arg is None:
                continue
            if kw.arg not in accepted:
                found.append((node.lineno, name, kw.arg))

    return found


def main() -> int:
    if not SRC.is_dir():
        print(f"ERROR: cannot read {SRC}", file=sys.stderr)
        return 1

    violations: list[str] = []
    seen_allowlist: set[str] = set()

    # Rule 1: hand-bound law call kwargs
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

    # Rule 3: law definitions must not have **kwargs catch-all
    # Scans every file, EXEMPT_FILES included: `attenuation.py` is exempt
    # from the hand-binding rule precisely because it defines the laws, which is
    # the one place this rule has to look.
    for path in sorted(SRC.rglob("*.py")):
        rel = relpath(path)
        for lineno, fn_name in scan_law_catchalls(path):
            violations.append(
                f"{rel}:{lineno}  @register_dust_law ... def {fn_name}(..., **kwargs)\n"
                "      A registered law must declare exactly the parameters it reads.\n"
                "      A catch-all makes the law ACCEPT a parameter it never uses and\n"
                "      discard the value in silence: the grammar then frees it, the\n"
                "      sampler explores it, and the curve never moves (#2185). Delete\n"
                "      the catch-all; callers narrow with `select_law_kwargs`."
            )

    # Rule 2: dict literals with shape parameters
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

    # Rule 2b: subscript-assigned dicts with shape parameters
    seen_subscript_dicts: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = relpath(path)
        if rel in EXEMPT_FILES or rel in DECLARATION_FILES:
            continue
        for lineno, key in scan_subscript_assignments(path):
            if rel in DICT_ALLOWLIST:
                seen_subscript_dicts.add(rel)
                continue
            violations.append(
                f'{rel}:{lineno}  d["{key}"] = ...\n'
                "      A hand-built law-parameter dict via subscript assignment.\n"
                "      Splatting a dict you filled in by hand wears the right shape\n"
                "      and carries the wrong contents. Get it from\n"
                "      `resolve_bc_diff_law_params`, or add this file to\n"
                "      DECLARATION_FILES if it enumerates parameters rather than\n"
                "      evaluating a law with them."
            )

    # Rule 4: law call sites must use correct parameter spellings
    # Build law_params table from attenuation.py
    attenuation_path = SRC / "components" / "dust" / "attenuation.py"
    law_params = build_law_params(attenuation_path)

    # Walk src/, tests/, bench/, examples/, analysis/ for call sites
    repo_root = SRC.parent.parent
    search_roots = [
        repo_root / "src",
        repo_root / "tests",
        repo_root / "bench",
        repo_root / "examples",
        repo_root / "analysis",
    ]
    seen_spellings: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        # Make rel paths relative to repo_root for allowlisting
        for path in sorted(root.rglob("*.py")):
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                continue

            for lineno, callee, bad_kw in scan_call_spellings(path, law_params):
                key = f"{rel}::{callee}::{bad_kw}"
                if key in SPELLING_ALLOWLIST:
                    seen_spellings.add(key)
                    continue
                violations.append(
                    f"{rel}:{lineno}  {callee}(..., {bad_kw}=...)\n"
                    f"      Unknown parameter '{bad_kw}' for law '{callee}'.\n"
                    "      Check the law's signature for the correct spelling,\n"
                    "      or add an allowlist entry if this is an intentional\n"
                    "      deprecated-spelling test."
                )

    # Stale allowlist checks
    for key in sorted(set(ALLOWLIST) - seen_allowlist):
        violations.append(
            f"stale allowlist entry `{key}` in {Path(__file__).name}: the site is\n"
            "      gone. Drop the entry so the next real one is not hidden behind it."
        )

    for key in sorted(set(DICT_ALLOWLIST) - (seen_dicts | seen_subscript_dicts)):
        violations.append(
            f"stale DICT_ALLOWLIST entry `{key}` in {Path(__file__).name}: the\n"
            "      hand-built dict is gone. Drop the entry."
        )

    for key in sorted(set(SPELLING_ALLOWLIST) - seen_spellings):
        violations.append(
            f"stale SPELLING_ALLOWLIST entry `{key}` in {Path(__file__).name}: the\n"
            "      call site is gone or the spelling has been fixed. Drop the entry."
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
        print("Dust-law shape parameters bound by hand or misspelled:\n", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}\n", file=sys.stderr)
        return 1

    n_exempt = len(EXEMPT_FILES) + len(DECLARATION_FILES)
    n_dicts = len(DICT_ALLOWLIST)
    n_spellings = len(SPELLING_ALLOWLIST)
    print(
        f"OK: every dust-law evaluation takes its parameters from a resolved dict, "
        f"and no registered law declares a **kwargs catch-all, and all law call "
        f"sites use correct parameter spellings "
        f"({len(ALLOWLIST)} call sites and {n_dicts} hand-built "
        f"dict{'' if n_dicts == 1 else 's'} and {n_spellings} deprecated-spelling "
        f"allowlisted, {n_exempt} files exempt)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
