#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a bare numeral must never stand in for a declared parameter default.

A model function's signature default, or the fallback literal in a
``p.get("name", <literal>)`` call, sometimes repeats a value a
:class:`~tengri.protocols.component.ParamDeclaration` (or a component's own
class-level prior attribute) already owns. That bare numeral is a second copy,
and the two drift silently: #2241 found fifteen such sites in the dust
emission tree, two of which had ALREADY drifted from the shared
``components/dust/_params.py`` table before anyone noticed (``dust_T`` and
``dust_beta_ir`` -- see ``_params.py``'s own comments on those two entries,
and issue #2261, filed to resolve which value is correct).

This is deliberately **not** an equality check against a registry value.
Checking "does this literal equal the current declared default" would (a)
still miss the ``dust_T`` / ``dust_beta_ir`` shape, because that pair had
*already* drifted -- a drifted pair is exactly what equality cannot see -- and
(b) requires a global, single-registry notion of "the" default that this
codebase does not have: some names are correctly read off the shared
``_params.py`` table, others off a component's own class-level attribute, and
the two can legitimately disagree (that disagreement is #2261's whole
subject). So instead of comparing values, this guard asks a structural
question: is a *name a declaration already owns* spelled here as a literal
number at all? If so, it is a second copy by construction, independent of
whether the two numbers currently happen to match.

Scope
-----
``src/tengri/components/dust/emission/`` by default (pass ``--scope`` to
widen). The disease is not confined to dust emission:
``components/dust/emission_templates.py`` -- one directory above the default
scope, despite its name -- holds 20 literal copies of its own
(``dl07_tabulated``, ``dale2014_emission_lnu``, ``schreiber2018_tabulated``,
``themis_emission`` and siblings each repeat a ``dust_umin`` / ``dust_qpah`` /
``dust_alpha_dale`` / ... default as a bare numeral); and
``attenuation.py``'s ``kriek_conroy`` law repeats
``dust_bump_strength: float = 1.0`` against a declared ``Fixed(0.0)`` in
``ATTENUATION_PARAMS`` (plausibly deliberate -- KC13's own published value --
but still an unguarded second copy). Widening this guard to those two files
and beyond is tracked as a follow-up to #2241; see the PR body for the count
and sites a ``--scope src/tengri/components`` run turns up today.

How the declared-name set is built
-----------------------------------
1. Every ``ParamDeclaration("<name>", ...)`` string literal found by walking
   every ``_params.py`` file inside ``--scope``, plus the dust component's
   own ``components/dust/_params.py`` (which sits one directory above the
   default scope and would otherwise never be seen).
2. Every class-level attribute assignment inside ``--scope`` whose value is a
   call to a known :class:`~tengri.parameters.priors.Distribution`
   constructor (``Fixed``, ``Uniform``, ``LogNormal``, ``LogUniform``,
   ``Gaussian``, ``TruncatedNormal``), prefixed with that class's
   ``parameter_prefix`` (read off a class-level ``parameter_prefix =`` /
   ``parameter_prefix: str =`` assignment in the same class body; when a
   class does not repeat it -- every concrete emission component inherits
   ``"dust_"`` from the shared ``EmissionComponent`` rather than restating it
   -- the fallback is ``"dust_"`` ONLY for a class defined under
   ``components/dust/`` (see :func:`_file_fallback_prefix`); a class defined
   elsewhere without its own ``parameter_prefix`` is skipped rather than
   guessed at, so a bare AGN/nebular/... attribute is never silently folded
   into a ``dust_`` name it has nothing to do with).

A class-level assignment like ``T = Fixed(30.0)`` is a name's DECLARING site
(it grows the declared-name set) and is never itself flagged: the disease is
a *literal copy elsewhere*, not the declaration.

What is flagged
----------------
Once the declared-name set is built, every occurrence in ``--scope`` of:

* a ``def``/``async def`` positional-or-keyword-only parameter whose name is
  in the declared-name set, with a numeric-literal default, or
* a ``<mapping>.get("<name>", <literal>)`` call where ``<name>`` is in the
  declared-name set and the second argument is a numeric literal,

is a violation. No allowlist: the fix is to read the value through
``declared_default(...)`` (``tengri.protocols.component``) or a shared named
module constant instead of repeating the number -- the fix this guard exists
to keep in place leaves zero sites. Every report line names the source
spelling beside the resolved declared name whenever a bare-to-prefixed
fallback was needed to match them (e.g. ``f_cold`` resolving to
``dust_f_cold``), and a flagged ``.get(...)`` always prints the key exactly
as written in the source, so the printed line always greps.

Usage
-----
    python tools/check_literal_param_defaults.py
    python tools/check_literal_param_defaults.py --scope src/tengri/components
    python tools/check_literal_param_defaults.py --list

Exit code 0 if there are no literal-copy sites; 1 with the violations listed
otherwise. ``--list`` prints every site the declared-name set can see, with
its OK/FAIL verdict, and always exits 0 (it is a census, not a gate).

stdlib-only: this guard imports nothing beyond ``ast``, ``argparse``,
``sys`` and ``pathlib``, so it needs no venv beyond the interpreter running
it, unlike ``check_param_defaults.py``'s sibling which imports the package to
consult the live prior registry.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCOPE = ROOT / "src" / "tengri" / "components" / "dust" / "emission"
_DUST_TREE = ROOT / "src" / "tengri" / "components" / "dust"

#: The dust component's own shared table. Sits one directory above the
#: default scope, so a narrow ``--scope`` run would otherwise never see the
#: ``ParamDeclaration`` names it owns (``dust_T_warm``, ``dust_f_cold``, ...,
#: read by ``energy_balance_split``, which declares nothing on its own class
#: -- see that component's docstring).
_DUST_PARAMS_FILE = _DUST_TREE / "_params.py"

#: The ONLY bare-name fallback prefix this guard knows, and only for a file
#: physically under ``components/dust/`` -- see :func:`_file_fallback_prefix`.
_DEFAULT_PREFIX = "dust_"


def _file_fallback_prefix(path: Path) -> str | None:
    """The bare-name fallback prefix for ``path``'s own subtree, or ``None``.

    Only ``components/dust/`` gets a fallback prefix: it is the one subtree
    whose shared base class (``EmissionComponent``) sets
    ``parameter_prefix = "dust_"`` without every subclass repeating it, and
    whose ``energy_balance_split`` closure spells declared names bare
    (``f_cold``, ``eta_balance``, ``L_agn_ir``, read by
    :func:`_resolve_declared_name`). Guessing a fallback prefix for any OTHER
    subtree misattributes an unrelated bare name to a ``dust_`` declaration
    by pure string collision -- #2241 review M1: AGN's
    ``skirtor_disk_spectrum(delta=...)`` (the disc power-law modulation) and
    ``frac_agn`` are not ``dust_delta``/``dust_frac_agn``, and a widened
    ``--scope`` run reported them as if they were. Returning ``None`` outside
    ``components/dust/`` disables the fallback there entirely: such a class
    or function is matched only on its EXACT declared spelling.
    """
    try:
        resolved = path.resolve()
        resolved.relative_to(_DUST_TREE)
    except ValueError:
        return None
    return _DEFAULT_PREFIX


#: Constructors from ``tengri.parameters.priors`` that mark a class-level
#: attribute as a declared free parameter (mirrors the set
#: ``SEDModelComponent.__init_subclass__`` auto-discovers via
#: ``isinstance(attr_value, Distribution)`` -- checked structurally here
#: since this guard is stdlib-only and cannot import the package).
_DISTRIBUTION_CONSTRUCTORS = frozenset(
    {"Fixed", "Uniform", "LogNormal", "LogUniform", "Gaussian", "TruncatedNormal"}
)


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text())
    except SyntaxError as exc:
        print(f"ERROR: cannot parse {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _declared_names_from_params_file(path: Path) -> set[str]:
    """Every ``ParamDeclaration("<name>", ...)`` string literal in ``path``."""
    if not path.exists():
        return set()
    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ParamDeclaration"
        ):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
    return names


def _class_parameter_prefix(class_node: ast.ClassDef, fallback: str | None) -> str | None:
    """A class-level ``parameter_prefix = "..."`` in this class's own body.

    Falls back to ``fallback`` when the class does not set its own (e.g.
    every concrete emission component inherits it from ``EmissionComponent``
    rather than repeating it) -- the caller passes
    :func:`_file_fallback_prefix` for the class's own file, which is
    ``"dust_"`` only under ``components/dust/`` and ``None`` elsewhere, so a
    class outside that subtree with no ``parameter_prefix`` of its own
    returns ``None`` rather than a guessed prefix (#2241 review M1).
    """
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "parameter_prefix" for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return fallback


def _declared_names_from_component_classes(
    tree: ast.Module, fallback_prefix: str | None
) -> set[str]:
    """Every class-level ``name = <Distribution>(...)`` attribute, prefixed.

    A class whose own ``parameter_prefix`` cannot be determined (no
    class-level assignment, and ``fallback_prefix`` is ``None`` because the
    file is outside ``components/dust/``) contributes nothing: skipped
    rather than guessed at.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        prefix = _class_parameter_prefix(node, fallback_prefix)
        if prefix is None:
            continue
        for stmt in node.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or target.id.startswith("_"):
                continue
            value = stmt.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in _DISTRIBUTION_CONSTRUCTORS
            ):
                names.add(f"{prefix}{target.id}")
    return names


def _iter_py_files(scope: Path) -> list[Path]:
    return sorted(scope.rglob("*.py"))


def collect_declared_names(scope: Path) -> set[str]:
    """The full declared-name set visible from ``scope`` (see module docstring)."""
    names = _declared_names_from_params_file(_DUST_PARAMS_FILE)
    for path in _iter_py_files(scope):
        if path.name.endswith("_params.py") or path.name == "params.py":
            names |= _declared_names_from_params_file(path)
        names |= _declared_names_from_component_classes(_parse(path), _file_fallback_prefix(path))
    return names


def _resolve_declared_name(local_name: str, declared: set[str], prefix: str | None) -> str | None:
    """Match a local parameter/key spelling against the declared-name set.

    Most sites spell the name with its domain prefix already
    (``dust_lambda_0_um``), but a component's own ``predict`` reads
    prefix-stripped keys (``p["f_cold"]`` for the declared ``dust_f_cold``),
    and the ``energy_balance_split`` *closure*'s own signature mixes both
    conventions in the same function (``dust_T_warm`` alongside bare
    ``f_cold``/``eta_balance``/``L_agn_ir``). Checking the bare spelling
    first and the ``prefix``-qualified spelling second catches both without
    special-casing either -- but ``prefix`` comes from
    :func:`_file_fallback_prefix` for the SITE'S OWN FILE, not a blind
    module-wide constant: ``None`` (any file outside ``components/dust/``)
    disables the fallback, so a bare name is only ever matched exactly
    (#2241 review M1).
    """
    if local_name in declared:
        return local_name
    if prefix is not None:
        prefixed = f"{prefix}{local_name}"
        if prefixed in declared:
            return prefixed
    return None


def _signature_default_sites(tree: ast.Module, declared: set[str], prefix: str | None):
    """Yield ``(source_name, matched_name, lineno, func_name, is_literal, repr)``
    for every default whose parameter name (as spelled, or resolved through
    ``prefix``) is in ``declared``. ``source_name`` is always the identifier
    exactly as written; ``matched_name`` is the declared name it resolves to
    (identical to ``source_name`` unless a prefix fallback fired)."""
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
            matched = _resolve_declared_name(arg.arg, declared, prefix)
            if matched is None:
                continue
            is_numeric_literal = (
                isinstance(default, ast.Constant)
                and isinstance(default.value, int | float)
                and not isinstance(default.value, bool)
            )
            yield (
                arg.arg,
                matched,
                default.lineno,
                node.name,
                is_numeric_literal,
                ast.unparse(default) if hasattr(ast, "unparse") else repr(default),
            )


def _get_fallback_sites(tree: ast.Module, declared: set[str], prefix: str | None):
    """Yield ``(source_name, matched_name, lineno, is_literal, repr)`` for
    every ``.get(name, ...)`` call whose first argument (as spelled, or
    resolved through ``prefix``) names a declared parameter. ``source_name``
    is the string literal exactly as written in the call, so a report line
    built from it always greps against the source."""
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ):
            continue
        if len(node.args) != 2:
            continue
        name_arg, default_arg = node.args
        if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
            continue
        source_name = name_arg.value
        matched = _resolve_declared_name(source_name, declared, prefix)
        if matched is None:
            continue
        is_numeric_literal = (
            isinstance(default_arg, ast.Constant)
            and isinstance(default_arg.value, int | float)
            and not isinstance(default_arg.value, bool)
        )
        yield (
            source_name,
            matched,
            node.lineno,
            is_numeric_literal,
            ast.unparse(default_arg) if hasattr(ast, "unparse") else repr(default_arg),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        default=str(DEFAULT_SCOPE),
        help="Directory to scan (default: src/tengri/components/dust/emission).",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print every site with its OK/FAIL verdict."
    )
    args = parser.parse_args()

    scope = Path(args.scope)
    if not scope.is_absolute():
        scope = ROOT / scope
    scope = scope.resolve()
    try:
        scope.relative_to(ROOT)
    except ValueError:
        print(
            f"check_literal_param_defaults: --scope must be a directory inside "
            f"this repository ({ROOT}); got {scope}, which is not. Point "
            "--scope at a directory under this checkout, e.g. "
            "--scope src/tengri/components.",
            file=sys.stderr,
        )
        return 2
    if not scope.is_dir():
        print(
            f"check_literal_param_defaults: --scope {scope} is not a directory.",
            file=sys.stderr,
        )
        return 2

    declared = collect_declared_names(scope)

    all_sites: list[tuple[Path, int, str, bool]] = []  # (rel, lineno, desc, is_violation)
    for path in _iter_py_files(scope):
        tree = _parse(path)
        rel = path.relative_to(ROOT)
        prefix = _file_fallback_prefix(path)
        for source_name, matched, lineno, func, is_literal, value_repr in _signature_default_sites(
            tree, declared, prefix
        ):
            name_display = (
                matched if matched == source_name else f"{matched} (spelled `{source_name}` here)"
            )
            desc = f"{func}(): {name_display} = {value_repr}  (signature default)"
            all_sites.append((rel, lineno, desc, is_literal))
        for source_name, matched, lineno, is_literal, value_repr in _get_fallback_sites(
            tree, declared, prefix
        ):
            # Always print the key exactly as written in the source (#2241
            # review M1), even when it only resolves through a prefix
            # fallback, so the printed line greps against the file.
            suffix = "" if matched == source_name else f"  (resolves to `{matched}`)"
            desc = f'.get("{source_name}", {value_repr})  (fallback){suffix}'
            all_sites.append((rel, lineno, desc, is_literal))

    all_sites.sort(key=lambda item: (str(item[0]), item[1]))
    violations = [site for site in all_sites if site[3]]

    if args.list:
        print(
            f"check_literal_param_defaults: {len(declared)} declared name(s) tracked "
            f"under {scope.relative_to(ROOT)}\n"
        )
        if not all_sites:
            print("  (no sites reference a declared name)")
        for rel, lineno, desc, is_literal in all_sites:
            verdict = "FAIL" if is_literal else "OK"
            print(f"  [{verdict}] {rel}:{lineno}  {desc}")
        return 0

    if not violations:
        print(
            "check_literal_param_defaults: OK -- no bare numeral stands in for a declared default."
        )
        return 0

    print(f"check_literal_param_defaults: {len(violations)} literal-copy site(s)\n")
    for rel, lineno, desc, _ in violations:
        print(f"  {rel}:{lineno}  {desc}")
    print(
        "\nA bare numeral for a name a ParamDeclaration or component class attribute "
        "already owns is a second copy that can silently drift from the declaration -- "
        "equality with the declaration is not the test, because a pair that has already "
        "drifted is exactly what equality cannot see (#2241). Read it instead:\n"
        "    from tengri.protocols.component import declared_default\n"
        "    DEFAULT_X = declared_default(PARAMS, 'dust_x')\n"
        "or, when the declaration disagrees with what the caller must actually use "
        "(#2261), a named module constant both sides import.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
