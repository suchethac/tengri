# SPDX-License-Identifier: BSD-3-Clause
"""Every in-tree constructor call must match the class's real ``__init__`` signature.

Guards a rot class that has reached main three times and was invisible to CI every
time, because the offending line lived on a path no test ever executed:

* ``Fitter._fit_batch_vmap_map`` built ``Posterior(log_weights=..., fitter=...)`` —
  neither is a parameter — while omitting the required ``params`` / ``wall_time_s``.
* ``cb19_precompute.precompute`` and ``feltre_precompute.precompute`` built
  ``PreintegratedGrid(...)`` with 3 of 8 required arguments, after #1122/#1133 added
  the extra fields without updating these call sites.
* ``test_precompute_collapse_seam.py`` built ``PreintegratedGrid(flux_scale=1.0)``
  after #1878 renamed that field to ``log10_flux_scale``. This guard scanned ``src/``
  only, so it could not see a call site under ``tests/``. #1765 and #1878 were each
  green against their own merge base, neither run saw the other's change, and the pair
  turned ``main`` red (#1899). Closing that blind spot is #1919.

A "does it import?" or "is it callable?" check cannot catch any of them — only
executing the line, or checking the call against the signature the way this test does.
This needs no ``data/`` files, so unlike an end-to-end precompute test it actually runs
on CI.

Two roots, resolved two different ways
--------------------------------------

``src/`` modules are imported, so a name resolves from the module namespace
(``dir(mod)``) *and* from every ``from tengri… import`` in the AST.

``tests/`` modules are deliberately **not** imported — importing a test module runs its
collection-time side effects — so names there resolve from the AST alone. Test files
import what they construct explicitly, so this costs very little in practice: extending
the guard to ``tests/`` surfaced zero violations across 623 files, i.e. a pure coverage
gain with no backlog to clear first.

Because "no violations" and "sees nothing" produce identical output,
``test_the_check_detects_a_stale_call`` feeds the AST-only path the exact call that
turned main red and asserts it is caught. Without that, this file could rot into a
guard that passes because it inspects nothing.

Precision (the reasons this does not fire on legitimate code):

* the class name is resolved in the **calling module's own namespace**, so two classes
  sharing a name never get confused;
* validation uses :func:`inspect.signature`, not ``dataclasses.fields`` — classes with a
  hand-written ``__init__`` whose parameters differ from their fields (e.g.
  ``PhotometryLikelihood(fnu_obs=...)``) are checked against what they actually accept;
* calls using positional args, ``*args`` or ``**kwargs`` unpacking are skipped as
  unverifiable, as are classes whose ``__init__`` accepts ``**kwargs``.

What this does not see
----------------------

Only bare-name calls — ``PreintegratedGrid(...)``. A call written through an attribute,
``grid_interp.PreintegratedGrid(...)``, is not checked, in either root: the name would
have to be resolved through the module object rather than the import, and the AST node
is an ``Attribute`` rather than a ``Name``. Both known incidents were written bare, so
this is a real gap rather than a theoretical one only in the sense that nothing has
fallen into it yet.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import warnings

import pytest

pytestmark = pytest.mark.contract

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_TESTS = _REPO / "tests"


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(_SRC)
    name = ".".join(rel.with_suffix("").parts)
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _iter_calls(tree: ast.AST):
    """Yield (name, lineno) for every ``Name(...)`` call with keyword-only args."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.args:  # positional -> cannot map to parameter names
            continue
        if any(kw.arg is None for kw in node.keywords):  # **kwargs unpacking
            continue
        if not node.keywords:
            continue
        yield node.func.id, node.lineno, {kw.arg for kw in node.keywords}


def _resolve_names(tree: ast.AST, mod=None) -> dict[str, type]:
    """Map every class name usable in this module to the real class object.

    Module-level ``getattr`` alone is not enough: tengri imports many symbols
    *inside functions* to break import cycles (e.g. ``Posterior`` in ``fitter.py``),
    and those never appear in the module namespace. Missing them would make this
    guard silently skip the very call sites most likely to have rotted — a guard
    that fails open. So walk every ``from X import Y`` at any nesting depth too.

    ``mod`` is None for test files, which are never imported. The AST half alone
    carries them: a test constructs what it explicitly imported.

    Warnings are suppressed because this only *inspects* a namespace. Reading a
    deprecated attribute is enough to fire its ``DeprecationWarning`` (``ModelConfig``
    does), and a guard that reports on other code should not editorialize in the
    warnings summary about names it merely looked at.
    """
    names: dict[str, type] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mod is not None:
            for attr in dir(mod):
                obj = getattr(mod, attr, None)
                if isinstance(obj, type):
                    names[attr] = obj
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None or node.level:
                continue
            if not node.module.startswith("tengri"):
                continue
            try:
                src = importlib.import_module(node.module)
            except Exception:
                continue
            for alias in node.names:
                obj = getattr(src, alias.name, None)
                if isinstance(obj, type):
                    names[alias.asname or alias.name] = obj
    return names


def _problems_in(tree: ast.AST, resolved: dict[str, type], where: str) -> list[str]:
    """Check every resolvable constructor call in one parsed module."""
    problems: list[str] = []
    for name, lineno, given in _iter_calls(tree):
        cls = resolved.get(name)
        # Only in-tree classes; third-party signatures are not ours to police.
        if not isinstance(cls, type) or not getattr(cls, "__module__", "").startswith("tengri"):
            continue
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            continue

        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue  # **kwargs accepts anything

        accepted = {
            n
            for n, p in params.items()
            if n != "self"
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        required = {
            n
            for n, p in params.items()
            if n != "self"
            and p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        unknown = given - accepted
        missing = required - given
        site = f"{where}:{lineno} {name}(...)"
        if unknown:
            problems.append(f"{site} passes non-existent argument(s) {sorted(unknown)}")
        if missing:
            problems.append(f"{site} omits required argument(s) {sorted(missing)}")
    return problems


def _check_source_module(path: pathlib.Path) -> list[str]:
    """A ``src/`` module: importable, so use its namespace and its AST."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    try:
        mod = importlib.import_module(_module_name(path))
    except Exception:
        return []
    return _problems_in(tree, _resolve_names(tree, mod), str(path.relative_to(_REPO)))


def _check_test_module(path: pathlib.Path) -> list[str]:
    """A ``tests/`` module: never imported, so resolve names from the AST alone."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    return _problems_in(tree, _resolve_names(tree), str(path.relative_to(_REPO)))


def test_no_stale_constructor_calls_in_src():
    """No ``src/`` constructor call passes unknown or omits required arguments."""
    problems: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        problems.extend(_check_source_module(path))

    assert not problems, "Stale constructor call(s) — these raise TypeError when run:\n  " + (
        "\n  ".join(problems)
    )


def test_no_stale_constructor_calls_in_tests():
    """The #1919 blind spot: a fixture goes stale exactly like a ``src/`` call site.

    Arguably worse than one in ``src/``: the test that would have caught the rename is
    itself what breaks, so the failure surfaces as a red suite naming a ``TypeError``
    in a fixture rather than as the rename's own consequence.
    """
    problems: list[str] = []
    for path in sorted(_TESTS.rglob("*.py")):
        problems.extend(_check_test_module(path))

    assert not problems, "Stale constructor call(s) in tests/ — these raise TypeError:\n  " + (
        "\n  ".join(problems)
    )


def test_the_check_detects_a_stale_call():
    """Non-vacuity: "no violations" must not be reachable by inspecting nothing.

    Uses the call that turned main red in #1899 — a field renamed out from under a
    fixture — driven through the AST-only path that ``tests/`` relies on.
    """
    stale = (
        "from tengri.utils.grid_interp import PreintegratedGrid\n"
        "PreintegratedGrid(flux_scale=1.0, n_filters=3)\n"
    )
    tree = ast.parse(stale)
    problems = _problems_in(tree, _resolve_names(tree), "<synthetic>")

    assert any("flux_scale" in p and "non-existent" in p for p in problems), (
        f"the AST-only path failed to flag a renamed field; got {problems}"
    )
    assert any("log10_flux_scale" in p and "omits required" in p for p in problems), (
        f"the AST-only path failed to flag the omitted replacement; got {problems}"
    )
