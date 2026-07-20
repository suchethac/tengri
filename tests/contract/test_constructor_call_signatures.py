# SPDX-License-Identifier: BSD-3-Clause
"""Every in-tree constructor call must match the class's real ``__init__`` signature.

Guards a rot class that reached main twice and was invisible to CI both times, because
the offending line lived on a path no test ever executed:

* ``Fitter._fit_batch_vmap_map`` built ``Posterior(log_weights=..., fitter=...)`` —
  neither is a parameter — while omitting the required ``params`` / ``wall_time_s``.
* ``cb19_precompute.precompute`` and ``feltre_precompute.precompute`` built
  ``PreintegratedGrid(...)`` with 3 of 8 required arguments, after #1122/#1133 added
  the extra fields without updating these call sites.

A "does it import?" or "is it callable?" check cannot catch either — only executing the
line, or checking the call against the signature the way this test does. This needs no
``data/`` files, so unlike an end-to-end precompute test it actually runs on CI.

Precision (the reasons this does not fire on legitimate code):

* the class name is resolved in the **calling module's own namespace**, so two classes
  sharing a name never get confused;
* validation uses :func:`inspect.signature`, not ``dataclasses.fields`` — classes with a
  hand-written ``__init__`` whose parameters differ from their fields (e.g.
  ``PhotometryLikelihood(fnu_obs=...)``) are checked against what they actually accept;
* calls using positional args, ``*args`` or ``**kwargs`` unpacking are skipped as
  unverifiable, as are classes whose ``__init__`` accepts ``**kwargs``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib

import pytest

pytestmark = pytest.mark.contract

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


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


def _resolve_names(tree: ast.AST, mod) -> dict[str, type]:
    """Map every class name usable in this module to the real class object.

    Module-level ``getattr`` alone is not enough: tengri imports many symbols
    *inside functions* to break import cycles (e.g. ``Posterior`` in ``fitter.py``),
    and those never appear in the module namespace. Missing them would make this
    guard silently skip the very call sites most likely to have rotted — a guard
    that fails open. So walk every ``from X import Y`` at any nesting depth too.
    """
    names: dict[str, type] = {}
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


def _check_module(path: pathlib.Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    try:
        mod = importlib.import_module(_module_name(path))
    except Exception:
        return []

    resolved = _resolve_names(tree, mod)
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
        where = f"{path.relative_to(_SRC.parent)}:{lineno} {name}(...)"
        if unknown:
            problems.append(f"{where} passes non-existent argument(s) {sorted(unknown)}")
        if missing:
            problems.append(f"{where} omits required argument(s) {sorted(missing)}")
    return problems


def test_no_stale_constructor_calls():
    """No in-tree constructor call passes unknown or omits required arguments."""
    problems: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        problems.extend(_check_module(path))

    assert not problems, "Stale constructor call(s) — these raise TypeError when run:\n  " + (
        "\n  ".join(problems)
    )
