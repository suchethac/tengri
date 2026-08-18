#!/usr/bin/env python3
"""CI guard: a MagicMock standing in for an SEDModel must declare ``observation``.

``Fitter._build_data_args`` reads the optional observation channels the way you
read optional config off any real object::

    obs = getattr(model, "observation", None)
    if obs is not None:
        line_ratio_cfg = getattr(obs, "line_ratios", None)
        if line_ratio_cfg is not None:
            args["line_ratio_obs"] = line_ratio_cfg.ratios
            ...

That is correct against a real model and **unsatisfiable against a bare
MagicMock**: every attribute auto-vivifies, so the ``None`` default is
unreachable and each ``is not None`` guard passes. A photometry-only stub then
silently claims spectroscopy covariance, line fluxes, line ratios and spectral
indices. Measured on the #1942 stub: nine phantom ``_data_args`` keys and four
likelihood channels where one was intended.

Nothing noticed for as long as nothing evaluated those channels. The eager
channel-scale pre-check (#1495, merged via #1905) does — it runs at loss-build
time on *every* fit, because ``Fitter`` auto-builds a protocol likelihood — and
it is deliberately fallback-free, so the phantom channels became four hard
failures in ``test_inference_context.py`` (#1942). The pre-check was right; the
stub was lying.

This is the over-declaring half of a two-sided class. The under-declaring half
is a stub *missing* a method the contract requires — ``_MockSpec.sample`` in
#1931 and again in #1942 — and it announces itself with an ``AttributeError``
the first time anything calls it. Over-declaration is the dangerous half: it
fails open, silently widening what the stub claims to be, and it cannot be
caught by asking the mock what it supports, because the answer is always yes.

So the rule is a declaration requirement: a MagicMock playing an SEDModel must
say what it does NOT have.

    model = MagicMock()
    model.spec = spec
    model.predict_photometry.return_value = ...
    model.observation = None                       # or a namespace of Nones

AST, not grep: the assignment that creates the mock, the assignments that shape
it, and the ``Fitter(...)`` call that consumes it routinely sit in different
functions — the violating file this guard was written against builds its mock in
a ``_make_mock_model`` helper and constructs the fitter forty lines away.

Usage
-----
    python tools/check_mock_model_declares_observation.py

Exit code 0 on success; 1 with violations listed otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

#: Constructors whose FIRST positional argument is the SED model.
FITTER_CONSTRUCTORS: frozenset[str] = frozenset({"Fitter", "PopulationFitter", "CatalogFitter"})

#: Attributes that mark a mock as standing in for an SEDModel. ``spec`` plus any
#: one of these is the stub signature; a mock carrying it is model-shaped no
#: matter what the variable is called or which function consumes it.
MODEL_ATTRS: frozenset[str] = frozenset(
    {
        "predict_photometry",
        "predict_spectrum",
        "predict_state",
        "predict_line_fluxes",
        "predict_properties",
    }
)

#: The attribute that must be declared.
REQUIRED_ATTR = "observation"

#: Files exempt from the rule, each with the reason. A mock that never reaches
#: a fitter's data-args builder cannot grow phantom channels; add an entry only
#: when that is true and say why.
EXEMPT_FILES: dict[str, str] = {}


def _bare_magicmock_targets(tree: ast.Module) -> set[str]:
    """Names bound to a no-argument ``MagicMock()`` call.

    A ``MagicMock(spec=SEDModel)`` is already constrained — autospec raises on
    an attribute the real class lacks — so only the argument-free form is at
    issue here.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if callee != "MagicMock" or call.args or call.keywords:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _assigned_attributes(tree: ast.Module) -> dict[str, set[str]]:
    """Map ``name -> {attributes assigned on it}``.

    ``model.spec = ...`` and ``model.predict_photometry.return_value = ...`` both
    count as shaping ``model``; the latter arrives as a nested Attribute, so walk
    down to the root Name.
    """
    shaped: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            outermost = target.attr
            inner: ast.expr = target.value
            while isinstance(inner, ast.Attribute):
                outermost = inner.attr
                inner = inner.value
            if isinstance(inner, ast.Name):
                shaped.setdefault(inner.id, set()).add(outermost)
    return shaped


def _fitter_first_args(tree: ast.Module) -> set[str]:
    """Names passed as the first positional argument to a fitter constructor."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if callee in FITTER_CONSTRUCTORS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name):
                names.add(first.id)
    return names


def scan(path: Path) -> list[str]:
    """Return violation messages for one test file."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return []

    bare = _bare_magicmock_targets(tree)
    if not bare:
        return []

    shaped = _assigned_attributes(tree)
    consumed = _fitter_first_args(tree)

    violations: list[str] = []
    for name in sorted(bare):
        attrs = shaped.get(name, set())
        is_model = ("spec" in attrs and attrs & MODEL_ATTRS) or name in consumed
        if not is_model or REQUIRED_ATTR in attrs:
            continue
        rel = path.relative_to(REPO_ROOT)
        violations.append(
            f"{rel}: `{name} = MagicMock()` is shaped as an SEDModel "
            f"({', '.join(sorted(attrs & (MODEL_ATTRS | {'spec'}))) or 'passed to a fitter'}) "
            f"but never declares `{name}.{REQUIRED_ATTR}`.\n"
            f"      Every `getattr(obs, ..., None)` in Fitter._build_data_args then "
            f"returns a truthy Mock instead of None, so the fitter builds line-flux, "
            f"line-ratio and spectral-index channels this stub cannot serve.\n"
            f"      Fix: `{name}.{REQUIRED_ATTR} = None`, or a SimpleNamespace whose "
            f"optional channels are explicitly None."
        )
    return violations


def main() -> int:
    violations: list[str] = []
    scanned = 0
    seen_exempt: set[str] = set()

    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in EXEMPT_FILES:
            seen_exempt.add(rel)
            continue
        scanned += 1
        violations.extend(scan(path))

    for rel in sorted(set(EXEMPT_FILES) - seen_exempt):
        violations.append(
            f"stale EXEMPT_FILES entry `{rel}` in {Path(__file__).name}: the file is "
            "gone, so the exemption now covers nothing and would silently excuse the "
            "next real one. Drop the entry."
        )

    if violations:
        print("MagicMock SED models that do not declare `observation`:\n", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}\n", file=sys.stderr)
        return 1

    n_exempt = len(EXEMPT_FILES)
    print(
        f"OK: every MagicMock standing in for an SEDModel declares `observation` "
        f"({scanned} test files scanned, {n_exempt} exempt)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
