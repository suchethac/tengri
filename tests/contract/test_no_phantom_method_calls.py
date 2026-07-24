# SPDX-License-Identifier: BSD-3-Clause
"""Every ``self.<name>(...)`` call must resolve to an attribute the class actually has.

Guards the bug class that shipped ``Fitter.fit_batch(method="map")`` dead on arrival:
``_fit_batch_vmap_map`` called ``self._bounded_from_unbounded(params_i)``, a method
that exists nowhere. The module imported fine, ``py_compile`` was happy, and the line
raised ``AttributeError`` the first time a user selected that method. No test executed
it, so CI stayed green over a public API that could never work.

A "does it import?" check cannot catch this, and neither can the sibling guard in
``test_constructor_call_signatures.py`` -- that one validates constructor *arity*, not
attribute existence. Only executing the line, or resolving the attribute the way this
test does, will find it.

Hybrid by design. The AST finds call sites (cheap, and covers branches no test
reaches); the class is then imported and the attribute resolved through its real
``__mro__``, which a pure AST walk cannot do -- it has no way to know
``self._foo()`` is defined on a base class three modules away.

Precision (why this does not fire on legitimate code):

* names bound to ``self`` anywhere in the class body are attributes, not methods --
  ``self._fn = jax.jit(...)`` followed by ``self._fn(x)`` is fine;
* class-body names are excluded too, so a dataclass field annotated ``Callable`` and
  invoked as ``self.design_matrix_builder(params)`` is not a hit (this is the shape
  used in ``inference/likelihoods/marginalized.py``);
* classes with a ``__getattr__`` anywhere in the MRO are skipped, since attribute
  presence is genuinely undecidable there;
* anything the import machinery cannot resolve is skipped rather than reported, so an
  optional-dependency import error can never manufacture a failure.

Mutation-tested: reintroducing the #1253 call into ``Fitter`` makes this test fail
with ``inference.fitter.Fitter._bounded_from_unbounded``; removing it makes it pass.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

pytestmark = pytest.mark.contract

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(_SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _bound_names(node: ast.ClassDef) -> set[str]:
    """Names that are attributes rather than methods, so calling them is legitimate."""
    bound: set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
            and isinstance(sub.ctx, (ast.Store, ast.Del))
        ):
            bound.add(sub.attr)
    # Class-body bindings: `field: Callable` and `field = default` alike.
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            bound.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    return bound


def _called_on_self(node: ast.ClassDef) -> set[str]:
    return {
        sub.func.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and isinstance(sub.func.value, ast.Name)
        and sub.func.value.id == "self"
    }


def _check_module(path: pathlib.Path) -> tuple[list[str], int]:
    """Return (problems, classes_actually_resolved).

    The second element exists so the test can prove it did real work. Without it a
    broken module path or a swallowed ImportError makes this guard report "no
    problems" for every file and pass vacuously — which is exactly what happened
    during development.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return [], 0
    # _module_name already yields the fully-qualified "tengri.…" path; prefixing it
    # again produced "tengri.tengri.…", whose ImportError the except below swallowed,
    # making this whole guard pass vacuously. Caught by the mutation test.
    name = _module_name(path)
    try:
        module = importlib.import_module(name)
    except Exception:
        return [], 0  # optional dep / import side effect — never fail on this

    problems: list[str] = []
    resolved = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        cls = getattr(module, node.name, None)
        if not isinstance(cls, type):
            continue
        resolved += 1
        if any("__getattr__" in vars(base) for base in cls.__mro__):
            continue

        candidates = _called_on_self(node) - _bound_names(node)
        for attr in sorted(candidates):
            if not hasattr(cls, attr):
                problems.append(
                    f"{path.relative_to(_SRC.parent)}: {node.name}.{attr}() "
                    f"is called on self but exists on no class in the MRO"
                )
    return problems, resolved


def test_no_phantom_method_calls():
    """No ``self.x()`` call may name an attribute the class does not have."""
    problems: list[str] = []
    resolved = 0
    for path in sorted(_SRC.rglob("*.py")):
        found, n = _check_module(path)
        problems.extend(found)
        resolved += n

    # Self-check: prove the scan resolved real classes. If a module-path or import
    # regression makes every lookup fail, this fires instead of passing on an empty
    # search space. tengri had ~322 importable classes when this was written.
    assert resolved > 150, (
        f"guard resolved only {resolved} classes — it is not inspecting the codebase, "
        "so its 'no problems' result is meaningless"
    )
    assert not problems, "Calls to methods that do not exist:\n  " + "\n  ".join(problems)
