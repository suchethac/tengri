# SPDX-License-Identifier: BSD-3-Clause
"""#2265: every ``DEFAULT_*`` constant the AGN parameter module exports reads its declaration.

A constant is a name for a declared default, so it must equal the declaration it names.
``tools/check_literal_param_defaults.py`` checks that call sites read such a name rather
than a numeral; this file checks the names themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tengri.components.agn import _params
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.contract


def _constant_declarations() -> dict[str, str | None]:
    """Every module-level ``DEFAULT_*`` assignment -> the declared name it reads, or ``None``.

    Every assignment is kept, whatever its shape, so a constant edited into an expression
    (not a bare ``declared_default(...)`` call) is still tested and fails by name.
    """
    tree = ast.parse(Path(_params.__file__).read_text())
    found: dict[str, str | None] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id.startswith("DEFAULT_")):
            continue
        found[target.id] = None
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Call)
                and getattr(sub.func, "id", None) == "declared_default"
                and len(sub.args) == 2
                and isinstance(sub.args[1], ast.Constant)
                and isinstance(sub.args[1].value, str)
            ):
                found[target.id] = sub.args[1].value
                break
    return found


_DECLARATIONS = _constant_declarations()


def test_every_exported_default_constant_is_declared_by_declared_default():
    exported = {name for name in _params.__all__ if name.startswith("DEFAULT_")}
    assert exported == set(_DECLARATIONS)


@pytest.mark.parametrize("constant", sorted(_DECLARATIONS))
def test_default_constant_equals_its_declaration(constant):
    declared_name = _DECLARATIONS[constant]
    assert declared_name is not None, f"{constant} is not a declared_default(...) read"
    assert getattr(_params, constant) == declared_default(_params.PARAMS, declared_name)
