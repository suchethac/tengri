# SPDX-License-Identifier: BSD-3-Clause
"""The three AGNfitter-rX build scripts' provenance defaults track the pinned ref.

``scripts/build_cat3d_wind_grid.py``, ``scripts/build_nk08_agnfitter_grid.py``
and ``scripts/build_schreiber2018_grid.py`` each write a ``source_label``
(``source_pickle`` / ``source_dir`` attribute) recording which file inside the
pinned upstream AGNfitter-rX archive a vendored grid was reduced from. All
three ``build()`` signatures default that label to
``scripts/_agnfitter_download.archive_relpath(repo_relpath)`` --
``AGNFITTER_REF`` plus the file's path inside the archive -- rather than a
literal ``"AGNfitter-rX_v0.1/..."`` string. The two scripts with ``--download``
CLI support (cat3d, nk08) already computed the label this way at their CLI
call site; only their unused function defaults duplicated the constant. The
third (schreiber2018) has no such CLI override, so its default is the label
every normal invocation actually writes.

Duplicating ``AGNFITTER_REF`` as a literal is a silent-drift trap: a future
ref bump (``AGNfitter-rX_v0.2``, say) moves the constant and everything that
calls ``archive_relpath`` with it, but leaves these three literal defaults
pointing at the old tag with nothing to fail. A plain value comparison cannot
catch that today -- the literal and the derived value are, correctly, the
same string right now -- so the load-bearing check parses each script's
source and asserts the default is a *call* to ``archive_relpath(...)``, not a
string constant that merely happens to agree with it.

This test does not run any of the three ``build()`` functions -- each needs
an upstream FITS/pickle archive this environment does not have -- it only
inspects the declared default, by source and by value.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path

import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from _agnfitter_download import archive_relpath

pytestmark = pytest.mark.contract

_CASES = [
    ("build_cat3d_wind_grid", "models/TORUS/CAT3D_mean_3p.pickle"),
    ("build_nk08_agnfitter_grid", "models/TORUS/NK0_mean_1p.pickle"),
    ("build_schreiber2018_grid", "models/STARBURST"),
]


def _source_label_default_node(module_name: str) -> ast.expr:
    """The AST node for ``build``'s ``source_label`` default expression.

    Reads the default from the *source text*, not the live signature, so a
    literal string that happens to equal ``archive_relpath(...)``'s output
    today is still distinguishable from an actual call to it.
    """
    path = _SCRIPTS_DIR / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build":
            arg_names = [a.arg for a in node.args.args]
            defaults = node.args.defaults
            # Defaults align to the tail of the positional/keyword arg list.
            for name, default in zip(arg_names[len(arg_names) - len(defaults) :], defaults):
                if name == "source_label":
                    return default
            raise AssertionError(f"{module_name}.build has no 'source_label' parameter")
    raise AssertionError(f"{module_name} declares no 'build' function")


@pytest.mark.parametrize("module_name, repo_relpath", _CASES)
def test_the_default_is_a_call_to_archive_relpath_not_a_literal(module_name, repo_relpath):
    """The load-bearing check: the default expression is a call, not a string.

    A stale literal and a derived call can print the same value today (they
    do); only the source-level shape tells them apart, which is exactly what
    would let a future AGNFITTER_REF bump leave a literal silently behind.
    """
    node = _source_label_default_node(module_name)
    is_derived_call = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "archive_relpath"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == repo_relpath
    )
    assert is_derived_call, (
        f"{module_name}.py's build(source_label=...) default is "
        f"{ast.dump(node)}, not archive_relpath({repo_relpath!r}) -- a "
        "literal here agrees with the constant today by coincidence, and a "
        "future AGNFITTER_REF bump would leave it silently stale."
    )


@pytest.mark.parametrize("module_name, repo_relpath", _CASES)
def test_the_default_value_matches_archive_relpath_today(module_name, repo_relpath):
    """Documents that, today, the derived value is exactly what shipped."""
    module = importlib.import_module(module_name)
    default = inspect.signature(module.build).parameters["source_label"].default
    expected = archive_relpath(repo_relpath)
    assert default == expected, (
        f"{module_name}.build's source_label default is {default!r}, but "
        f"archive_relpath({repo_relpath!r}) is {expected!r}."
    )


def test_every_build_script_with_a_literal_default_is_covered():
    """No fourth script picks up the same stale-literal pattern unnoticed."""
    import re

    covered = {name for name, _ in _CASES}
    literal_pattern = re.compile(r'source_label:\s*str\s*=\s*"AGNfitter-rX')
    offenders = []
    for path in sorted(_SCRIPTS_DIR.glob("build_*.py")):
        if path.stem in covered:
            continue
        if literal_pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == [], (
        f"found a literal 'AGNfitter-rX' source_label default not covered by "
        f"this test: {offenders}. Add it to _CASES above and derive it from "
        "archive_relpath instead."
    )
