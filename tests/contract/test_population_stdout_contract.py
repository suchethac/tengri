# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the population package logs diagnostics, it does not print them (#1581).

``fit_interim`` shipped an unconditional ``[DEBUG]`` block that wrote six lines to
stdout on every call, with no verbose flag and no way for a caller to silence it.

Worth recording *what* it was debugging: the ``(expect (N, K, 16))`` hints were
scaffolding for the exact shape confusion #1575 turned up — the pilot asserted
``(N, 1000, 16)`` when ``fit_interim`` returns ``n_samples // thin`` draws.
Someone hit the confusion, added prints to see the shapes, and both the prints
and the wrong assertion stayed. A print added while debugging is the kind of
thing that survives precisely because nothing fails when it does.

The scan is AST-based rather than textual: a regex for ``print(`` also matches
the word inside docstrings, comments, and string literals, and would have to be
un-taught each time one appears.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tengri.inference.population as population

pytestmark = pytest.mark.contract

_PACKAGE_DIR = Path(population.__file__).parent

_SYNTHETIC_OFFENDER = """
def f():
    print("this one is real")           # line 3

def g():
    '''A docstring that says print( and must not count.'''
    x = "print(also not a call)"
    return x
"""


def _print_call_sites(path, source=None):
    """Every ``print(...)`` call site in a module, as ``"name:line"`` strings."""
    text = source if source is not None else path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    return [
        f"{path.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


class TestTheInstrument:
    """A scanner that cannot find a print would pass this file for free."""

    def test_it_finds_a_real_call(self):
        found = _print_call_sites(Path("synthetic.py"), source=_SYNTHETIC_OFFENDER)
        assert found == ["synthetic.py:3"]

    def test_it_ignores_the_word_in_docstrings_and_literals(self):
        """The reason this is an AST walk and not a grep."""
        found = _print_call_sites(Path("synthetic.py"), source=_SYNTHETIC_OFFENDER)
        assert len(found) == 1


class TestTheContract:
    def test_there_is_source_to_scan(self):
        """An empty scan is not evidence of absence. If the package layout moves,
        this fails loudly instead of the ban below passing vacuously."""
        modules = sorted(_PACKAGE_DIR.glob("*.py"))
        assert len(modules) >= 5, [p.name for p in modules]
        assert (_PACKAGE_DIR / "interim.py").is_file()

    def test_no_module_writes_to_stdout(self):
        """Report the offending sites, not their count — a bare number sends the
        next reader back to re-derive which ones they were."""
        offenders = []
        for path in sorted(_PACKAGE_DIR.glob("*.py")):
            offenders.extend(_print_call_sites(path))
        assert offenders == [], (
            f"Library code must not write to stdout; use the module logger. "
            f"Offending call sites: {offenders}"
        )
