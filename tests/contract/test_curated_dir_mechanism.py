# SPDX-License-Identifier: BSD-3-Clause
"""One implementation behind every curated ``__dir__`` (#1431).

Fourteen namespaces trim their tab-completion to a curated list (#1288). Each
carried its own nine-line ``def __dir__``, and the copies had already drifted
into two spellings -- ``sorted(__all__)`` in nine, ``list(__all__)`` in five --
which no caller can tell apart, because :func:`dir` sorts whatever ``__dir__``
returns. They now all bind :func:`tengri._completion.curated_dir`.

Sibling surface: ``test_curated_dir_surface.py`` pins *which symbols* the
top-level menu offers (#1455). This file pins *the mechanism* -- that every
namespace curates through the one helper, and that ``dir()`` still reports the
curated names.

The two guards that keep this from decaying:

* :func:`test_no_module_hand_rolls_dir` fails on any new module-level ``def
  __dir__``, so the boilerplate cannot grow back one namespace at a time.
* :func:`test_the_census_is_complete` fails when a namespace starts curating
  without joining the census below, so the guard cannot silently narrow to the
  fourteen that happened to exist when it was written.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tengri"

#: Namespaces whose completion menu is exactly their ``__all__``.
ALL_BACKED = (
    "tengri.builders",
    "tengri.builders.agn",
    "tengri.builders.dust",
    "tengri.builders.igm",
    "tengri.builders.neb",
    "tengri.builders.radio",
    "tengri.builders.sfh",
    "tengri.builders.xray",
    "tengri.components.agn",
    "tengri.components.nebular",
    "tengri.components.stellar.sfh",
    "tengri.recipes",
)

#: Namespaces that advertise less than they re-export, so the menu is a
#: separate ``_CURATED_DIR`` tuple rather than ``__all__``.
CURATED_BACKED = (
    "tengri",
    "tengri.components.dust",
)

CURATING = ALL_BACKED + CURATED_BACKED

#: Names that leaked into completion before #1288 -- a module's own imports and
#: internal helpers, sitting beside the physics.
INCIDENTAL = ("annotations", "Any", "Callable", "make_factory", "short_form", "inspect")


# ── the mechanism ────────────────────────────────────────────────


@pytest.mark.parametrize("modname", CURATING)
def test_namespace_curates_through_the_one_helper(modname):
    """A namespace that curates must do it through ``curated_dir``."""
    mod = importlib.import_module(modname)
    dunder = mod.__dict__.get("__dir__")
    assert dunder is not None, f"{modname} lost its curated __dir__ entirely"
    assert dunder.__qualname__.startswith("curated_dir"), (
        f"{modname} defines its own __dir__ ({dunder.__qualname__}); bind "
        "tengri._completion.curated_dir instead"
    )


@pytest.mark.parametrize("modname", ALL_BACKED)
def test_dir_reports_exactly_all(modname):
    mod = importlib.import_module(modname)
    assert dir(mod) == sorted(mod.__all__)


@pytest.mark.parametrize("modname", CURATED_BACKED)
def test_dir_reports_exactly_the_curated_tuple(modname):
    mod = importlib.import_module(modname)
    assert dir(mod) == sorted(mod._CURATED_DIR)


@pytest.mark.parametrize("modname", CURATING)
def test_incidental_imports_stay_hidden(modname):
    """The point of #1288: a namespace must not complete its own imports."""
    leaked = sorted(set(dir(importlib.import_module(modname))) & set(INCIDENTAL))
    assert leaked == [], f"{modname} offers incidental names in completion: {leaked}"


# ── the guards that keep this from decaying ──────────────────────


def _hand_rolled_dirs() -> list[str]:
    """Every module-level ``def __dir__`` under ``src/tengri``.

    Module level only. A ``__dir__`` *method* on a class is a different thing:
    it composes instance state (``Posterior`` appends its model's property
    names) and has no boilerplate to share.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # top level only -- not ast.walk
            if isinstance(node, ast.FunctionDef) and node.name == "__dir__":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return offenders


def test_no_module_hand_rolls_dir():
    """The boilerplate must not grow back one namespace at a time."""
    offenders = _hand_rolled_dirs()
    assert offenders == [], (
        "hand-rolled module-level __dir__ found — bind "
        "`__dir__ = curated_dir(__all__)` instead:\n  " + "\n  ".join(offenders)
    )


def _modules_binding_dir() -> set[str]:
    """Every module under ``src/tengri`` that assigns ``__dir__`` at top level."""
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else []
            if not any(isinstance(t, ast.Name) and t.id == "__dir__" for t in targets):
                continue
            rel = path.relative_to(SRC)
            parts = rel.parts[:-1] if rel.name == "__init__.py" else (*rel.parts[:-1], rel.stem)
            found.add(".".join(("tengri", *parts)))
    return found


def test_the_census_is_complete():
    """A new curating namespace must join the census, not slip past it.

    Without this, the parametrized tests above stay green while covering only
    the fourteen namespaces that happened to exist when they were written.
    """
    assert _modules_binding_dir() == set(CURATING)
