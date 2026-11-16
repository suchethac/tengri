# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``DUST_EMISSION_MODELS`` is defined in exactly one module.

Two definitions of the registry dict were a defect (owner rule: two ways to do
the same thing). The live registry is ``tengri.components.dust.emission.emission``;
``tengri.components.dust.emission_templates`` must not define the name at all.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

NAME = "DUST_EMISSION_MODELS"
OWNER = "tengri.components.dust.emission.emission"
OTHER = "tengri.components.dust.emission_templates"
DUST_SRC = Path(__file__).resolve().parents[2] / "src" / "tengri" / "components" / "dust"

EXPECTED_KEYS = {
    "astrodust",
    "bosa",
    "casey2012",
    "dale2014",
    "dale2014_cigale",
    "dh02_ce01",
    "dl07",
    "dl14",
    "draine_li2007",
    "draine_li2014",
    "graybody",
    "mbb",
    "modified_blackbody",
    "pah_drude",
    "schreiber2016",
    "schreiber2018",
    "themis",
}


def _module_level_definers() -> list[str]:
    """Return dotted names of every dust module that assigns NAME at module level."""
    definers: list[str] = []
    for path in sorted(DUST_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            targets = []
            if isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = node.targets
            if any(isinstance(t, ast.Name) and t.id == NAME for t in targets):
                rel = path.relative_to(DUST_SRC.parents[2]).with_suffix("")
                definers.append(".".join(rel.parts))
                break
    return definers


def test_exactly_one_module_defines_registry_name():
    assert _module_level_definers() == [OWNER], f"{NAME} must be defined only in {OWNER}"


def test_other_module_does_not_expose_registry_name():
    module = importlib.import_module(OTHER)
    assert not hasattr(module, NAME), f"{OTHER} must not expose {NAME}"


def _fresh_registry_keys() -> set[str]:
    """Import the owner module in a clean interpreter, so other tests' registrations stay out."""
    code = (
        f"import importlib,json; m=importlib.import_module({OWNER!r}); "
        f"print(json.dumps(sorted(getattr(m, {NAME!r}))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(json.loads(out))


def test_owner_registry_has_expected_keys_and_callables():
    assert _fresh_registry_keys() == EXPECTED_KEYS
    registry = getattr(importlib.import_module(OWNER), NAME)
    for key, value in registry.items():
        assert callable(value), f"registry entry {key!r} is not callable"


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_every_key_resolves_to_callable(key):
    registry = getattr(importlib.import_module(OWNER), NAME)
    assert callable(registry[key])
