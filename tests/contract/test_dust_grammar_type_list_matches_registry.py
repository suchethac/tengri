# SPDX-License-Identifier: BSD-3-Clause
"""``check_dust_group_grammar.py``'s type literals must match what tengri accepts.

The guard recognizes a dict in docs or notebooks as a tengri dust group partly
by its ``'type'`` value, and it carries those values as literals because it is
deliberately stdlib-only: it runs in CI without importing tengri, JAX or
matplotlib. That is a reasonable constraint for the tool and a silent trap for
the list, which is a hand-maintained copy of
``tengri.parameters.groups._valid_dust_emission_types``.

A name missing from the copy does not make the guard complain. It makes the
guard **stop recognizing** that dict, so every grammar rule it exists to
enforce is skipped for the block. Nothing fails; coverage just quietly shrinks.
``graybody`` had drifted out this way.

A test may import tengri even though the tool may not, so the drift is checked
here instead. This file is the reason the literals can stay literals.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tengri.parameters.groups import _valid_dust_emission_types

pytestmark = pytest.mark.contract

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_dust_group_grammar.py"


def _literal_set(name: str) -> set[str]:
    """The names inside a ``frozenset({...})`` literal in the tool, read as text.

    Parsed rather than imported: importing the tool would work, but reading the
    source is what keeps this test honest if the literal is ever replaced by
    something computed -- at which point this test should be deleted, not made
    to pass.
    """
    source = TOOL.read_text(encoding="utf-8")
    match = re.search(rf"{name} = frozenset\(\s*\{{(.*?)\}}", source, re.S)
    assert match is not None, f"{name} is no longer a frozenset literal in {TOOL.name}"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_the_emission_type_literals_match_what_the_grammar_accepts():
    """Drift either way is a defect, and the two directions differ.

    Missing a real name silently drops grammar coverage for every doc using it.
    Carrying a name tengri does not accept is milder -- the guard would only
    recognize a dict that cannot be built anyway -- but it is still a copy that
    no longer describes its source, so both are failed here.
    """
    hand = _literal_set("DUST_EMISSION_TYPES")
    real = set(_valid_dust_emission_types())

    missing = sorted(real - hand)
    extra = sorted(hand - real)

    assert not missing, (
        f"{TOOL.name} does not list {missing}, so a dust group using one is not "
        "recognized and its grammar is never checked"
    )
    assert not extra, (
        f"{TOOL.name} lists {extra}, which _valid_dust_emission_types() does not accept"
    )
