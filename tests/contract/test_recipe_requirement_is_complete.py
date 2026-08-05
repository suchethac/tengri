# SPDX-License-Identifier: BSD-3-Clause
"""A discovery field must not hand the user half a sentence.

``list_recipes()`` carries an ``ssp_requirement`` column so a new user can tell
which SSP grid a recipe needs before building anything. The parser read a
single physical line of a docstring paragraph that numpydoc wraps at the line
limit, so eight of ten recipes rendered as a fragment — five of them as
``'bare-stellar (Cue nebular backend; see'``, an unclosed parenthesis and a
cross-reference pointing at nothing.

The truncation was invisible because a fragment is still a plausible string:
the field was never empty, so no emptiness check fired. What it cut was the
consequence clause every docstring puts second — ``doing so raises
CueWNESSPError`` for :func:`star_forming_photometry`, and the warning about
dropping the nebular contribution for :func:`high_z`.

The guard is the rule, not the eight instances: *every* recipe's requirement
must read as a finished sentence. A future recipe whose requirement wraps is
covered the day it lands.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.registry import _parse_ssp_requirement

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

#: Below this the sweep is not exercising anything and the assertions are
#: vacuous — the count is asserted, not assumed.
_MIN_RECIPES = 5


def _requirements() -> list[tuple[str, str]]:
    return [(r["name"], r["ssp_requirement"]) for r in tengri.list_recipes()]


def test_the_sweep_is_not_vacuous():
    rows = _requirements()
    assert len(rows) >= _MIN_RECIPES, (
        f"only {len(rows)} recipes — guard is not exercising anything"
    )


@pytest.mark.parametrize("name,req", _requirements())
class TestEveryRequirementReadsAsAFinishedSentence:
    def test_ends_with_terminal_punctuation(self, name, req):
        """A value ending in ``,`` ``;`` ``(`` or a dangling word is truncated.

        ``rstrip(")")`` because a sentence may close inside a parenthetical —
        ``"(… carries no baked nebular lines.)"`` is finished, not cut.
        """
        assert req.rstrip(")").endswith("."), (
            f"recipes.{name}: ssp_requirement does not end in a full stop, so it was "
            f"cut mid-sentence: {req!r}"
        )

    def test_parentheses_are_balanced(self, name, req):
        """``'bare-stellar (Cue nebular backend; see'`` — the original symptom."""
        assert req.count("(") == req.count(")"), (
            f"recipes.{name}: unbalanced parentheses, so the value is a fragment: {req!r}"
        )

    def test_carries_no_raw_rest_markup(self, name, req):
        """The column renders in a terminal, never through Sphinx."""
        for markup in ("``", ":func:", ":meth:", ":class:", ":mod:"):
            assert markup not in req, (
                f"recipes.{name}: raw reST {markup!r} reaches the user's table: {req!r}"
            )


class TestTheParserReadsTheWholeParagraph:
    """Direct pin on the mechanism, so the guard fails on the old implementation.

    The sweep above passes for a recipe whose requirement happens to fit on one
    line; only a wrapped input distinguishes a paragraph reader from a line
    reader.
    """

    def test_a_wrapped_requirement_is_not_cut_at_the_line_break(self):
        doc = (
            "Summary line.\n"
            "\n"
            "**SSP requirement:** bare-stellar (e.g., ``a.h5``,\n"
            "``b.h5``). Pairing it with wNE raises ``SomeError``.\n"
            "\n"
            "**Configuration:**\n"
        )
        got = _parse_ssp_requirement(doc)
        assert got.endswith("Pairing it with wNE raises SomeError."), got
        assert "see" not in got

    def test_a_following_field_terminates_the_paragraph(self):
        """No blank line between fields must not swallow the next field."""
        doc = "**SSP requirement:** any.\n**Data requirement:** none of your business.\n"
        got = _parse_ssp_requirement(doc)
        assert got == "any."

    def test_a_recipe_without_the_tag_defaults_to_any(self):
        assert _parse_ssp_requirement("Just a summary.\n") == "any"
