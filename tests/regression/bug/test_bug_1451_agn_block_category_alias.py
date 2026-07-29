# SPDX-License-Identifier: BSD-3-Clause
"""list_agn_blocks must answer to the grammar key, and must not fail open (#1451).

Two defects in one verb:

* Five of the six composable slots have ``category == grammar key``. The sixth
  does not: the registry label is ``'attenuation'`` while the build-grammar key
  is ``'atten'`` — and ``list_agn_blocks`` prints ``agn={'atten': ...}`` in its
  own ``use:`` field. Filtering compared the raw argument against the label, so
  the function returned an empty table for the exact name it had just told the
  user to type.
* The filter fell through to "nothing matched", so a typo, an empty string and
  a valid-but-wrong key were indistinguishable from a category that genuinely
  has no blocks. A discovery verb that answers "nothing here" to a misspelling
  is the same failure class as #1275/#1276 — a guard that fails open.
"""

import pytest

import tengri

pytestmark = pytest.mark.regression_bug

_SLOTS = ["disc", "torus", "nlr", "blr", "feii"]


@pytest.mark.parametrize("category", [*_SLOTS, "atten", "attenuation"])
def test_every_grammar_key_returns_blocks(category):
    """No selectable slot may return an empty table for its own key."""
    rows = tengri.list_agn_blocks(category=category)
    assert len(rows) > 0, (
        f"list_agn_blocks(category={category!r}) is empty — the grammar accepts "
        "this key, so the discovery verb must answer to it"
    )


def test_atten_and_attenuation_agree():
    """The grammar key and the registry label must name the same set."""
    by_key = {r["name"] for r in tengri.list_agn_blocks(category="atten")}
    by_label = {r["name"] for r in tengri.list_agn_blocks(category="attenuation")}
    assert by_key == by_label, (by_key, by_label)


def test_the_advertised_use_key_is_queryable():
    """Close the loop: parse the key out of the row's own hint and query it.

    This is the property that actually broke — not "atten works" in the
    abstract, but that the string the table prints can be handed straight back
    to the function. Derived from the hint rather than hard-coded, so it keeps
    holding if the advertised key ever changes.
    """
    import re

    for row in tengri.list_agn_blocks():
        match = re.search(r"agn=\{'(\w+)':", row["use"])
        assert match, f"unparseable use string: {row['use']!r}"
        advertised = match.group(1)
        assert len(tengri.list_agn_blocks(category=advertised)) > 0, (
            f"{row['name']!r} advertises agn={{{advertised!r}: ...}} but "
            f"list_agn_blocks(category={advertised!r}) is empty"
        )


@pytest.mark.parametrize("bogus", ["bogus_not_a_category", "", "Atten", "DISC"])
def test_an_unknown_category_raises_instead_of_returning_empty(bogus):
    """An empty table must mean "no blocks", never "I did not understand you"."""
    with pytest.raises(ValueError) as excinfo:
        tengri.list_agn_blocks(category=bogus)
    message = str(excinfo.value)
    # The message has to be actionable, not merely loud.
    assert "disc" in message and "torus" in message, message


def test_no_category_still_lists_everything():
    """Guard against the fail-loud fix over-reaching onto the default path."""
    assert len(tengri.list_agn_blocks()) > len(tengri.list_agn_blocks(category="disc"))
