# SPDX-License-Identifier: BSD-3-Clause
"""#1677: ``Catalog.from_histories``'s ``met=`` advice named a group that did not exist.

Passing ``met=`` to :meth:`Catalog.from_histories` on a model whose metallicity
is not tabulated raised a message telling the reader to rebuild with
``met={'type': 'table'}``. At the time there was no ``met`` group — the grammar
dispatched on agn, dust, foreground, igm, neb, radio, sfh, shock, stellar, xray
— so a user who made one mistake was handed a second one.

**#1678 fixed the message; #1720 fixed the grammar, which is where the defect
actually was.** ``stellar`` was the only group naming its structural key
something other than ``'type'``, and the only one whose name did not match what
it configured. ``met={'type': 'table'}`` — what this message said all along, and
what a reader writes by analogy with ``sfh={'type': 'table'}`` — is now the
form, and ``stellar={'met_mode': ...}`` is gone. The message was right before
the grammar was.

So these tests changed sides: they used to pin "``met`` is not a group, say
``stellar``", and now pin that ``met`` is the group and ``stellar`` is refused
with a translation. The invariant underneath is unchanged and is the whole
point of the file — **recovery advice must be a form the grammar accepts**.

The general guard is the fourth arm of
``tests/contract/test_error_message_advice_parses.py``, which reads every
literal advice snippet out of ``src/`` rather than relying on a hand-maintained
trigger list. These tests pin the specific claim that arm cannot make: that the
advice does not merely *parse* but actually selects the mode it promises.
"""

from __future__ import annotations

import pytest

from tengri.parameters.groups import parse_groups

pytestmark = [pytest.mark.regression_bug]


def test_the_advised_form_selects_the_tabulated_mode() -> None:
    """Parsing is necessary but not sufficient — it must set ``met_mode``.

    A snippet the grammar merely tolerates would satisfy the contract guard and
    still leave the user exactly where they started.
    """
    spec = parse_groups(met={"type": "table"})
    assert spec.met_mode == "table", (
        f"met={{'type': 'table'}} parsed but produced met_mode={spec.met_mode!r}; "
        f"the advice would send a stuck user in a circle."
    )


def test_met_is_a_grammar_group() -> None:
    """The premise of #1720, asserted rather than assumed.

    #1677's advice was refused because ``met`` was not a group. It is one now,
    which is what makes the advice executable rather than merely corrected.
    """
    from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

    assert "met" in _GROUP_STRUCTURAL_KEYS
    assert "type" in _GROUP_STRUCTURAL_KEYS["met"], "selects like every other group"


def test_the_replaced_spelling_is_refused_with_a_translation() -> None:
    """``stellar`` is gone, and a reader holding it must not get "unknown key".

    It was a real group one release ago and ``difflib`` will not suggest ``met``
    for it, so the generic error would be a dead end.
    """
    with pytest.raises(ValueError, match="the 'stellar' group is gone") as exc:
        parse_groups(stellar={"met_mode": "table"})
    assert "met={'type': 'table'}" in str(exc.value)


def test_the_message_names_a_form_the_grammar_accepts() -> None:
    """Read the shipped docstring and check the form it names is the live one.

    ``docs/api/*.rst`` are autodoc stubs, so this docstring *is* the published
    API reference. No assertion on wording, only on the thing a reader types.
    """
    from tengri.inference.catalog import Catalog

    source = Catalog.from_histories.__doc__ or ""
    assert "met={'type': 'table'}" in source, (
        "the from_histories docstring should name the working build form; it is "
        "the published API reference for this call"
    )
    assert "stellar={'met_mode'" not in source, (
        "the removed spelling must not survive in the published reference"
    )
    # And the named form must actually work, which is the property #1677 lacked.
    assert parse_groups(met={"type": "table"}).met_mode == "table"
