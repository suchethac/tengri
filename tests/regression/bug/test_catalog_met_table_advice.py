# SPDX-License-Identifier: BSD-3-Clause
"""#1677: ``Catalog.from_histories``'s ``met=`` advice named a group that does not exist.

Passing ``met=`` to :meth:`Catalog.from_histories` on a model whose metallicity is not
tabulated raised a message telling the reader to rebuild with
``met={'type': 'table'}``. There is no ``met`` group — the grammar dispatches on
agn, dust, foreground, igm, neb, radio, sfh, shock, stellar, xray — so a user who
made one mistake was handed a second one.

``"table"`` is the single metallicity mode that *cannot* be inferred from
parameter names, because it declares no fittable parameters
(``met_registry._MET_MODE_DISCRIMINATORS``). It has to be named explicitly, and
the way to name it through the build grammar is ``stellar={'met_mode': 'table'}``.

The general guard is the fourth arm of
``tests/contract/test_error_message_advice_parses.py``, which reads every
literal advice snippet out of ``src/`` rather than relying on a hand-maintained
trigger list. These tests pin the specific claim that arm cannot make: that the
replacement advice does not merely *parse* but actually selects the mode the
message promises.
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
    spec = parse_groups(stellar={"met_mode": "table"})
    assert spec.met_mode == "table", (
        f"stellar={{'met_mode': 'table'}} parsed but produced met_mode="
        f"{spec.met_mode!r}; the advice would send a stuck user in a circle."
    )


def test_the_old_advice_is_still_refused() -> None:
    """Pin the reason the message changed, so it cannot quietly come back."""
    with pytest.raises(ValueError, match=r"Unknown group key 'met'"):
        parse_groups(met={"type": "table"})


def test_met_is_not_a_grammar_group() -> None:
    """The premise of the fix, asserted rather than assumed."""
    with pytest.raises(ValueError) as exc:
        parse_groups(met={"type": "table"})
    valid = str(exc.value).split("Valid groups:")[-1]
    assert "stellar" in valid and " met," not in valid


def test_the_message_names_a_form_the_grammar_accepts() -> None:
    """Read the shipped message and feed its own suggestion back through.

    This is the end-to-end version of the fix: no assertion on wording, only on
    the thing the reader will actually type.
    """
    from tengri.inference.catalog import Catalog

    source = Catalog.from_histories.__doc__ or ""
    assert "stellar={'met_mode': 'table'}" in source, (
        "the from_histories docstring should name the working build form; it is the "
        "published API reference for this call"
    )
    assert "met={'type': 'table'}" not in source
