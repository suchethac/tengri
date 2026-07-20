# SPDX-License-Identifier: BSD-3-Clause
"""Recovery advice inside a grammar error must itself be accepted.

An error message is the one piece of documentation a user is *guaranteed* to
read, because they only see it when already stuck. When its suggested fix also
raises, the user is bounced from one error to another with nothing to try next.

That shipped: mixing the legacy ``radio={'type': ...}`` key with the ``sf`` /
``agn`` sub-blocks raised a message recommending ``radio={'type': 'bell2003'}``,
and the very next validator rejected it — ``bell2003`` is an ``sf`` variant,
never a legacy ``type``. The only accepted legacy types are ``condon92`` and
``none``.

The guard is deliberately general: it pulls every ``group={...}`` snippet out of
the raised message and feeds it back through :func:`parse_groups`. Anything the
grammar suggests, the grammar must accept.

The same rule covers the library's other two channels of build advice, which
are equally unexecuted by anything else:

* the ``use:`` hint on every discovery-menu row (advertised, not merely raised);
* ``describe()``'s "registered in N places" note, whose hard-coded
  ``describe_agn_block(...)`` suggestion silently became wrong for radio blocks
  once a second categorized menu existed (#1276).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tengri.parameters.groups import parse_groups

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _dict_snippets(message: str) -> list[tuple[str, dict]]:
    """Every ``name={...}`` python-dict literal in ``message``.

    Brace-matched rather than regex-matched: the suggestions nest
    (``radio={'sf': {'type': 'bell2003'}}``) and a non-greedy regex stops at the
    first inner ``}``, silently checking a truncated snippet that happens to
    parse — a guard that passes for the wrong reason.
    """
    out: list[tuple[str, dict]] = []
    for i, ch in enumerate(message):
        if ch != "{":
            continue
        j = message.rfind("=", 0, i)
        if j == -1 or j != i - 1:
            continue
        k = j - 1
        while k >= 0 and (message[k].isalnum() or message[k] == "_"):
            k -= 1
        name = message[k + 1 : j]
        if not name:
            continue
        depth, end = 0, None
        for m in range(i, len(message)):
            if message[m] == "{":
                depth += 1
            elif message[m] == "}":
                depth -= 1
                if depth == 0:
                    end = m
                    break
        if end is None:
            continue
        try:
            value = ast.literal_eval(message[i : end + 1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            out.append((name, value))
    return out


# Each entry provokes a grammar error that offers recovery advice.
TRIGGERS = [
    pytest.param(
        {"radio": {"type": "condon92", "sf": {"type": "bell2003"}}},
        id="radio-mixes-legacy-type-with-subblocks",
    ),
    pytest.param({"radio": True}, id="radio-bool-gate-form"),
    pytest.param({"xray": True}, id="xray-bool-gate-form"),
    pytest.param({"shock": True}, id="shock-bool-gate-form"),
]


@pytest.mark.parametrize("kwargs", TRIGGERS)
def test_error_message_advice_is_itself_valid(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)) as excinfo:
        parse_groups(**kwargs)

    snippets = _dict_snippets(str(excinfo.value))
    assert snippets, (
        f"no group={{...}} advice found in the message for {kwargs!r}; either the "
        f"message stopped offering a fix or the extractor rotted. Message: {excinfo.value}"
    )

    for name, value in snippets:
        try:
            parse_groups(**{name: value})
        except Exception as exc:
            pytest.fail(
                f"the error message for {kwargs!r} recommends {name}={value!r}, "
                f"but that raises {type(exc).__name__}: {exc}. Recovery advice "
                f"must be accepted by the grammar it describes."
            )


def _skip_reason(name: str, value: dict, status: str) -> str | None:
    """Why this advice is exempt from "must be accepted", or ``None``.

    Two exemptions, both principled rather than a name allowlist:

    * ``status == "unvalidated"`` — the menu already tells the reader, in the
      same row as the hint, that the builder refuses this entry. The advice is
      not lying; the row is self-consistent.
    * the hint names a file that does not exist — e.g. the CLOUDY backend's
      ``gridfile='grid.h5'`` placeholder. Its job is to teach the *key*, and no
      literal path could be correct for every install.
    """
    if status == "unvalidated":
        return "menu flags the entry as unvalidated"
    for v in value.values():
        if isinstance(v, str) and Path(v).suffix and not Path(v).exists():
            return f"hint names a placeholder path {v!r}"
    return None


def test_every_menu_usage_hint_is_accepted_by_the_grammar() -> None:
    """A ``use:`` hint is advice too — and it is advertised, not just raised.

    Every menu row carries a copy-pasteable ``SEDModel.build(...)`` call. This
    walks all of them and feeds the group dict back through the grammar, so a
    menu can never advertise a spelling the builder rejects (#1179), and a hint
    can never drift from the key it names.
    """
    from tengri.registry import _menu_listers

    checked = 0
    failures: list[str] = []
    for lister in _menu_listers():
        for row in lister():
            for name, value in _dict_snippets(row.get("use", "")):
                if _skip_reason(row["name"], value, row.get("status", "")):
                    continue
                checked += 1
                try:
                    parse_groups(**{name: value})
                except Exception as exc:
                    failures.append(
                        f"{lister.__name__} row {row['name']!r} advertises "
                        f"{name}={value!r} -> {type(exc).__name__}: {exc}"
                    )

    assert checked > 50, f"only {checked} hints checked — the extractor probably rotted"
    assert not failures, "menus advertise build calls the grammar rejects:\n" + "\n".join(failures)


def test_ambiguity_note_advice_is_accepted_by_the_grammar() -> None:
    """``describe()``'s "registered in N places" note must also be executable.

    It quotes each alternative's own build call; those must parse for the same
    reason the menu hints must.
    """
    import tengri
    from tengri.registry import _menu_listers

    names = {name for fn in _menu_listers() for name in fn().names()}
    notes = {n: dict(tengri.describe(n)).get("also_registered_as") for n in names}
    notes = {n: note for n, note in notes.items() if note}
    assert notes, "no ambiguous names found — this guard would pass vacuously"

    # How many build calls each note *owes* the reader: one per menu row that
    # claims the name and advertises a group dict. Without this the test is
    # vacuous — the note this replaced named a helper function and contained no
    # ``{...}`` at all, so the parse loop below ran zero times and passed.
    owed: dict[str, int] = {}
    for fn in _menu_listers():
        for row in fn():
            if row["name"] in notes and "={" in row.get("use", ""):
                owed[row["name"]] = owed.get(row["name"], 0) + 1

    for owner, note in notes.items():
        snippets = _dict_snippets(note)
        assert len(snippets) >= owed.get(owner, 1), (
            f"describe({owner!r})'s note names {owed.get(owner, 1)} menu entries but "
            f"offers only {len(snippets)} build call(s) — it is describing the "
            f"alternatives without showing how to reach them. Note: {note}"
        )
        for name, value in snippets:
            if _skip_reason(owner, value, ""):
                continue
            try:
                parse_groups(**{name: value})
            except Exception as exc:
                pytest.fail(
                    f"describe({owner!r})'s ambiguity note recommends {name}={value!r}, "
                    f"but that raises {type(exc).__name__}: {exc}"
                )


def test_snippet_extractor_handles_nesting() -> None:
    """The extractor itself must survive nested dicts — else the guard is vacuous."""
    msg = "Use either: radio={'type': 'condon92'} or radio={'sf': {'type': 'bell2003'}}."
    found = _dict_snippets(msg)
    assert ("radio", {"type": "condon92"}) in found
    assert ("radio", {"sf": {"type": "bell2003"}}) in found, (
        "nested suggestion was truncated — a non-greedy match would do this"
    )
