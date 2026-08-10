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

A fourth arm reads the advice out of the source rather than waiting for someone
to trigger it. Those three arms are only as wide as what reaches them, and the
raised-error arm is a hand-maintained list of four ``TRIGGERS`` against eighteen
``raise`` sites that emit a ``group={...}`` snippet. That is how
``Catalog.from_histories`` shipped advising ``met={'type': 'table'}`` — a group
key the grammar does not have (#1677) — at a site no trigger reached.
"""

from __future__ import annotations

import ast
import re
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


# ── Fourth arm: the census, not a trigger list ──────────────────────
#
# The three arms above are only as wide as what reaches them, and TRIGGERS is a
# hand-maintained list of four. Eighteen `raise` sites across ten modules emit a
# `group={...}` snippet, so fourteen of them were never checked — which is how
# `Catalog.from_sfh` shipped advising `met={'type': 'table'}`, a group key the
# grammar does not have (#1677).
#
# Reaching every site by triggering it means reproducing eighteen distinct
# failure preconditions. Reading the advice straight out of the source does not,
# and a literal in a `raise` is exactly what the user will be shown.

_SRC = Path(__file__).resolve().parents[2] / "src" / "tengri"

#: Names that are NOT build-grammar keys, so a dict literal against them is not
#: build advice. This is deliberately a denylist of known non-grammar kwargs and
#: **not** an allowlist of valid groups: filtering down to names the grammar
#: already accepts would make the check structurally incapable of catching
#: advice that names a group which does not exist — which is exactly how #1677
#: (`met={'type': 'table'}`) reads. An allowlist here passes the neuter check
#: for the wrong reason; this was caught by running one.
_NOT_GRAMMAR_KEYS = frozenset({"priors", "params", "data", "kwargs", "metadata"})

#: An f-string interpolation survives `ast.unparse` as a bare `{name}` and
#: `literal_eval`s into that placeholder text. The live message substitutes a
#: real value, so those snippets are not judgeable from source.
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][\w.\[\]'\"()]*\}")


def _literal_advice_in_source() -> list[tuple[str, int, str, dict]]:
    """Every fully-literal ``group={...}`` snippet inside a ``raise`` in src/."""
    out: list[tuple[str, int, str, dict]] = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - src/ always parses
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            # f-strings render literal braces doubled; collapse before matching.
            text = ast.unparse(node).replace("{{", "{").replace("}}", "}")
            for name, value in _dict_snippets(text):
                if name in _NOT_GRAMMAR_KEYS or not name.islower():
                    continue
                if _PLACEHOLDER.search(repr(value)):
                    continue
                out.append((str(path.relative_to(_SRC)), node.lineno, name, value))
    return out


def test_the_source_census_finds_advice_to_check() -> None:
    """A census that silently collapses would make the check below vacuous.

    The failure mode this pins is a change to ``ast.unparse`` output or to the
    brace-collapsing above quietly matching nothing, which reads identically to
    "all advice is valid".
    """
    found = _literal_advice_in_source()
    assert len(found) >= 5, (
        f"expected the source scan to find several literal advice snippets, got "
        f"{found!r}. An empty census passes the next test for the wrong reason."
    )


def test_every_literal_advice_snippet_in_src_is_accepted_by_the_grammar() -> None:
    """Advice written as a literal in a raise must parse — at every site.

    This is the arm that does not depend on someone remembering to add a
    trigger. `TRIGGERS` reaches four sites; this reaches every one whose
    suggestion is spelled out in the source.
    """
    failures = []
    for relpath, lineno, name, value in _literal_advice_in_source():
        try:
            parse_groups(**{name: value})
        except Exception as exc:
            failures.append(
                f"  {relpath}:{lineno} advises {name}={value!r}\n"
                f"      but that raises {type(exc).__name__}: {exc}"
            )
    assert not failures, "error-message advice the grammar refuses:\n" + "\n".join(failures)
