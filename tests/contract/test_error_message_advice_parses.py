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

from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS, parse_groups

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _grammar_groups() -> set[str]:
    """The top-level group names :func:`parse_groups` accepts.

    Derived from ``_GROUP_STRUCTURAL_KEYS`` — the map the production path
    validates user keys against, so a group cannot be *accepted by the
    grammar* without an entry and cannot escape this set either. Sub-group
    keys (``dust.emission``) are not top-level names, hence the dot filter.
    """
    return {key for key in _GROUP_STRUCTURAL_KEYS if "." not in key}


#: Advice snippets whose ``name=`` is not a build group at all, so
#: :func:`parse_groups` is simply the wrong grammar to judge them by. The scan
#: below matches any ``name={...}`` literal, which assumes every such snippet is
#: build-grammar advice — true until #1722 added a ``fit_interim`` message
#: advising ``map_options={'n_steps': 40000}``. That advice is *correct*; it
#: names a keyword argument, not a group, and feeding it to the build grammar
#: produced "Unknown group key 'map_options'" and a red main.
#:
#: Each entry must carry a reason, and the test below fails if a name stops
#: appearing in the source or ever becomes a real group — so an exemption
#: cannot quietly outlive the thing it excuses.
_NOT_GRAMMAR_ADVICE: dict[str, str] = {
    "map_options": (
        "a fit_interim()/fit() keyword argument carrying MAP backend options "
        "(n_steps, learning_rate, n_restarts), not a build group — see #1720/#1722"
    ),
}


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

#: Verbs that make the snippet after them something to REMOVE, not to follow.
#: Not every dict literal in a message is a recommendation: the CLOUDY wNE
#: error says "keep this SSP and drop the ``neb={'type': 'cloudy'}`` group",
#: quoting the construct precisely so the reader can delete it. Feeding that to
#: the grammar asks the wrong question — and answers it by accident, since it
#: parses whenever a CLOUDY grid happens to be installed.
_REMOVAL_CUES = ("drop", "remove", "delete", "without", "instead of", "rather than")

#: How far back to look for one of those cues. Long enough to catch "drop the
#: `neb=...` group", short enough not to swallow an unrelated earlier clause.
_CUE_WINDOW = 48


def _is_removal_advice(text: str, at: int) -> bool:
    """Does the message tell the reader to take this snippet OUT?"""
    return any(cue in text[max(0, at - _CUE_WINDOW) : at].lower() for cue in _REMOVAL_CUES)


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
                # Conservative: skip only when EVERY occurrence of this snippet
                # in the message is something the reader is told to remove.
                sites = [m.start() for m in re.finditer(re.escape(name) + r"=\{", text)]
                if sites and all(_is_removal_advice(text, i) for i in sites):
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


def test_removal_advice_is_not_treated_as_a_recommendation() -> None:
    """ "Drop ``x={...}``" quotes a construct to delete, not one to adopt.

    Assumed otherwise at first, and CI caught it: the CLOUDY wNE error says
    "keep this SSP and drop the ``neb={'type': 'cloudy'}`` group", and the arm
    below dutifully fed that to the grammar. It passed locally only because a
    CLOUDY grid happened to be installed, and failed on a runner without one —
    an accidental answer to the wrong question.
    """
    assert _is_removal_advice(
        "keep this SSP and drop the neb={", len("keep this SSP and drop the ")
    )
    assert _is_removal_advice("remove the dust={", len("remove the "))
    assert not _is_removal_advice("Use either: radio={", len("Use either: "))
    assert not _is_removal_advice("Pass one via neb={", len("Pass one via "))


def test_the_exemption_does_not_swallow_the_bug_it_sits_next_to() -> None:
    """The #1677 advice must still be judged, cues or not.

    An exemption that quietly widened until it covered everything would leave
    the arm green and blind — the failure mode this whole file exists to catch.
    """
    message = (
        "met= needs a model built with stellar={'met_mode': 'table'}; either "
        "rebuild with a tabulated metallicity or drop met=."
    )
    names = [name for name, _ in _dict_snippets(message)]
    assert "stellar" in names
    at = message.index("stellar={")
    assert not _is_removal_advice(message, at), (
        "the trailing 'drop met=.' must not exempt the recommendation that precedes it"
    )


def test_every_literal_advice_snippet_in_src_is_accepted_by_the_grammar() -> None:
    """Advice written as a literal in a raise must parse — at every site.

    This is the arm that does not depend on someone remembering to add a
    trigger. `TRIGGERS` reaches four sites; this reaches every one whose
    suggestion is spelled out in the source.
    """
    failures = []
    judged = 0
    for relpath, lineno, name, value in _literal_advice_in_source():
        if name in _NOT_GRAMMAR_ADVICE:
            continue
        judged += 1
        try:
            parse_groups(**{name: value})
        except Exception as exc:
            failures.append(
                f"  {relpath}:{lineno} advises {name}={value!r}\n"
                f"      but that raises {type(exc).__name__}: {exc}"
            )
    assert judged >= 5, (
        f"only {judged} snippets were judged — the exemption list has grown "
        "until this arm checks almost nothing."
    )
    assert not failures, "error-message advice the grammar refuses:\n" + "\n".join(failures)


class TestTheNonGrammarExemptions:
    """An exemption must rot loudly rather than outlive what it excuses."""

    def test_every_exemption_carries_a_reason(self):
        empty = sorted(name for name, why in _NOT_GRAMMAR_ADVICE.items() if not why.strip())
        assert not empty, f"these exemptions state no reason: {empty}"

    def test_no_exemption_names_a_real_group(self):
        """The whole justification is "this is not a build group".

        If one ever becomes a group, the exemption stops being a scoping fix
        and starts hiding exactly the failure this file exists to catch.
        """
        promoted = sorted(set(_NOT_GRAMMAR_ADVICE) & _grammar_groups())
        assert not promoted, (
            f"{promoted} are now real build groups — delete the exemption and "
            "let the grammar judge that advice."
        )

    def test_every_exemption_is_still_advised_somewhere(self):
        """A stale exemption is a silent widening of the sweep."""
        advised = {name for _, _, name, _ in _literal_advice_in_source()}
        unused = sorted(set(_NOT_GRAMMAR_ADVICE) - advised)
        assert not unused, (
            f"{unused} no longer appear as advice in src/ — delete the entry "
            "rather than leaving a standing exemption for nothing."
        )

    def test_the_group_set_is_derived_and_not_empty(self):
        """Anti-vacuity: an empty group set would pass the promotion test."""
        groups = _grammar_groups()
        assert len(groups) >= 8, f"only {len(groups)} grammar groups found: {sorted(groups)}"
        assert "dust" in groups and "sfh" in groups
