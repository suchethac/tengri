# SPDX-License-Identifier: BSD-3-Clause
"""Recovery advice inside a grammar error must itself be accepted.

An error message is the one piece of documentation a user is *guaranteed* to
read, because they only see it when already stuck. When its suggested fix also
raises, the user is bounced from one error to another with nothing to try next.

The legacy ``radio={'type': ...}`` form is now retired (PR6). Error messages
for the legacy form suggest the composable ``radio={'sf': {...}, 'agn': {...}}``
form, which must be accepted by the grammar.

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


def _literal_eval_with_sentinels(expr: str) -> dict:
    """Parse a dict literal that may contain FIXED and FREE sentinels.

    Parses an expression using ast, resolving bare names FIXED and FREE to
    the real tengri sentinels. All other bare names are rejected (strict
    literal eval). Everything else (numbers, strings, nested dicts) must be
    valid Python literals.
    """

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as err:
        msg = f"Failed to parse as expression: {expr}"
        raise ValueError(msg) from err

    # Walk the AST and resolve names
    def resolve_names(node):
        if isinstance(node, ast.Dict):
            keys = [resolve_names(k) for k in node.keys]
            values = [resolve_names(v) for v in node.values]
            return dict(zip(keys, values))
        elif isinstance(node, ast.Name):
            if node.id == "FIXED":
                from tengri import FIXED

                return FIXED
            elif node.id == "FREE":
                from tengri import FREE

                return FREE
            else:
                msg = f"Unknown bare name: {node.id!r}. Only FIXED and FREE are allowed."
                raise ValueError(msg)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [resolve_names(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(resolve_names(elt) for elt in node.elts)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            val = resolve_names(node.operand)
            return -val
        else:
            msg = f"Unsupported AST node type: {type(node).__name__}"
            raise ValueError(msg)

    return resolve_names(tree.body)


def _dict_snippets(message: str) -> list[tuple[str, dict]]:
    """Every ``name={...}`` python-dict literal in ``message``.

    Brace-matched rather than regex-matched: the suggestions nest
    (``radio={'sf': {'type': 'bell2003'}}``) and a non-greedy regex stops at the
    first inner ``}``, silently checking a truncated snippet that happens to
    parse — a guard that passes for the wrong reason.

    Handles sentinels FIXED and FREE via _literal_eval_with_sentinels.
    Filters out snippets that are shown for removal/replacement (e.g., "Replace
    dust={...} with dust_attenuation={...}" only suggests the dust_attenuation part).
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

        # Skip removal advice (e.g., "replace dust={...}" or "drop neb={...}")
        if _is_removal_advice(message, j):
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
            value = _literal_eval_with_sentinels(message[i : end + 1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            out.append((name, value))
    return out


# Each entry provokes a grammar error that offers recovery advice.
from tengri import FIXED

TRIGGERS = [
    pytest.param({"radio": True}, id="radio-bool-gate-form"),
    pytest.param({"xray": True}, id="xray-bool-gate-form"),
    pytest.param({"shock": True}, id="shock-bool-gate-form"),
    # Dust retirement: old unified dust= form, lawless (README shape from #2015)
    pytest.param(
        {"dust": {"type": "two_component", "all_params": FIXED}},
        id="dust-two-component-lawless",
    ),
    # Dust retirement: with explicit law
    pytest.param(
        {"dust": {"type": "two_component", "law": "calzetti", "tau_bc": 0.5, "tau_diff": 0.3}},
        id="dust-two-component-with-law",
    ),
    # Dust retirement: nested emission form
    pytest.param(
        {
            "dust_attenuation": {
                "type": "two_component",
                "law": "calzetti",
                "emission": {"type": "dale2014"},
            }
        },
        id="dust-nested-emission",
    ),
    # Dust retirement: two_component lone law_bc (pre-#1989 shape, benchmark had this)
    pytest.param(
        {
            "dust": {
                "type": "two_component",
                "law_bc": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 0.3,
            }
        },
        id="dust-two-component-lone-law-bc",
    ),
    # Dust retirement: two_component lone law_diff
    pytest.param(
        {
            "dust": {
                "type": "two_component",
                "law_diff": "power_law",
                "tau_bc": 0.5,
                "tau_diff": 0.3,
            }
        },
        id="dust-two-component-lone-law-diff",
    ),
    # Dust retirement: single_component with law_bc (pre-#1989, accepted per migration)
    pytest.param(
        {"dust": {"type": "single_component", "law_bc": "calzetti", "tau_v": 0.5}},
        id="dust-single-component-with-law-bc",
    ),
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
        # Skip snippets with ellipsis — they're templates for the reader to complete
        if _has_ellipsis(value):
            continue

        try:
            result = parse_groups(**{name: value})
        except Exception as exc:
            pytest.fail(
                f"the error message for {kwargs!r} recommends {name}={value!r}, "
                f"but that raises {type(exc).__name__}: {exc}. Recovery advice "
                f"must be accepted by the grammar it describes."
            )

        # For dust_attenuation blocks, verify that the law actually landed
        # in the result. A suggestion that parses but silently drops the law
        # is the same bug one layer deeper (#2030).
        if name == "dust_attenuation" and isinstance(value, dict):
            dust_type = value.get("type")
            if dust_type == "two_component":
                # Either dust_law_bc (from 'law') or both dust_law_bc and
                # dust_law_diff must be present in result
                has_law_bc = hasattr(result, "dust_law_bc") and result.dust_law_bc is not None
                has_law_diff = (
                    hasattr(result, "dust_law_diff") and result.dust_law_diff is not None
                )
                assert has_law_bc and has_law_diff, (
                    f"dust_attenuation type='two_component' suggestion {name}={value!r} "
                    f"parsed but lost the attenuation law (no dust_law_bc/"
                    f"dust_law_diff in result). The suggestion is incomplete."
                )

                # For lone-law_bc merges, verify the law value was preserved
                # (not defaulted to power_law). Lone law_bc applies to both screens.
                if "law" in value and "law_bc" not in value and "law_diff" not in value:
                    # This is the 'law' form; check it matches what the user gave
                    expected_law = value.get("law")
                    if expected_law:
                        actual_law_bc = (
                            result.dust_law_bc if hasattr(result, "dust_law_bc") else None
                        )
                        assert actual_law_bc == expected_law, (
                            f"dust_attenuation suggestion merged 'law' to dust_law_bc, "
                            f"but the value changed: expected {expected_law!r}, "
                            f"got {actual_law_bc!r}. Merge was not behavior-preserving."
                        )

            elif dust_type == "single_component":
                # Must have dust_law_bc (single screen, stored on both _bc/_diff)
                has_law = (
                    hasattr(result, "dust_law_bc")
                    and result.dust_law_bc is not None
                    and hasattr(result, "dust_law_diff")
                    and result.dust_law_diff is not None
                )
                assert has_law, (
                    f"dust_attenuation type='single_component' suggestion "
                    f"{name}={value!r} parsed but lost the attenuation law "
                    f"(no dust_law_bc in result). The suggestion is incomplete."
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
#: parses whenever a CLOUDY grid happens to be installed. Retirement messages
#: like "Replace dust={...} with dust_attenuation={...}" quote the old form
#: for removal context ("Replace ... with ...").
_REMOVAL_CUES = ("drop", "remove", "delete", "without", "instead of", "rather than", "replace")

#: How far back to look for one of those cues. Long enough to catch "drop the
#: `neb=...` group", short enough not to swallow an unrelated earlier clause.
_CUE_WINDOW = 48


def _is_removal_advice(text: str, at: int) -> bool:
    """Does the message tell the reader to take this snippet OUT?"""
    return any(cue in text[max(0, at - _CUE_WINDOW) : at].lower() for cue in _REMOVAL_CUES)


#: Words that mark a message as talking about *building a model*. Only those
#: snippets are build-grammar advice; ``kwarg={...}`` in a message about some
#: other API is that API's own argument shape.
#: Stems, not whole words: the #1677 message says "needs a model **built**
#: with ...", and a ``"build"`` cue does not match ``"built"``. The neuter
#: check below is what caught that — the narrowed guard silently stopped
#: recognizing the very message it was written for.
_BUILD_CUES = ("buil", "sedmodel", "parse_groups", "recipe", "group")


def _is_build_advice(text: str) -> bool:
    """Is this message telling the reader how to build a model?

    ``_NOT_GRAMMAR_KEYS`` used to answer this by listing key names that are not
    groups — a hand-written list, so it went stale the moment a new API grew a
    dict-shaped kwarg. #1722 added ``map_options must be a dict of MAP backend
    options ... Example: map_options={'n_steps': 40000}`` to ``fit_interim``,
    and this guard reported it as build advice the grammar refuses, turning
    main red. ``map_options`` is a ``fit_interim`` argument; the message never
    mentions building anything.

    Keying on the message's own subject is derived rather than listed. The
    alternative — accept only snippets whose key is already a valid group —
    cannot work: it would skip exactly the case this guard exists for, advice
    naming a group that does *not* exist (``met={'type': 'table'}``, #1677).
    """
    return any(cue in text.lower() for cue in _BUILD_CUES)


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
            if not _is_build_advice(text):
                continue
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
        "met= needs a model built with met={'type': 'table'}; either "
        "rebuild with a tabulated metallicity or drop met=."
    )
    names = [name for name, _ in _dict_snippets(message)]
    assert "met" in names
    at = message.index("met={")
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
    for relpath, lineno, name, value in _literal_advice_in_source():
        if _has_ellipsis(value):
            # ``met={'type': ...}`` names the group and the key but leaves the
            # value for the reader — prose, not a runnable line, the same
            # category as an f-string interpolation. Executing it would fail on
            # the placeholder rather than on anything the writer got wrong.
            #
            # NOT skipped, though: the whole point of this arm is that #1677 was
            # a wrong *group name*, and a placeholder value hides nothing about
            # the name. It is still checked, so the arm would still have caught
            # `met={'type': ...}` back when `met` did not exist — the neuter
            # check that matters here.
            failures.extend(_bad_group_name(relpath, lineno, name, value))
            continue
        try:
            parse_groups(**{name: value})
        except Exception as exc:
            failures.append(
                f"  {relpath}:{lineno} advises {name}={value!r}\n"
                f"      but that raises {type(exc).__name__}: {exc}"
            )
    assert not failures, "error-message advice the grammar refuses:\n" + "\n".join(failures)


def _has_ellipsis(value) -> bool:
    """Does this snippet leave a value for the reader to fill in?"""
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(_has_ellipsis(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_ellipsis(v) for v in value)
    return False


def _bad_group_name(relpath, lineno, name, value) -> list[str]:
    """Check the group name — the part a placeholder value leaves intact.

    Only the name. Keys are deliberately *not* checked: ``met={'logzsol': ...}``
    is legitimate advice naming a per-parameter override, and those are not
    structural keys, so judging keys here would flag correct advice. The name is
    what #1677 got wrong, and the name is what this still catches.
    """
    from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

    valid_groups = {k for k in _GROUP_STRUCTURAL_KEYS if "." not in k}
    if name in valid_groups:
        return []
    return [
        f"  {relpath}:{lineno} advises {name}={value!r}\n"
        f"      but '{name}' is not a grammar group. Valid: {sorted(valid_groups)}"
    ]


def test_the_scope_rule_still_catches_the_bug_it_was_written_for() -> None:
    """Narrowing the guard must not narrow what it catches.

    ``_NOT_GRAMMAR_KEYS`` answered "is this build advice?" with a hand-written
    list of key names, so #1722's ``map_options={'n_steps': 40000}`` — a
    ``fit_interim`` argument, not a group — was reported as build advice the
    grammar refuses, and main went red. The rule now keys on the message's own
    subject.

    The first attempt at that rule used the cue ``"build"``, which does not
    match ``"built"`` — and the #1677 message reads "needs a model **built**
    with ...". The narrowed guard silently stopped recognizing the message it
    exists for. Both directions are pinned here.
    """
    built = "raise ValueError(\"met= needs a model built with met={'type': 'table'}\")"
    other_api = (
        'raise TypeError("map_options must be a dict of MAP backend options. '
        "Example: map_options={'n_steps': 40000}.\")"
    )
    assert _is_build_advice(built), (
        "advice about building a model is no longer recognized as build advice; "
        "the #1677 bug would pass unchecked."
    )
    assert not _is_build_advice(other_api), (
        "another API's dict-shaped kwarg is being validated against the build "
        "grammar; that is what turned main red after #1722."
    )
