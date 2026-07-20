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
"""

from __future__ import annotations

import ast

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


def test_snippet_extractor_handles_nesting() -> None:
    """The extractor itself must survive nested dicts — else the guard is vacuous."""
    msg = "Use either: radio={'type': 'condon92'} or radio={'sf': {'type': 'bell2003'}}."
    found = _dict_snippets(msg)
    assert ("radio", {"type": "condon92"}) in found
    assert ("radio", {"sf": {"type": "bell2003"}}) in found, (
        "nested suggestion was truncated — a non-greedy match would do this"
    )
