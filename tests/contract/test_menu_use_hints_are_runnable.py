# SPDX-License-Identifier: BSD-3-Clause
"""The ``use:`` column is a copy-paste instruction, so it has to run.

Every ``list_*`` row carries a ``use`` hint, and it is the line a fresh user
copies out of a menu or out of ``describe()``. 162 of them are
``SEDModel.build(..., group={...})`` snippets. One did not parse::

    list_nebular_backends  cloudy
      use: SEDModel.build(..., neb={'type': 'cloudy', 'gridfile': 'grid.h5'})

      ValueError: Unknown key 'gridfile' in group 'neb'. Did you mean: grid?

The key is ``grid``. What makes this one worth a guard rather than a one-line
correction: the source comment above it says the ``gridfile`` key was added
*"so the hint builds"* — a bare ``neb={'type': 'cloudy'}`` raises "needs a grid
file". So a line written **because** the hint failed failed differently, and
nothing checked either version.

#1678 already guards literal advice appearing in ``src/``. This is a different
population: the ``use`` column is assembled by ``list_*`` at call time, so a
source scan never sees it. Same failure class (#1446 "advice that raises"), a
census no guard covered.

Checked by parsing, not by pattern-matching the string: a snippet that merely
*looks* like the others is not evidence it is accepted.
"""

from __future__ import annotations

import ast
import re
import warnings

import pytest

import tengri
from tengri import Fixed
from tengri.parameters.groups import parse_groups

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _kwargs_in(call: str):
    """Yield ``(group, literal_text)`` for each ``group={...}`` in a build call.

    Brace-matched rather than regex-terminated: the AGN and dust hints nest a
    sub-block (``agn={'atten': {'type': ...}}``), which a non-greedy regex
    truncates at the inner brace.
    """
    for match in re.finditer(r"\b([a-z_]+)\s*=\s*\{", call):
        start = match.end() - 1
        depth, i = 0, start
        while i < len(call):
            if call[i] == "{":
                depth += 1
            elif call[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield match.group(1), call[start : i + 1]


def _build_hints() -> list[tuple[str, str, str]]:
    """(menu, name, hint) for every row whose ``use`` is a ``SEDModel.build``."""
    out = []
    for menu in sorted(n for n in dir(tengri) if n.startswith("list_")):
        fn = getattr(tengri, menu, None)
        if not callable(fn):
            continue
        try:
            rows = fn()
        except Exception:
            continue
        if not (isinstance(rows, list) and rows and isinstance(rows[0], dict)):
            continue
        for row in rows:
            hint = str(row.get("use", "") or "").strip()
            if hint.startswith("SEDModel.build"):
                out.append((menu, str(row.get("name")), hint))
    return out


_HINTS = _build_hints()


class TestTheCensus:
    def test_there_are_hints_to_check(self):
        assert len(_HINTS) > 100, (
            f"only {len(_HINTS)} build hints found — the scan broke, or the "
            f"use: column stopped carrying them."
        )

    def test_every_hint_yields_at_least_one_group(self):
        """A hint with no extractable group would pass the parse test vacuously."""
        empty = [(m, n) for m, n, h in _HINTS if not list(_kwargs_in(h))]
        assert not empty, f"these hints contain no group={{...}} to check: {empty[:6]}"

    def test_the_brace_matcher_handles_a_nested_sub_block(self):
        """AGN and dust hints nest one dict inside another."""
        hint = "SEDModel.build(..., agn={'atten': {'type': 'polar_dust'}})"
        groups = dict(_kwargs_in(hint))
        assert groups == {"agn": "{'atten': {'type': 'polar_dust'}}"}, groups


@pytest.mark.parametrize(("menu", "name", "hint"), _HINTS, ids=[f"{m}:{n}" for m, n, _ in _HINTS])
def test_the_hint_a_user_would_copy_is_accepted(menu, name, hint):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for group, body in _kwargs_in(hint):
            try:
                value = ast.literal_eval(body)
            except (ValueError, SyntaxError) as exc:  # pragma: no cover - shape guard
                pytest.fail(f"[{menu}] {name}: {group}={body} is not a literal ({exc})")
            try:
                parse_groups(**{group: value}, redshift=Fixed(0.1))
            except Exception as exc:
                pytest.fail(
                    f"[{menu}] {name} advertises a hint the grammar rejects.\n"
                    f"  use     : {hint}\n"
                    f"  snippet : {group}={body}\n"
                    f"  error   : {type(exc).__name__}: {exc}"
                )


def test_the_cloudy_hint_names_the_key_the_grammar_takes():
    """The one that shipped wrong, pinned by behavior rather than by string."""
    row = next(r for r in tengri.list_nebular_backends() if r["name"] == "cloudy")
    groups = dict(_kwargs_in(row["use"]))
    assert "neb" in groups, row["use"]
    value = ast.literal_eval(groups["neb"])
    assert "gridfile" not in value, (
        "the hint still advertises 'gridfile'; the grammar's key is 'grid'."
    )
    parse_groups(neb=value, redshift=Fixed(0.1))
