# SPDX-License-Identifier: BSD-3-Clause
"""Test assertions over the engagement report (#3505 X1): they read records, never re-run gates."""

from __future__ import annotations

import re

from tengri.forward.precompute_report import PrecomputeState, engagement_reports


def assert_engaged(target: object, what: str, *, regime: str | None = None) -> None:
    """Assert that mechanism ``what`` engaged for every galaxy the target covers."""
    _check(target, what, regime, reason=None)


def assert_declined(
    target: object, what: str, reason: str | re.Pattern[str], *, regime: str | None = None
) -> None:
    """Assert that ``what`` declined, for every galaxy, with a reason matching ``reason``.

    ``reason`` is required: a substring of the recorded reason, or a compiled pattern
    searched in it.
    """
    if not isinstance(reason, re.Pattern) and (not isinstance(reason, str) or not reason.strip()):
        raise TypeError(f"assert_declined needs a non-blank reason or a pattern, got {reason!r}")
    _check(target, what, regime, reason=reason)


def _check(
    target: object, what: str, regime: str | None, reason: str | re.Pattern[str] | None
) -> None:
    reports = engagement_reports(target, regime=regime)
    failures: dict[int, str] = {}
    for index, report in enumerate(reports):
        try:
            state = report.state_of(what)
        except KeyError as missing:
            raise AssertionError(str(missing.args[0])) from missing
        problem = _problem(state, reason)
        if problem:
            failures[index] = problem
    if not failures:
        return
    if len(reports) == 1:
        raise AssertionError(f"{what!r}: {failures[0]}")
    listing = "\n".join(f"  galaxy {index}: {problem}" for index, problem in failures.items())
    raise AssertionError(
        f"{what!r} failed on galaxies {sorted(failures)} of {len(reports)}:\n{listing}"
    )


def _problem(state: PrecomputeState, reason: str | re.Pattern[str] | None) -> str | None:
    """Return why ``state`` does not meet the expectation, or None when it does."""
    if reason is None:
        return None if state.state == "engaged" else _describe(state)
    if state.state != "declined":
        return f"is {state.state}, not declined"
    actual = state.reason or ""
    matched = (
        reason.search(actual) is not None if isinstance(reason, re.Pattern) else reason in actual
    )
    return None if matched else f"declined as {actual!r}, expected {reason!r}"


def _describe(state: PrecomputeState) -> str:
    if state.state == "declined":
        return f"declined: {state.reason}"
    return state.state
