# SPDX-License-Identifier: BSD-3-Clause
"""``tools/check_gradient_assertions.py`` must catch the bugs that motivated it.

The guard enforces one rule: a predicate over a tri-state (good / bad /
undecided) must name the good state, never merely exclude the bad one. For a
gradient that means **finite AND non-zero, asserted together** — #2100 shipped
an identically zero float32 gradient past a finite-only check (zero is finite),
and #2178 shipped a float32 NaN past a non-zero-only check (``nan != 0.0`` is
``True``).

A guard whose evidence is its own unit tests is worth very little. The mutation
corpus under ``tests/fixtures/assertion_holes/`` therefore carries one function
per hole shape, and ``historical.py`` there transcribes the two shipped
assertions **verbatim** as they stood before their fixes. If a future edit stops
the guard firing on those two, the guard has regressed regardless of what
anything else here says.
"""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "assertion_holes"

sys.path.insert(0, str(ROOT / "tools"))

import check_gradient_assertions as guard


def _scan(name: str, *, precision: bool = False):
    path = FIXTURES / name
    return guard.scan_file(path, name, precision)


def _flagged(name: str, *, precision: bool = False) -> dict[str, str]:
    violations, _ = _scan(name, precision=precision)
    return {v.func: v.missing for v in violations}


# ── the mutation corpus ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("func", "missing"),
    [
        ("hole_2100_finite_only", "nonzero"),
        ("hole_2100_through_a_pytree", "nonzero"),
        ("hole_named_but_not_traced", "nonzero"),
        ("hole_behind_a_negated_isnan", "nonzero"),
        ("hole_2178_nonzero_only", "finite"),
        ("hole_2178_on_a_scalar", "finite"),
    ],
)
def test_every_hole_shape_is_detected(func, missing):
    """One mutation per shape, and the guard must name which half is absent."""
    flagged = _flagged("holes.py")
    assert func in flagged, (
        f"{func} is a reproduction of a shipped hole and the guard did not see it — "
        f"it reported {sorted(flagged)}"
    )
    assert flagged[func] == missing, (
        f"{func}: guard says the missing half is {flagged[func]!r}, not {missing!r}"
    )


def test_an_array_under_test_is_in_scope_only_under_precision_rules():
    """Outside ``tests/regression/precision/`` a forward value is not a gradient.

    Scope is a claim in its own right. Widening it silently would make the guard
    fire on unrelated code and get switched off; narrowing it silently would let
    the #2178 line-flux case (a *forward* array collapsing to zero) back in.
    """
    assert "precision_hole_on_a_forward_value" not in _flagged("holes.py")
    assert "precision_hole_on_a_forward_value" in _flagged("holes.py", precision=True)


def test_the_accepted_forms_are_silent():
    """A guard that flags everything is as useless as one that flags nothing."""
    violations, problems = _scan("clean.py")
    assert not violations, [str(v) for v in violations]
    assert not problems, problems


# ── the escape hatch ───────────────────────────────────────────────────────


def test_a_marker_carrying_no_reason_is_itself_a_failure():
    """An escape hatch that costs no explanation is a mute button, not a hatch."""
    violations, problems = _scan("holes.py")
    text = "\n".join(problems)
    assert "must name the half being skipped" in text, problems
    assert "carries no reason" in text, problems
    # And neither malformed marker suppressed anything.
    flagged = {v.func for v in violations}
    assert "hole_marker_without_a_reason" in flagged
    assert "hole_marker_without_a_half" in flagged


def test_a_marker_with_a_reason_suppresses_only_its_own_assertion():
    flagged = _flagged("clean.py")
    assert "suppressed_by_a_marker_above_the_assert" not in flagged
    assert "suppressed_by_a_marker_on_the_assert" not in flagged


# ── history, which is the only evidence that counts ────────────────────────


@pytest.mark.parametrize(("func", "missing"), [("pre_2100", "nonzero"), ("pre_2178", "finite")])
def test_the_guard_fires_on_the_two_shipped_bugs(func, missing):
    """The assertions as they stood while each defect shipped. Both must fire."""
    flagged = _flagged("historical.py", precision=True)
    assert func in flagged, (
        f"{func} is the verbatim pre-fix assertion for the bug this guard exists for, "
        f"and the guard did not fire on it — reported {sorted(flagged)}"
    )
    assert flagged[func] == missing


# ── the repository itself ──────────────────────────────────────────────────


def test_the_tree_is_clean():
    """The audit that came with the guard has to stay done."""
    violations, problems = guard.collect([ROOT / "tests"])
    assert not problems, problems
    assert not violations, [str(v) for v in violations]


def test_a_holed_file_exits_nonzero():
    """CI mode is a verdict, not a report: the exit code has to move."""
    assert guard.main([str(FIXTURES / "holes.py")]) == 1
    assert guard.main([str(FIXTURES / "clean.py")]) == 0
