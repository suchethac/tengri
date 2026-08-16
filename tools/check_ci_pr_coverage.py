#!/usr/bin/env python3
"""CI guard: every pull request must reach this workflow, and every job must
have made a deliberate decision about which pull requests it runs for.

The test workflow used to declare ``pull_request: branches: [main]``. A PR
stacked on any other branch therefore ran none of it. That is not
hypothetical: #1806 (the nebular band integration) and #1810 (its correction,
which added ``tests/contract/test_precomp_channel_drift.py`` and the bound
pinning the known ``single_component`` defect) both merged into
``worktree-fix-1738-precompute-state`` having run exactly ONE check --
``build``, from ``docs-preview.yml``, the only workflow with no branch filter.
A comparable PR against ``main`` ran twenty. The tests written to pin an
acknowledged defect were themselves never executed, and the stack's first real
verdict arrived only when it retargeted to ``main``: bundled, and long after
the commit anyone would have had to bisect to.

This guard closes the loop on both halves of that.

**The trigger.** ``pull_request`` must carry no ``branches:`` filter, so the
workflow fires for a PR against any base. Re-adding one fails the build.

**The jobs.** Firing is not the same as running everything: the expensive jobs
test ``github.base_ref`` and stand down for a PR that is not aimed at ``main``.
That is a deliberate trade, so every job must appear in exactly one bucket below
with a written reason. A job in none of them fails the guard, which is the
point -- the next person to add a job decides its tier on purpose instead of
inheriting whatever the job above it happened to say. The ``tier`` job only
narrates the decision; nothing depends on it.

Dependencies: standard library only. The ``lint`` job installs ruff and nothing
else, so this must not import ``yaml`` or ``tengri``.

Usage
-----
    python tools/check_ci_pr_coverage.py

Exit code 0 when the trigger is unfiltered and every job is classified; 1
otherwise, naming what is wrong.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# Jobs that run for EVERY pull request, whatever its base. Cheap enough that
# withholding them would save little and cost the early signal that is the
# entire reason a stacked PR runs anything at all.
ALL_PR_JOBS: dict[str, str] = {
    "tier": "narrates which tier this run got; blocks nothing, so it must run "
    "everywhere in order to be able to say when coverage was withheld",
    "lint": "ruff + the stdlib guard scripts; ~1 min, catches most breakage",
    "security": "bandit + pip-audit; cheap and base-independent",
    "smoke": "import/collection guards; `test` declares `needs: smoke`, so it "
    "cannot be withheld without also disabling the full tier",
}

# Jobs whose own `if:` tests `github.base_ref`, so they run for push / schedule
# / dispatch and for a PR based on `main`, and stand down otherwise. These are
# the expensive ones; a stack earns them when it retargets to `main`, which is
# when its code is actually proposed for `main`. Membership here is enforced --
# see `guards_on_base_ref`.
FULL_TIER_JOBS: dict[str, str] = {
    "test": "eight shards, ~19 min wall; the dominant cost of a run",
    "gallery-changes": "gates `gallery`; withholding it cascades correctly",
}

# Jobs whose own `if:` already excludes ordinary pull requests, so the tier is
# irrelevant to them. Listed rather than ignored: an unlisted job is an
# unmade decision, and that is what this guard exists to catch.
ALREADY_GATED_JOBS: dict[str, str] = {
    "gallery": "runs only when `gallery-changes` says it is needed",
    "coverage": "`schedule` / `workflow_dispatch` only; never fires on a PR",
    "slow": "opt-in via the `run-slow-tests` label",
    "crossval": "opt-in via the `run-crossval` label",
}

_JOB_NAME = re.compile(r"^  ([A-Za-z][A-Za-z0-9_-]*):\s*$", re.MULTILINE)
_PULL_REQUEST_BLOCK = re.compile(
    r"^  pull_request:\s*$(.*?)(?=^  \w|\Z)", re.MULTILINE | re.DOTALL
)


def strip_comments(text: str) -> str:
    """Drop YAML comments so prose cannot be mistaken for a declaration.

    This is load-bearing, not tidiness -- the same lesson
    ``tools/check_test_paths_covered.py`` records, for the same reason. The
    comment above the trigger in ``tests.yml`` explains the removed filter and
    therefore contains the literal text ``branches: [main]``. A guard that read
    the raw file would find that comment and report the filter as present, or
    (worse, once the wording drifts) as absent when it is not. A guard that
    reads its own documentation as evidence fails open, which is the one thing
    a guard may never do.
    """
    return "\n".join(re.sub(r"#.*$", "", line) for line in text.splitlines())


def job_names(text: str) -> list[str]:
    """Top-level job keys, in file order."""
    body = text.split("\njobs:\n", 1)
    if len(body) != 2:
        return []
    return _JOB_NAME.findall(body[1])


def pull_request_filters_by_branch(text: str) -> bool:
    """True when the ``pull_request`` trigger restricts by base branch."""
    match = _PULL_REQUEST_BLOCK.search(text)
    if match is None:
        return False
    return re.search(r"^\s+branches(-ignore)?:", match.group(1), re.MULTILINE) is not None


def job_block(text: str, name: str) -> str:
    """The YAML body of one job, up to the next job key."""
    pattern = re.compile(
        rf"^  {re.escape(name)}:\s*$(.*?)(?=^  [A-Za-z]|\Z)", re.MULTILINE | re.DOTALL
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def guards_on_base_ref(block: str) -> bool:
    """True when a job's own ``if:`` tests the pull request's base branch.

    The tier condition is inline rather than carried on a ``needs:`` edge --
    see the comment above the ``tier`` job for why making the test matrix wait
    on a runner for a boolean was the wrong trade. Inline means the expression
    is repeated, and a repeated expression drifts, which is the failure this
    whole guard exists to prevent. So it is asserted rather than trusted.
    """
    return "github.base_ref" in block


def main() -> int:
    if not WORKFLOW.is_file():
        print(f"ERROR: cannot read {WORKFLOW.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    text = strip_comments(WORKFLOW.read_text(encoding="utf-8"))
    problems: list[str] = []

    if pull_request_filters_by_branch(text):
        problems.append(
            "`pull_request` declares a `branches:` filter, so PRs against any\n"
            "other base run none of this workflow. That is how #1806 and #1810\n"
            "merged with one check apiece. Remove the filter; use the `tier` job\n"
            "to decide how much of the suite a given base earns."
        )

    classified = {**ALL_PR_JOBS, **FULL_TIER_JOBS, **ALREADY_GATED_JOBS}
    found = job_names(text)

    if not found:
        problems.append("could not parse any job names out of the workflow")

    for name in found:
        if name not in classified:
            problems.append(
                f"job `{name}` is in no tier bucket in {Path(__file__).name}.\n"
                "Add it to ALL_PR_JOBS, FULL_TIER_JOBS, or ALREADY_GATED_JOBS\n"
                "with a reason. An unlisted job is an unmade decision about\n"
                "which pull requests it runs for."
            )

    for name in FULL_TIER_JOBS:
        if name in found and not guards_on_base_ref(job_block(text, name)):
            problems.append(
                f"job `{name}` is listed as full-tier-only but its `if:` does not\n"
                "test `github.base_ref`, so it will run for a PR against any base.\n"
                "The tier condition is inline by design (a `needs:` edge would put\n"
                "a runner acquisition on the test matrix's critical path); inline\n"
                "means repeated, and repeated means it can drift, which is what\n"
                "this checks."
            )

    for name in sorted(set(classified) - set(found)):
        problems.append(
            f"`{name}` is classified in {Path(__file__).name} but no longer\n"
            "exists in the workflow. Drop the stale entry."
        )

    if problems:
        print("CI pull-request coverage is not intact:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}\n", file=sys.stderr)
        return 1

    print(
        f"OK: `pull_request` fires for every base, and all {len(found)} jobs declare their tier."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
