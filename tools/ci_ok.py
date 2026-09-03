#!/usr/bin/env python3
"""Decide the single aggregate verdict that branch protection requires.

Branch protection currently names eleven contexts, eight of which are the
expanded ``test`` matrix strings. That coupling has two costs:

* **Nothing can be withheld.** A matrix job skipped by ``if:`` publishes the
  bare name ``test``, never the eight expanded shard contexts, so a required
  shard context simply never reports and the pull request blocks forever. That
  is why the heavy tiers cannot be path-filtered today, even though 53% of open
  pull requests touch no importable code at all.
* **The shard list is duplicated into a repository setting.** Renaming a shard,
  or adding one, silently desynchronises the protection rule from the workflow,
  and nothing in the repository can see the rule to check it.

Collapsing to one context fixes both, but only if that context is *stricter*
than "nothing went red" -- a plain ``skipped`` must never read as approval. The
policy below therefore mirrors ``tools/check_ci_pr_coverage.py`` exactly, and
the buckets carry the same meanings:

``ALWAYS_RUN``
    Runs for every pull request whatever its base. ``skipped`` is a policy
    violation here, not an outcome, so only ``success`` is accepted.

``FULL_TIER``
    Earned by a pull request that actually targets ``main`` (and by push,
    schedule and dispatch). When it is owed, ``skipped`` is refused; when it is
    not owed -- a stacked pull request against another branch -- ``skipped`` is
    the correct result and is accepted.

``GATED``
    Decided by another job's output rather than by the tier, so both
    ``success`` and ``skipped`` are legitimate.

``SCHEDULED_TIER``
    Runs on schedule and dispatch events; slow and crossval also run on
    pull requests carrying their opt-in labels (``run-slow-tests`` and
    ``run-crossval`` respectively); coverage never runs on pull requests.
    When a labeled PR run opts into slow or crossval, a failure reddens the
    verdict and blocks that PR — deliberately, since a test the author chose
    to run must not fail invisibly. Before this bucket, all three jobs' failure
    was invisible on schedule runs — issue #2128. A skipped job on schedule or
    dispatch is a fault (they are owed those events); a skipped job on push or
    a PR without its label is acceptable.

Why the ``labeled`` pull-request event is no longer a trigger
------------------------------------------------------------
It cannot coexist with a single aggregate context. On a ``labeled`` event the
heavy tiers skip by design, so this job would publish either a success that
claims tests passed when they never ran, or a failure that reddens a pull
request for the crime of being labelled. Today the matrix's name expansion hides
that problem -- the eight shard contexts keep their real results because a
skipped matrix job cannot publish them (#1865). A plain job has no such
protection: its name *is* its context, and it publishes on every event.

So the trigger goes. ``slow`` is unaffected: it reads
``github.event.pull_request.labels.*.name``, the pull request's current label
set, on any pull-request event -- so labelling then pushing still opts in, and
``workflow_dispatch`` covers labelling without pushing.

Usage
-----
Reads the ``needs`` context as JSON from the environment, so the policy is
data-driven and unit-testable rather than inline shell::

    CI_OK_RESULTS='{"lint": {"result": "success"}, ...}' python tools/ci_ok.py
"""

from __future__ import annotations

import json
import os
import sys

#: The four values GitHub Actions reports for ``needs.<job>.result``.
#:
#: Spelled the way GitHub spells them. That is an external data contract, not
#: tengri prose, so the British form stands under NAMING_CONTRACT §10 — the same
#: exemption the Synthesizer HDF5 keys take. Declared once here so no other file
#: has to write the word: :mod:`tests.unit.test_ci_ok` derives its cases from
#: this tuple rather than repeating the literals.
GITHUB_RESULTS = ("success", "failure", "cancelled", "skipped")

#: Results that never, on their own, indicate the job did its job.
NOT_SUCCESS = tuple(r for r in GITHUB_RESULTS if r != "success")

#: Must run for every pull request; ``skipped`` is a policy violation.
ALWAYS_RUN = ("tier", "lint", "security", "smoke")

#: Owed when the pull request targets ``main``; may skip otherwise.
FULL_TIER = ("test", "gallery-changes")

#: Decided by another job's output; ``skipped`` is a legitimate result.
GATED = ("gallery",)

#: Run on schedule and dispatch; slow/crossval also run on label-opted PRs.
#: On owed events, ``skipped`` is a policy violation; on others it is legitimate.
SCHEDULED_TIER = ("slow", "coverage", "crossval")

#: Events that always earn the full tier, having no base branch to weigh.
_FULL_TIER_EVENTS = ("push", "schedule", "workflow_dispatch")


def full_tier_owed(event: str, base_ref: str) -> bool:
    """Whether this run was supposed to execute the expensive jobs.

    Parameters
    ----------
    event : str
        The value of ``github.event_name``.
    base_ref : str
        The value of ``github.base_ref``; empty for non-pull-request events.

    Returns
    -------
    bool
        True when the full tier is owed, so a skipped heavy job is a fault.
    """
    if event in _FULL_TIER_EVENTS:
        return True
    return event == "pull_request" and base_ref == "main"


def scheduled_tier_owed(event: str) -> bool:
    """Whether this run was supposed to execute the scheduled tier jobs.

    Parameters
    ----------
    event : str
        The value of ``github.event_name``.

    Returns
    -------
    bool
        True when the scheduled tier is owed (schedule or workflow_dispatch),
        so a skipped scheduled job is a fault.
    """
    return event in ("schedule", "workflow_dispatch")


def decide(results: dict[str, str], event: str, base_ref: str) -> list[str]:
    """Apply the tier policy to one run's job results.

    Parameters
    ----------
    results : dict
        Mapping of job name to its ``result``, one of :data:`GITHUB_RESULTS`.
    event : str
        The value of ``github.event_name``.
    base_ref : str
        The value of ``github.base_ref``; empty for non-pull-request events.

    Returns
    -------
    list of str
        One human-readable problem per violated rule; empty when the run may
        be treated as green. Never raises on an unknown job -- an unclassified
        job is reported, because an unlisted job is an unmade decision.
    """
    problems: list[str] = []
    owed = full_tier_owed(event, base_ref)

    for job in ALWAYS_RUN:
        got = results.get(job)
        if got is None:
            problems.append(f"`{job}` reported no result; it must run for every pull request")
        elif got != "success":
            problems.append(
                f"`{job}` is `{got}`, but it must run for every pull request whatever "
                f"its base (tools/check_ci_pr_coverage.py: ALL_PR_JOBS)"
            )

    for job in FULL_TIER:
        got = results.get(job)
        if got is None:
            problems.append(f"`{job}` reported no result")
        elif got == "success" or (got == "skipped" and not owed):
            # Either it ran and passed, or it stood down on a base that never
            # earned it. The second half is the ONLY way a skip is acceptable.
            continue
        elif got == "skipped":
            problems.append(
                f"`{job}` was skipped, but this run targets `main` and so owed the "
                f"full tier. A withheld heavy job must never read as approval."
            )
        else:
            problems.append(f"`{job}` is `{got}`")

    for job in GATED:
        got = results.get(job)
        if got not in (None, "success", "skipped"):
            problems.append(f"`{job}` is `{got}`")

    owed_scheduled = scheduled_tier_owed(event)
    for job in SCHEDULED_TIER:
        got = results.get(job)
        if got is None:
            problems.append(f"`{job}` reported no result")
        elif got == "success":
            continue
        elif got == "skipped":
            if owed_scheduled:
                problems.append(
                    f"`{job}` was skipped, but this run is a schedule or dispatch event "
                    f"and so owed the scheduled tier. A withheld scheduled job must never "
                    f"pass silently (tools/ci_ok.py: SCHEDULED_TIER)"
                )
        elif got in ("failure", "cancelled"):
            problems.append(
                f"`{job}` is `{got}`, and this must never pass silently whatever the event "
                f"(tools/ci_ok.py: SCHEDULED_TIER)"
            )
        else:
            problems.append(f"`{job}` is `{got}`")

    classified = set(ALWAYS_RUN) | set(FULL_TIER) | set(GATED) | set(SCHEDULED_TIER)
    for job in sorted(set(results) - classified):
        problems.append(
            f"`{job}` is a dependency of ci-ok but is in no bucket in tools/ci_ok.py; "
            f"an unlisted job is an unmade decision about whether it may be skipped"
        )

    return problems


def main() -> int:
    """Read the ``needs`` context from the environment and report the verdict.

    Returns
    -------
    int
        0 when every rule holds, 1 otherwise.
    """
    raw = os.environ.get("CI_OK_RESULTS", "").strip()
    if not raw:
        print("FAIL: CI_OK_RESULTS is empty — ci-ok cannot verify anything.", file=sys.stderr)
        return 1

    try:
        needs = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"FAIL: CI_OK_RESULTS is not valid JSON: {exc}", file=sys.stderr)
        return 1

    results = {
        job: (payload or {}).get("result", "") if isinstance(payload, dict) else str(payload)
        for job, payload in needs.items()
    }
    event = os.environ.get("CI_OK_EVENT", "")
    base_ref = os.environ.get("CI_OK_BASE_REF", "")

    problems = decide(results, event, base_ref)
    width = max((len(j) for j in results), default=0)
    print("Job results this run:")
    for job in sorted(results):
        print(f"  {job:<{width}}  {results[job] or '<none>'}")
    # stdout and stderr are separately buffered, so without this the verdict
    # below prints ABOVE the table that justifies it.
    sys.stdout.flush()

    if problems:
        print(
            f"\nFAIL: {len(problems)} problem(s) — this run does not satisfy the tier policy:\n",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    scope = "full tier" if full_tier_owed(event, base_ref) else "reduced tier (base is not main)"
    print(f"\nOK: every job this run owed came back green ({scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
