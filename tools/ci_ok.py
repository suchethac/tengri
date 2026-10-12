#!/usr/bin/env python3
"""Decide the single aggregate verdict that branch protection requires.

Branch protection names four contexts: ``ci-ok``, ``lint``, ``security`` and
``smoke``. The expanded ``test (<shard>)`` contexts are deliberately NOT required:
the test matrix is built from the run's scope, so a shard the scope withholds has
no job and its context never reports. A required context that never reports
blocks the merge forever, which is why the shard list may not live in a
repository setting no file here can see; this module is what holds it.

Collapsing to one verdict is only safe if that verdict is *stricter* than
"nothing went red" -- a plain ``skipped`` must never read as approval. The
policy below therefore mirrors ``tools/check_ci_pr_coverage.py`` exactly, and
the buckets carry the same meanings:

``ALWAYS_RUN``
    Runs for every pull request whatever its base. ``skipped`` is a policy
    violation here, not an outcome, so only ``success`` is accepted.

``FULL_TIER``
    Earned by a pull request that actually targets ``main`` (and by push,
    schedule and dispatch). When it is owed, ``skipped`` is refused; when it is
    not owed -- a stacked pull request against another branch -- ``skipped`` is
    the correct result and is accepted. The one exception is ``test`` on a
    draft, whose scope ``none`` owes no shard and so builds no matrix.

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

Which shards ran
----------------
The ``test`` job is a DYNAMIC matrix: ``gallery-changes`` builds it from the run's
scope, so a shard the scope withholds has no job at all (a draft, scope ``none``,
has no matrix and ``test`` is skipped). ``needs.test.result`` is the worst of the
shards that RAN, so it cannot show a shard that was owed and never started.
:func:`shard_problems` closes that: the workflow lists the run's ``test (<shard>)``
jobs through the API, and this module requires that the gate's own ``expected``
list equals what the scope owes (:data:`SCOPE_SHARDS`), that the set that ran
equals that list, and that every shard that ran came back ``success``. A scope the
event may not choose is refused by :func:`scope_problems` (a push, the nightly and
manual runs and a ``full-tests`` pull request owe scope ``all``; a draft owes
``none``). A failure to list the jobs is a problem, never an approval.

``labeled``, ``unlabeled`` and ``edited`` are pull-request triggers so that adding
the ``full-tests`` label starts the full run and a base retarget re-runs the gate.
Every such event re-derives the same scope from the pull request's current labels
and diff and runs it for real, so the newest check suite -- the one branch
protection reads -- always carries a true verdict. The one exception is ``slow`` /
``crossval``: they stand down on a label event that is not their own label, and
:func:`labeled_job_owed` agrees, so an unrelated label never re-runs a multi-hour
suite nor reddens the pull request.

Usage
-----
Reads the ``needs`` context as JSON from the environment, so the policy is
data-driven and unit-testable rather than inline shell::

    CI_OK_RESULTS='{"lint": {"result": "success"}, ...}' python tools/ci_ok.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import NamedTuple

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

#: Events that always earn the full tier, having no base branch to weigh. A
#: ``push`` to ``main`` runs the full suite after the commit lands.
_FULL_TIER_EVENTS = ("push", "schedule", "workflow_dispatch")


#: The label that asks a pull request for the full suite instead of the cheap tier.
FULL_TESTS_LABEL = "full-tests"

#: The ``test`` scopes ``gallery-changes`` may choose, per event. A scope outside
#: the event's set means the gate withheld shards the event owed (or ran a tier it
#: should not). ``pull_request`` is narrowed further by :func:`scope_problems`.
ALLOWED_SCOPES = {
    "push": ("all",),
    "schedule": ("all",),
    "workflow_dispatch": ("all",),
    "pull_request": ("all", "docs", "cheap", "cheap-docs", "none"),
}

#: The shards of the cheap tier: what a pull request without ``full-tests`` and
#: without documentation in its diff runs.
CHEAP_SHARDS = frozenset({"physics", "regression-c"})

#: The shards whose tests read docs/, notebooks/, reproduction/ or top-level
#: ``*.md``, and so run when the diff touches those trees.
DOCS_SHARDS = frozenset({"contract", "components", "regression-a", "regression-d"})

#: Which shards each scope owes. ``all`` is every shard the gate's table defines,
#: so it is resolved against the ``all_shards`` list the gate publishes. This is
#: the policy's own copy of the table's ``scopes`` field, and
#: ``tests/contract/test_ci_gate_script.py`` pins the two together.
SCOPE_SHARDS: dict[str, frozenset[str] | None] = {
    "all": None,
    "docs": DOCS_SHARDS,
    "cheap": CHEAP_SHARDS,
    "cheap-docs": CHEAP_SHARDS | DOCS_SHARDS,
    "none": frozenset(),
}

#: How GitHub names a matrix job of ``test``: ``test (<shard>)``.
_SHARD_JOB = re.compile(r"^test \((?P<shard>[^()]+)\)$")


class ShardEvidence(NamedTuple):
    """What the run says about its ``test`` shards.

    Attributes
    ----------
    expected : sequence of str or None
        The ``expected`` output of ``gallery-changes``: the shards it put in the
        matrix. ``None`` when the output was missing or unparsable.
    all_shards : sequence of str or None
        The ``all_shards`` output: every shard the gate's table defines.
    jobs : mapping of str to str or None
        Shard name to the conclusion of its ``test (<shard>)`` job, from the
        run's job listing. ``None`` when the listing could not be obtained.
    """

    expected: Sequence[str] | None
    all_shards: Sequence[str] | None
    jobs: Mapping[str, str] | None


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


def parse_labels(raw: str) -> list[str]:
    """Parse the label names GitHub hands over as a JSON array.

    Parameters
    ----------
    raw : str
        ``toJSON(github.event.pull_request.labels.*.name)``; empty for events
        without a pull request.

    Returns
    -------
    list of str
        The label names. A label whose name contains a comma stays one name,
        which a comma-joined list could not represent.

    Raises
    ------
    ValueError
        If ``raw`` is not a JSON array of strings.
    """
    if not raw.strip():
        return []
    value = json.loads(raw)
    if value is None:  # `toJSON` of a missing label list is the literal null
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"labels must be a JSON array of strings, got {raw!r}")
    return value


def scope_shards(scope: str, all_shards: Sequence[str]) -> frozenset[str] | None:
    """The shards a scope owes.

    Parameters
    ----------
    scope : str
        A scope ``gallery-changes`` may choose.
    all_shards : sequence of str
        Every shard the gate's table defines.

    Returns
    -------
    frozenset of str or None
        The owed shards; ``None`` for a scope that is not in :data:`SCOPE_SHARDS`.
    """
    if scope not in SCOPE_SHARDS:
        return None
    fixed = SCOPE_SHARDS[scope]
    return frozenset(all_shards) if fixed is None else fixed


def scope_problems(
    event: str, scope: str, labels: Sequence[str] = (), draft: bool = False
) -> list[str]:
    """Check the ``test`` scope the gate chose against what the event owed.

    A scope that is wrong withholds shards no job result can reveal, because a
    withheld shard simply does not exist. This is the half of the verdict that
    can tell "withheld correctly" from "withheld by a bug in the gate".

    Parameters
    ----------
    event : str
        The value of ``github.event_name``.
    scope : str
        The ``scope`` output of ``gallery-changes``.
    labels : sequence of str, optional
        Names from ``github.event.pull_request.labels.*.name``.
    draft : bool, optional
        Whether the pull request is a draft.

    Returns
    -------
    list of str
        One problem per violated rule; empty when the scope is acceptable.
    """
    allowed = ALLOWED_SCOPES.get(event)
    if allowed is None:
        return [f"event `{event}` has no scope policy in tools/ci_ok.py: ALLOWED_SCOPES"]
    if scope not in allowed:
        return [
            f"`gallery-changes` chose scope `{scope or '<empty>'}` for a `{event}` run, "
            f"which may only be {', '.join(allowed)}; shards were withheld that this "
            f"event owed"
        ]
    if event != "pull_request":
        return []
    if draft and scope != "none":
        return [f"a draft pull request must be scope `none`, got `{scope}`"]
    if not draft and scope == "none":
        return ["a pull request that is not a draft was scoped `none`; no shard ran"]
    if not draft and FULL_TESTS_LABEL in set(labels) and scope != "all":
        return [
            f"the pull request carries `{FULL_TESTS_LABEL}` and so owes the full suite "
            f"(scope `all`), but the gate chose `{scope}`"
        ]
    return []


def shard_problems(scope: str, evidence: ShardEvidence) -> list[str]:
    """Check that the shards which ran are exactly the shards the scope requires.

    Parameters
    ----------
    scope : str
        The ``scope`` output of ``gallery-changes``.
    evidence : ShardEvidence
        The gate's ``expected`` and ``all_shards`` outputs and the run's
        ``test (<shard>)`` job conclusions.

    Returns
    -------
    list of str
        One problem per violated rule; empty when the shards match.
    """
    if evidence.expected is None or evidence.all_shards is None:
        return [
            "`gallery-changes` published no usable shard list "
            "(`expected` / `all_shards`), so the shards that ran cannot be checked"
        ]
    owed = scope_shards(scope, evidence.all_shards)
    if owed is None:
        return []  # an unknown scope is reported by scope_problems
    problems: list[str] = []
    missing_from_table = (CHEAP_SHARDS | DOCS_SHARDS) - set(evidence.all_shards)
    if missing_from_table:
        problems.append(
            f"the gate's shard table lacks {sorted(missing_from_table)}, which "
            f"tools/ci_ok.py: CHEAP_SHARDS / DOCS_SHARDS name"
        )
    expected = list(evidence.expected)
    if len(set(expected)) != len(expected):
        problems.append(f"the gate listed a shard twice: {expected}")
    if set(expected) != owed:
        problems.append(
            f"the gate put {sorted(set(expected))} in the matrix for scope `{scope}`, "
            f"which owes {sorted(owed)}"
        )
    if evidence.jobs is None:
        if expected:
            problems.append(
                "this run's `test (<shard>)` jobs could not be listed, so whether the "
                "owed shards ran cannot be verified"
            )
        return problems
    for shard in sorted(owed):
        got = evidence.jobs.get(shard)
        if got is None:
            problems.append(f"shard `{shard}` is owed by scope `{scope}` but never ran")
        elif got != "success":
            problems.append(f"shard `{shard}` is `{got}`")
    for shard in sorted(set(evidence.jobs) - owed):
        problems.append(f"shard `{shard}` ran but scope `{scope}` does not owe it")
    return problems


def parse_shard_jobs(raw: str) -> dict[str, str] | None:
    """Parse the run's job listing into shard name -> conclusion.

    Parameters
    ----------
    raw : str
        A JSON array of ``{"name": ..., "conclusion": ...}`` objects, or empty
        when the listing failed.

    Returns
    -------
    dict of str to str or None
        Shard name to conclusion (``"none"`` for a job with no conclusion yet).
        Skipped jobs are left out, since a skipped job ran no shard. ``None`` when
        ``raw`` is empty or unparsable.
    """
    if not raw.strip():
        return None
    try:
        listing = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(listing, list):
        return None
    jobs: dict[str, str] = {}
    for entry in listing:
        if not isinstance(entry, dict):
            return None
        match = _SHARD_JOB.match(str(entry.get("name", "")))
        if match is None:
            return None
        conclusion = str(entry.get("conclusion") or "none")
        # A skipped job did not run. A skipped shard the scope owes is then
        # reported as never run by shard_problems; an unowed one is not a shard
        # at all. Either way a skipped entry is not evidence that a shard ran.
        if conclusion == "skipped":
            continue
        jobs[match["shard"]] = conclusion
    return jobs


def labeled_job_owed(
    event: str, job: str, labels: Sequence[str], action: str = "", event_label: str = ""
) -> bool:
    """Whether a labeled job is owed on this pull request.

    Parameters
    ----------
    event : str
        The value of ``github.event_name``.
    job : str
        The job name (e.g., 'slow' or 'crossval').
    labels : sequence of str
        Label names from ``github.event.pull_request.labels.*.name``.
    action : str, optional
        The value of ``github.event.action``.
    event_label : str, optional
        The value of ``github.event.label.name``: the label a ``labeled`` or
        ``unlabeled`` event is about; empty otherwise.

    Returns
    -------
    bool
        True when the job is owed by virtue of its opt-in label being present
        on a pull request. A ``labeled`` / ``unlabeled`` event about some OTHER
        label does not owe it: the job stands down on those by design (a 3-hour
        suite must not re-run for every unrelated label), so the verdict of that
        event is the cheap tier's alone.
    """
    if event != "pull_request":
        return False
    wanted = {"slow": "run-slow-tests", "crossval": "run-crossval"}.get(job)
    if wanted is None or wanted not in set(labels):
        return False
    return not (action in ("labeled", "unlabeled") and event_label != wanted)


def decide(
    results: dict[str, str],
    event: str,
    base_ref: str,
    labels: Sequence[str] = (),
    scope: str | None = None,
    draft: bool = False,
    action: str = "",
    event_label: str = "",
    shards: ShardEvidence | None = None,
) -> list[str]:
    """Apply the tier policy to one run's job results.

    Parameters
    ----------
    results : dict
        Mapping of job name to its ``result``, one of :data:`GITHUB_RESULTS`.
    event : str
        The value of ``github.event_name``.
    base_ref : str
        The value of ``github.base_ref``; empty for non-pull-request events.
    labels : sequence of str, optional
        Label names from ``github.event.pull_request.labels.*.name``; empty for
        non-pull-request events.
    scope : str or None, optional
        The ``scope`` output of ``gallery-changes``; ``None`` when that job did
        not run (a stacked pull request), which skips the scope check.
    draft : bool, optional
        Whether the pull request is a draft.
    action : str, optional
        The value of ``github.event.action``.
    event_label : str, optional
        The value of ``github.event.label.name``.
    shards : ShardEvidence or None, optional
        The run's shard evidence; ``None`` skips the which-shards-ran check (the
        job-result rules still apply).

    Returns
    -------
    list of str
        One human-readable problem per violated rule; empty when the run may
        be treated as green. Never raises on an unknown job -- an unclassified
        job is reported, because an unlisted job is an unmade decision.
    """
    problems: list[str] = []
    owed = full_tier_owed(event, base_ref)
    if scope is not None:
        problems.extend(scope_problems(event, scope, labels, draft))
        if shards is not None:
            problems.extend(shard_problems(scope, shards))

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
        # A draft (scope `none`) owes no shard, so `test` builds no matrix and is
        # skipped. scope_problems makes that scope legal only for a draft.
        owes_nothing = job == "test" and scope == "none"
        if got is None:
            problems.append(f"`{job}` reported no result")
        elif got == "success" or (got == "skipped" and (not owed or owes_nothing)):
            # Either it ran and passed, or it stood down on a base that never
            # earned it, or it had nothing to run.
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
            elif labeled_job_owed(event, job, labels, action, event_label):
                problems.append(
                    f"`{job}` was skipped, but this pull request is labeled to run it "
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


def _json_list(raw: str) -> list[str] | None:
    """Parse a JSON array of strings from the environment; ``None`` if unusable."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def shard_jobs_for(test_result: str, listing: dict[str, str] | None) -> dict[str, str] | None:
    """The shard evidence the run really has, given the ``test`` job's result.

    A skipped ``test`` job built no matrix, so no shard ran. GitHub may still
    publish its skipped matrix entries in the run's job listing, under the literal
    name ``test (${{ matrix.shard }})``; read as shards, those would look like
    shards that ran without being owed. A skipped ``test`` therefore has an empty
    listing whatever the API returned. The scope and tier rules judge whether that
    skip was allowed.

    Parameters
    ----------
    test_result : str
        ``needs.test.result``.
    listing : dict of str to str or None
        The parsed ``test (<shard>)`` job listing, or ``None`` when unavailable.

    Returns
    -------
    dict of str to str or None
        ``{}`` when ``test`` was skipped, otherwise ``listing`` unchanged.
    """
    if test_result == "skipped":
        return {}
    return listing


def _read_shard_jobs() -> dict[str, str] | None:
    """Read the run's ``test (<shard>)`` job listing the workflow saved to a file."""
    path = os.environ.get("CI_OK_SHARD_JOBS_FILE", "").strip()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return parse_shard_jobs(handle.read())
    except OSError:
        return None


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
    try:
        labels = parse_labels(os.environ.get("CI_OK_LABELS", ""))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: CI_OK_LABELS is not a JSON array of strings: {exc}", file=sys.stderr)
        return 1

    # The gate's scope is checked only when `gallery-changes` ran and succeeded;
    # its failure is already reported by the FULL_TIER bucket.
    scope = None
    shards = None
    if results.get("gallery-changes") == "success":
        scope = os.environ.get("CI_OK_SCOPE", "").strip()
        shards = ShardEvidence(
            expected=_json_list(os.environ.get("CI_OK_EXPECTED_SHARDS", "")),
            all_shards=_json_list(os.environ.get("CI_OK_ALL_SHARDS", "")),
            jobs=shard_jobs_for(results.get("test", ""), _read_shard_jobs()),
        )
    draft = os.environ.get("CI_OK_DRAFT", "").strip().lower() == "true"
    action = os.environ.get("CI_OK_ACTION", "")
    event_label = os.environ.get("CI_OK_EVENT_LABEL", "")

    problems = decide(results, event, base_ref, labels, scope, draft, action, event_label, shards)
    width = max((len(j) for j in results), default=0)
    print("Job results this run:")
    for job in sorted(results):
        print(f"  {job:<{width}}  {results[job] or '<none>'}")
    if shards is not None and shards.jobs is not None:
        print("Test shards that ran:")
        for shard in sorted(shards.jobs):
            print(f"  {shard}  {shards.jobs[shard]}")
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

    tier = "full tier" if full_tier_owed(event, base_ref) else "reduced tier (base is not main)"
    print(f"\nOK: every job this run owed came back green ({tier}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
