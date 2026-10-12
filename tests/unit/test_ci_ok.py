# SPDX-License-Identifier: BSD-3-Clause
"""The aggregate CI verdict refuses everything it is supposed to refuse.

``ci-ok`` is about to become the single context branch protection requires, so
it is the only thing standing between a red run and ``main``. A guard that can
only pass is worth nothing, and this repository has shipped three of those in
one day (#1717, #1723, #1746). Every rule below therefore has a negative case:
the test asserts the verdict goes RED for the specific mistake the rule exists
to catch, not merely that it goes green when all is well.

The rule the whole design turns on is
:func:`~tools.ci_ok.decide` refusing a *skipped* heavy job on a pull request
that targets ``main``. Collapsing eleven required contexts into one is only
safe if ``skipped`` never reads as approval.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_CI_OK = pathlib.Path(__file__).resolve().parents[2] / "tools" / "ci_ok.py"
_spec = importlib.util.spec_from_file_location("ci_ok", _CI_OK)
ci_ok = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_ok)


def _all_green() -> dict[str, str]:
    """A result set in which every job this repo gates on came back green."""
    return {
        "tier": "success",
        "lint": "success",
        "security": "success",
        "smoke": "success",
        "test": "success",
        "gallery-changes": "success",
        "gallery": "success",
        "slow": "success",
        "coverage": "success",
        "crossval": "success",
    }


class TestGreenPath:
    """The verdict passes only when it genuinely should."""

    def test_all_success_on_a_pr_against_main_passes(self):
        assert ci_ok.decide(_all_green(), "pull_request", "main") == []

    def test_all_success_on_push_passes(self):
        assert ci_ok.decide(_all_green(), "push", "") == []

    def test_gallery_skipped_is_fine_because_gallery_changes_decides_it(self):
        results = _all_green() | {"gallery": "skipped"}
        assert ci_ok.decide(results, "pull_request", "main") == []


class TestSkippedIsNotApproval:
    """The rule that makes one aggregate context safe at all."""

    @pytest.mark.parametrize("job", ["test", "gallery-changes"])
    def test_a_skipped_heavy_job_fails_when_the_pr_targets_main(self, job):
        results = _all_green() | {job: "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main")
        assert problems, f"a skipped `{job}` on a main-targeting PR must not read as approval"
        assert any(job in p for p in problems)

    @pytest.mark.parametrize("job", ["test", "gallery-changes"])
    def test_a_skipped_heavy_job_is_accepted_on_a_stacked_pr(self, job):
        # A PR based on another branch has not earned the full tier; the tier
        # job narrates that, and withholding is correct rather than a fault.
        results = _all_green() | {job: "skipped"}
        assert ci_ok.decide(results, "pull_request", "worktree-something") == []

    @pytest.mark.parametrize("job", ["tier", "lint", "security", "smoke"])
    def test_a_skipped_always_run_gate_always_fails(self, job):
        # These four run for every PR whatever its base, so `skipped` is a
        # policy violation and not an outcome — on any base.
        for base in ("main", "worktree-something"):
            results = _all_green() | {job: "skipped"}
            problems = ci_ok.decide(results, "pull_request", base)
            assert problems, f"`{job}` skipped on base {base!r} must fail"


class TestRedResultsFail:
    """Nothing red survives, whichever job it was."""

    # Derived from ci_ok.GITHUB_RESULTS rather than spelled here: those are
    # GitHub's own strings, and writing them again would put an upstream
    # data-contract word into tengri prose (NAMING_CONTRACT §10).
    @pytest.mark.parametrize("job", sorted(_all_green()))
    @pytest.mark.parametrize("bad", [r for r in ci_ok.NOT_SUCCESS if r != "skipped"])
    def test_any_red_job_fails_the_verdict(self, job, bad):
        results = _all_green() | {job: bad}
        problems = ci_ok.decide(results, "pull_request", "main")
        assert problems, f"`{job}` = {bad} must fail the aggregate verdict"

    def test_a_missing_required_job_fails(self):
        results = _all_green()
        del results["test"]
        assert ci_ok.decide(results, "pull_request", "main")


class TestUnclassifiedJob:
    """An unlisted dependency is an unmade decision, mirroring check_ci_pr_coverage."""

    def test_a_dependency_in_no_bucket_is_reported(self):
        results = _all_green() | {"brand-new-job": "success"}
        problems = ci_ok.decide(results, "pull_request", "main")
        assert any("brand-new-job" in p for p in problems)


class TestFullTierOwed:
    """The base branch, not the event alone, decides what was owed."""

    @pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
    def test_non_pr_events_always_owe_the_full_tier(self, event):
        assert ci_ok.full_tier_owed(event, "") is True

    def test_a_pr_against_main_owes_the_full_tier(self):
        assert ci_ok.full_tier_owed("pull_request", "main") is True

    def test_a_stacked_pr_does_not(self):
        assert ci_ok.full_tier_owed("pull_request", "worktree-fix-1738") is False


class TestBucketsMirrorTheCoveragePolicy:
    """ci_ok.py and check_ci_pr_coverage.py must not drift apart.

    They encode the same tier decision for the same jobs. If one grows a job
    and the other does not, the aggregate verdict starts either demanding a job
    that legitimately withholds or excusing one that must run.
    """

    def test_every_ci_ok_job_is_classified_by_the_coverage_guard(self):
        cov_path = _CI_OK.parent / "check_ci_pr_coverage.py"
        spec = importlib.util.spec_from_file_location("check_ci_pr_coverage", cov_path)
        cov = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cov)

        classified = set(cov.ALL_PR_JOBS) | set(cov.FULL_TIER_JOBS) | set(cov.ALREADY_GATED_JOBS)
        mine = (
            set(ci_ok.ALWAYS_RUN)
            | set(ci_ok.FULL_TIER)
            | set(ci_ok.GATED)
            | set(ci_ok.SCHEDULED_TIER)
        )
        assert mine <= classified, (
            f"ci_ok.py gates on {sorted(mine - classified)}, which "
            "tools/check_ci_pr_coverage.py does not classify"
        )

    def test_the_always_run_bucket_matches(self):
        cov_path = _CI_OK.parent / "check_ci_pr_coverage.py"
        spec = importlib.util.spec_from_file_location("check_ci_pr_coverage", cov_path)
        cov = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cov)

        # ci-ok itself is in ALL_PR_JOBS there but is not one of its own
        # dependencies, so exclude it from the comparison.
        assert set(ci_ok.ALWAYS_RUN) == set(cov.ALL_PR_JOBS) - {"ci-ok"}


class TestScheduledTierOwed:
    """Schedule and dispatch events owe the scheduled tier jobs."""

    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
    def test_schedule_and_dispatch_owe_scheduled_tier(self, event):
        assert ci_ok.scheduled_tier_owed(event) is True

    @pytest.mark.parametrize("event", ["push", "pull_request"])
    def test_push_and_pr_do_not_owe_scheduled_tier(self, event):
        assert ci_ok.scheduled_tier_owed(event) is False


class TestScheduledTierRedFails:
    """A failure or cancellation in scheduled tier is always a problem."""

    # The two red results, derived from ci_ok.NOT_SUCCESS so the result GitHub
    # spells the British way is written only in ci_ok.py (NAMING_CONTRACT §10:
    # that file declares it once so no other file has to write the word).
    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("bad", [r for r in ci_ok.NOT_SUCCESS if r != "skipped"])
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "pull_request", "push"])
    def test_scheduled_tier_red_fails_on_every_event(self, job, bad, event):
        results = _all_green() | {job: bad}
        base = "main" if event == "pull_request" else ""
        problems = ci_ok.decide(results, event, base)
        assert problems, f"`{job}` = {bad} on {event} must fail"
        assert any(job in p for p in problems)


class TestScheduledTierSkipped:
    """A skipped scheduled job is a fault on schedule/dispatch, OK elsewhere."""

    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
    def test_scheduled_tier_skipped_fails_on_schedule_and_dispatch(self, job, event):
        results = _all_green() | {job: "skipped"}
        problems = ci_ok.decide(results, event, "")
        assert problems, f"`{job}` skipped on {event} must fail"
        assert any(job in p for p in problems)

    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    def test_scheduled_tier_skipped_is_ok_on_push(self, job):
        results = _all_green() | {job: "skipped"}
        # Push never owes the scheduled tier, so skipped is acceptable.
        assert ci_ok.decide(results, "push", "") == []

    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("base", ["main", "worktree-fix-1738"])
    def test_scheduled_tier_skipped_is_ok_on_pull_request(self, job, base):
        results = _all_green() | {job: "skipped"}
        # PR to any base never unconditionally owes the scheduled tier
        # (they require labels on slow/crossval, and never run on coverage).
        assert ci_ok.decide(results, "pull_request", base) == []


class TestScheduledTierMissing:
    """A missing scheduled job is always a problem."""

    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "pull_request", "push"])
    def test_scheduled_tier_missing_fails_on_every_event(self, job, event):
        results = _all_green()
        del results[job]
        base = "main" if event == "pull_request" else ""
        problems = ci_ok.decide(results, event, base)
        assert problems, f"`{job}` missing on {event} must fail"
        assert any(job in p for p in problems)


class TestScheduledTierSuccess:
    """A successful scheduled job is always acceptable."""

    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "pull_request", "push"])
    def test_scheduled_tier_success_passes_on_every_event(self, job, event):
        results = _all_green()
        base = "main" if event == "pull_request" else ""
        # All green by construction, so this should pass on every event.
        assert ci_ok.decide(results, event, base) == []


class TestScheduledTierMalformed:
    """Malformed result values (not in GITHUB_RESULTS) are always a problem."""

    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("bad_result", ["", "mystery"])
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "pull_request", "push"])
    def test_scheduled_tier_malformed_fails_on_every_event(self, job, bad_result, event):
        results = _all_green() | {job: bad_result}
        base = "main" if event == "pull_request" else ""
        problems = ci_ok.decide(results, event, base)
        assert problems, f"`{job}` = {bad_result!r} on {event} must fail"
        assert any(job in p for p in problems)


class TestLabeledPullRequestSlowCrossval:
    """On pull requests, labeled slow/crossval opt-in makes skipped a violation."""

    def test_labeled_slow_with_skipped_slow_fails(self):
        """A PR labeled run-slow-tests must not have slow skipped."""
        results = _all_green() | {"slow": "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main", labels=["run-slow-tests"])
        assert problems, "labeled PR with skipped slow must fail"
        assert any("slow" in p for p in problems)

    def test_labeled_crossval_with_skipped_crossval_fails(self):
        """A PR labeled run-crossval must not have crossval skipped."""
        results = _all_green() | {"crossval": "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main", labels=["run-crossval"])
        assert problems, "labeled PR with skipped crossval must fail"
        assert any("crossval" in p for p in problems)

    def test_unlabeled_pr_with_skipped_slow_passes(self):
        """A PR without run-slow-tests label may have slow skipped."""
        results = _all_green() | {"slow": "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main", labels=[])
        assert not problems, "unlabeled PR with skipped slow should pass"

    def test_unlabeled_pr_with_skipped_crossval_passes(self):
        """A PR without run-crossval label may have crossval skipped."""
        results = _all_green() | {"crossval": "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main", labels=[])
        assert not problems, "unlabeled PR with skipped crossval should pass"

    def test_schedule_event_with_skipped_slow_fails(self):
        """Schedule event with skipped slow is a violation (unchanged)."""
        results = _all_green() | {"slow": "skipped"}
        problems = ci_ok.decide(results, "schedule", "", labels=[])
        assert problems, "schedule with skipped slow must fail"
        assert any("slow" in p for p in problems)


class TestCheapTierFailureDetection:
    """A withheld shard reports `success`; a shard that ran and went red does not.

    `test` is a matrix whose result is the worst of its shards, so a failed or
    cancelled `physics` shard in a cheap-tier run reaches ci-ok as `failure` or
    `cancelled`, while the shards the scope stood down contribute nothing red.
    """

    @pytest.mark.parametrize("scope", ["cheap", "cheap-docs", "docs", "all"])
    @pytest.mark.parametrize("bad", [r for r in ci_ok.NOT_SUCCESS if r != "skipped"])
    def test_a_red_test_job_fails_a_pull_request_whatever_the_scope(self, scope, bad):
        results = _all_green() | {"test": bad}
        problems = ci_ok.decide(results, "pull_request", "main", scope=scope)
        assert any("`test`" in p for p in problems)

    @pytest.mark.parametrize("bad", [r for r in ci_ok.NOT_SUCCESS if r != "skipped"])
    def test_a_red_test_job_fails_a_push(self, bad):
        results = _all_green() | {"test": bad}
        problems = ci_ok.decide(results, "push", "", scope="all")
        assert any("`test`" in p for p in problems)

    def test_a_cheap_pr_with_withheld_shards_standing_down_passes(self):
        assert ci_ok.decide(_all_green(), "pull_request", "main", scope="cheap") == []

    def test_a_skipped_test_job_on_a_cheap_pr_still_fails(self):
        results = _all_green() | {"test": "skipped"}
        assert ci_ok.decide(results, "pull_request", "main", scope="cheap")


class TestScopeMatchesTheEvent:
    """The one failure the job results cannot show: shards withheld wrongly."""

    @pytest.mark.parametrize("scope", ["cheap", "cheap-docs", "none", ""])
    @pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
    def test_full_suite_events_refuse_a_narrowed_scope(self, event, scope):
        assert ci_ok.scope_problems(event, scope)

    @pytest.mark.parametrize("scope", ["docs"])
    @pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
    def test_the_nightly_run_refuses_even_docs(self, event, scope):
        assert ci_ok.scope_problems(event, scope)

    @pytest.mark.parametrize("scope", ["docs", "cheap", "cheap-docs", "none", ""])
    def test_push_must_be_all(self, scope):
        assert ci_ok.scope_problems("push", scope)

    def test_push_accepts_all(self):
        assert ci_ok.scope_problems("push", "all") == []

    def test_merge_group_has_no_scope_policy(self):
        # The merge queue is unavailable on this repository; its event is gone.
        assert ci_ok.scope_problems("merge_group", "all")
        assert ci_ok.full_tier_owed("merge_group", "") is False

    def test_a_draft_must_be_scope_none(self):
        assert ci_ok.scope_problems("pull_request", "cheap", draft=True)
        assert ci_ok.scope_problems("pull_request", "none", draft=True) == []

    def test_a_ready_pr_must_not_be_scope_none(self):
        assert ci_ok.scope_problems("pull_request", "none")

    @pytest.mark.parametrize("scope", ["cheap", "cheap-docs", "docs"])
    def test_full_tests_label_owes_scope_all(self, scope):
        assert ci_ok.scope_problems("pull_request", scope, labels=["area:ci", "full-tests"])

    def test_full_tests_label_accepts_scope_all(self):
        assert ci_ok.scope_problems("pull_request", "all", labels=["area:ci", "full-tests"]) == []

    @pytest.mark.parametrize("scope", ["cheap", "cheap-docs", "docs", "all"])
    def test_an_unlabeled_pr_may_be_any_non_draft_scope(self, scope):
        assert ci_ok.scope_problems("pull_request", scope, labels=["area:ci"]) == []

    def test_an_unknown_event_has_no_policy(self):
        assert ci_ok.scope_problems("repository_dispatch", "all")

    def test_decide_reports_a_wrong_scope_even_when_every_job_is_green(self):
        problems = ci_ok.decide(_all_green(), "schedule", "", scope="cheap")
        assert any("scope" in p for p in problems)

    def test_decide_skips_the_scope_check_when_the_gate_did_not_run(self):
        # A stacked PR: gallery-changes is skipped, so there is no scope.
        results = _all_green() | {"gallery-changes": "skipped", "test": "skipped"}
        assert ci_ok.decide(results, "pull_request", "feature-branch", scope=None) == []


class TestUnrelatedLabelEvents:
    """slow/crossval stand down on a label event that is not their own label."""

    @pytest.mark.parametrize("action", ["labeled", "unlabeled"])
    def test_an_unrelated_label_event_does_not_owe_slow(self, action):
        results = _all_green() | {"slow": "skipped"}
        problems = ci_ok.decide(
            results,
            "pull_request",
            "main",
            labels=["run-slow-tests", "area:ci"],
            scope="cheap",
            action=action,
            event_label="area:ci",
        )
        assert problems == []

    def test_the_slow_label_event_itself_still_owes_slow(self):
        results = _all_green() | {"slow": "skipped"}
        problems = ci_ok.decide(
            results,
            "pull_request",
            "main",
            labels=["run-slow-tests"],
            scope="cheap",
            action="labeled",
            event_label="run-slow-tests",
        )
        assert any("slow" in p for p in problems)

    def test_a_synchronize_event_with_the_label_still_owes_slow(self):
        results = _all_green() | {"slow": "skipped"}
        problems = ci_ok.decide(
            results, "pull_request", "main", labels=["run-slow-tests"], action="synchronize"
        )
        assert any("slow" in p for p in problems)

    def test_a_red_slow_job_fails_even_on_an_unrelated_label_event(self):
        results = _all_green() | {"slow": "failure"}
        problems = ci_ok.decide(
            results,
            "pull_request",
            "main",
            labels=["run-slow-tests", "area:ci"],
            action="labeled",
            event_label="area:ci",
        )
        assert any("slow" in p for p in problems)


_ALL_SHARDS = [
    "contract",
    "components",
    "regression-a",
    "regression-b1",
    "regression-b2",
    "regression-b3",
    "regression-c",
    "regression-d",
    "physics",
]


def _evidence(scope, *, expected=None, ran=None, all_shards=None, jobs_listed=True):
    """Shard evidence for ``scope``: by default the gate and the run agree."""
    all_shards = list(_ALL_SHARDS if all_shards is None else all_shards)
    owed = sorted(ci_ok.scope_shards(scope, all_shards))
    expected = owed if expected is None else expected
    ran = owed if ran is None else ran
    jobs = {shard: "success" for shard in ran} if jobs_listed else None
    return ci_ok.ShardEvidence(expected=expected, all_shards=all_shards, jobs=jobs)


class TestWhichShardsRan:
    """The dynamic matrix: the set that ran must equal the set the scope requires."""

    @pytest.mark.parametrize("scope", ["all", "docs", "cheap", "cheap-docs", "none"])
    def test_the_owed_shards_having_run_passes(self, scope):
        results = _all_green()
        if scope == "none":
            results["test"] = "skipped"
        problems = ci_ok.decide(
            results,
            "pull_request",
            "main",
            scope=scope,
            draft=scope == "none",
            shards=_evidence(scope),
        )
        assert problems == []

    @pytest.mark.parametrize("scope", ["all", "docs", "cheap", "cheap-docs"])
    def test_a_missing_shard_fails_even_with_test_green(self, scope):
        owed = sorted(ci_ok.scope_shards(scope, _ALL_SHARDS))
        evidence = _evidence(scope, ran=owed[1:])
        problems = ci_ok.decide(_all_green(), "pull_request", "main", scope=scope, shards=evidence)
        assert any(owed[0] in p and "never ran" in p for p in problems)

    def test_a_shard_that_ran_but_was_not_owed_fails(self):
        evidence = _evidence("cheap", ran=["physics", "regression-c", "contract"])
        problems = ci_ok.decide(
            _all_green(), "pull_request", "main", scope="cheap", shards=evidence
        )
        assert any("contract" in p and "does not owe" in p for p in problems)

    @pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped", "none"])
    def test_a_red_shard_fails(self, bad):
        evidence = _evidence("cheap")
        jobs = dict(evidence.jobs) | {"physics": bad}
        evidence = ci_ok.ShardEvidence(evidence.expected, evidence.all_shards, jobs)
        problems = ci_ok.decide(
            _all_green(), "pull_request", "main", scope="cheap", shards=evidence
        )
        assert any("physics" in p and bad in p for p in problems)

    def test_a_gate_that_listed_the_wrong_shards_for_its_scope_fails(self):
        # The scope says `all`; the gate quietly put only the cheap pair in the
        # matrix and the run dutifully ran just those. Every job is green.
        evidence = _evidence(
            "all", expected=["physics", "regression-c"], ran=["physics", "regression-c"]
        )
        problems = ci_ok.decide(_all_green(), "push", "", scope="all", shards=evidence)
        assert any("owes" in p for p in problems)

    def test_an_unlistable_run_is_a_problem_not_an_approval(self):
        evidence = _evidence("cheap", jobs_listed=False)
        problems = ci_ok.decide(
            _all_green(), "pull_request", "main", scope="cheap", shards=evidence
        )
        assert any("could not be listed" in p for p in problems)

    def test_an_unlistable_run_is_fine_when_no_shard_is_owed(self):
        results = _all_green() | {"test": "skipped"}
        evidence = _evidence("none", jobs_listed=False)
        problems = ci_ok.decide(
            results, "pull_request", "main", scope="none", draft=True, shards=evidence
        )
        assert problems == []

    def test_a_missing_gate_output_is_a_problem(self):
        evidence = ci_ok.ShardEvidence(expected=None, all_shards=None, jobs={})
        problems = ci_ok.decide(
            _all_green(), "pull_request", "main", scope="cheap", shards=evidence
        )
        assert any("no usable shard list" in p for p in problems)

    def test_a_table_without_the_cheap_shards_is_a_problem(self):
        evidence = _evidence("all", all_shards=["contract"])
        problems = ci_ok.decide(_all_green(), "push", "", scope="all", shards=evidence)
        assert any("lacks" in p for p in problems)

    def test_all_owes_every_shard_the_gate_defines(self):
        assert ci_ok.scope_shards("all", ["a", "b"]) == frozenset({"a", "b"})
        assert ci_ok.scope_shards("bogus", ["a"]) is None

    def test_the_cheap_and_docs_sets_are_disjoint(self):
        assert not ci_ok.CHEAP_SHARDS & ci_ok.DOCS_SHARDS


class TestDraftRunsNoShards:
    """Scope `none` builds no matrix, so `test` is skipped and that is correct."""

    def test_a_draft_with_a_skipped_test_job_passes(self):
        results = _all_green() | {"test": "skipped", "gallery": "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main", scope="none", draft=True)
        assert problems == []

    def test_a_skipped_test_job_is_still_a_fault_for_a_ready_pr(self):
        results = _all_green() | {"test": "skipped"}
        assert ci_ok.decide(results, "pull_request", "main", scope="cheap")

    def test_a_skipped_test_job_is_a_fault_on_a_push(self):
        results = _all_green() | {"test": "skipped"}
        assert ci_ok.decide(results, "push", "", scope="all")

    def test_a_draft_whose_gate_failed_still_fails(self):
        results = _all_green() | {"test": "skipped", "gallery-changes": "failure"}
        assert ci_ok.decide(results, "pull_request", "main", scope=None, draft=True)


class TestLabelParsing:
    """Labels arrive as a JSON array; a comma inside a name must survive."""

    def test_a_json_array_parses(self):
        assert ci_ok.parse_labels('["area:ci","full-tests"]') == ["area:ci", "full-tests"]

    def test_empty_means_no_labels(self):
        assert ci_ok.parse_labels("") == []
        assert ci_ok.parse_labels("[]") == []

    def test_a_label_containing_a_comma_is_one_label(self):
        names = ci_ok.parse_labels('["needs, review"]')
        assert names == ["needs, review"]
        assert ci_ok.FULL_TESTS_LABEL not in names

    def test_a_label_that_merely_contains_the_name_does_not_opt_in(self):
        names = ci_ok.parse_labels('["full-tests-later"]')
        assert ci_ok.scope_problems("pull_request", "cheap", names) == []

    @pytest.mark.parametrize("raw", ['{"a": 1}', '["a", 1]', "not json"])
    def test_malformed_labels_are_refused(self, raw):
        with pytest.raises(ValueError):  # JSONDecodeError is a ValueError
            ci_ok.parse_labels(raw)


class TestShardJobListing:
    """The API listing is turned into shard -> conclusion, or refused."""

    def test_job_names_become_shard_names(self):
        raw = (
            '[{"name":"test (physics)","conclusion":"success"},'
            '{"name":"test (regression-c)","conclusion":"failure"}]'
        )
        assert ci_ok.parse_shard_jobs(raw) == {"physics": "success", "regression-c": "failure"}

    def test_an_empty_listing_is_an_empty_run(self):
        assert ci_ok.parse_shard_jobs("[]") == {}

    @pytest.mark.parametrize("raw", ["", "not json", '{"a": 1}', '[{"name":"lint"}]', "[1]"])
    def test_an_unusable_listing_is_none(self, raw):
        assert ci_ok.parse_shard_jobs(raw) is None

    def test_a_job_without_a_conclusion_is_not_success(self):
        jobs = ci_ok.parse_shard_jobs('[{"name":"test (physics)","conclusion":null}]')
        assert jobs == {"physics": "none"}

    def test_a_skipped_matrix_entry_is_not_a_shard_that_ran(self):
        raw = '[{"name":"test (${{ matrix.shard }})","conclusion":"skipped"}]'
        assert ci_ok.parse_shard_jobs(raw) == {}


class TestDraftListingIsNotShards:
    """A draft's skipped matrix may be listed under its literal template name.

    GitHub publishes the skipped ``test (${{ matrix.shard }})`` job of a draft run.
    Read as a shard, it would be a shard that ran without being owed, which turns
    the draft's green verdict red. A skipped ``test`` job has no shards, whatever
    the listing says.
    """

    _LITERAL = '[{"name":"test (${{ matrix.shard }})","conclusion":"skipped"}]'

    def test_a_draft_with_the_literal_skipped_listing_passes(self, monkeypatch, tmp_path):
        listing = tmp_path / "shard_jobs.json"
        listing.write_text(self._LITERAL, encoding="utf-8")
        results = _all_green() | {"test": "skipped", "gallery-changes": "success"}
        env = {
            "CI_OK_RESULTS": json.dumps({job: {"result": v} for job, v in results.items()}),
            "CI_OK_EVENT": "pull_request",
            "CI_OK_BASE_REF": "main",
            "CI_OK_SCOPE": "none",
            "CI_OK_EXPECTED_SHARDS": "[]",
            "CI_OK_ALL_SHARDS": json.dumps(list(_ALL_SHARDS)),
            "CI_OK_SHARD_JOBS_FILE": str(listing),
            "CI_OK_DRAFT": "true",
            "CI_OK_LABELS": "[]",
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert ci_ok.main() == 0

    def test_the_skipped_listing_is_the_empty_run(self):
        assert ci_ok.shard_jobs_for("skipped", ci_ok.parse_shard_jobs(self._LITERAL)) == {}

    def test_a_listing_is_kept_when_test_ran(self):
        jobs = {"physics": "success"}
        assert ci_ok.shard_jobs_for("success", jobs) == jobs
        assert ci_ok.shard_jobs_for("failure", None) is None

    def test_a_skipped_test_job_on_a_cheap_pr_still_fails_with_the_listing(self):
        jobs = ci_ok.shard_jobs_for("skipped", ci_ok.parse_shard_jobs(self._LITERAL))
        evidence = ci_ok.ShardEvidence(expected=[], all_shards=_ALL_SHARDS, jobs=jobs)
        results = _all_green() | {"test": "skipped"}
        problems = ci_ok.decide(results, "pull_request", "main", scope="cheap", shards=evidence)
        assert problems, "a withheld cheap tier on a ready pull request must not pass"


class TestNullLabels:
    """``toJSON`` of a missing label list is the literal ``null``, not a list."""

    def test_null_means_no_labels(self):
        assert ci_ok.parse_labels("null") == []
        assert ci_ok.parse_labels("  null  ") == []

    def test_a_non_null_non_list_is_still_refused(self):
        with pytest.raises(ValueError):
            ci_ok.parse_labels('"full-tests"')
