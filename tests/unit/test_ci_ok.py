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

    # Derived from ci_ok.GITHUB_RESULTS; failure and cancellation always fail.
    @pytest.mark.parametrize("job", sorted(ci_ok.SCHEDULED_TIER))
    @pytest.mark.parametrize("bad", ["failure", "cancelled"])
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
