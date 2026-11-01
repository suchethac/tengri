# SPDX-License-Identifier: BSD-3-Clause
"""The weekly workflow is covered by the CI guards, and each guard can go red.

``tests-weekly.yml`` is a second workflow with a ``pull_request`` trigger. Two rules keep it
from rotting: every job of every such workflow declares a tier
(``tools/check_ci_pr_coverage.py``), and every ``weekly``-marked test is selected by a
``-m weekly`` workflow step (``tools/check_slow_marks_reachable.py``).
"""

from __future__ import annotations

import importlib.util
import pathlib

_TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cov = _load("check_ci_pr_coverage")
reach = _load("check_slow_marks_reachable")


def test_every_pull_request_workflow_job_is_classified():
    assert cov.other_workflow_problems() == []


def test_an_unclassified_workflow_is_refused(monkeypatch):
    monkeypatch.setattr(cov, "OTHER_WORKFLOW_JOBS", {})
    problems = cov.other_workflow_problems()
    assert any("tests-weekly.yml" in p for p in problems)


def test_a_job_whose_tier_text_drifted_is_refused(monkeypatch):
    jobs = {**cov.OTHER_WORKFLOW_JOBS}
    jobs["tests-weekly.yml"] = {
        **jobs["tests-weekly.yml"],
        "slow-weekly": ("opt-in via a label", "no-such-label"),
    }
    monkeypatch.setattr(cov, "OTHER_WORKFLOW_JOBS", jobs)
    assert any("slow-weekly" in p for p in cov.other_workflow_problems())


def test_every_weekly_test_is_selected_by_a_weekly_step():
    assert reach.unreachable_weekly_files() == []
    assert any(p.endswith("test_profile_mass.py") for p in reach.weekly_selected_paths())


def test_a_weekly_test_no_step_selects_is_refused(monkeypatch):
    monkeypatch.setattr(reach, "weekly_selected_paths", lambda *a, **k: [])
    assert [p.name for p in reach.unreachable_weekly_files()] == ["test_profile_mass.py"]
