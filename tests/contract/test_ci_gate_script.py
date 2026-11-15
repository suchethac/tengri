# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the `gallery-changes` gate in `.github/workflows/tests.yml`.

The gate decides which `test` shards a run owes and whether the gallery runs. A
wrong answer is invisible in a green run, so the script is extracted from the
workflow and EXECUTED here, against a fake `gh`, for every event shape it must
handle. Reading the script instead would let the tests drift from the thing they
guard.

The reproductions behind these cases:

* a changed-file list longer than the pipe buffer, piped into ``grep -q`` under
  ``set -o pipefail``, makes the writer take SIGPIPE and the pipeline report
  failure for a list that DOES match, so a code file hidden behind documentation
  went undetected;
* the pull-request files API stops at 3000 entries and the compare API pages at
  300, so a list that long may be truncated;
* a rename out of ``src/`` into ``docs/`` lists only the new path unless
  ``previous_filename`` is read too.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

pytestmark = pytest.mark.contract

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "tests.yml"
_CI_OK = _ROOT / "tools" / "ci_ok.py"

_FAKE_GH = """#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
endpoint = next(a for a in args if a.startswith("repos/"))
expr = args[args.index("--jq") + 1]
d = os.environ["FAKE_GH_DIR"]
name = "pr_files.json" if "/pulls/" in endpoint else "compare.json"
path = os.path.join(d, name)
if not os.path.exists(path):
    sys.exit(1)
out = subprocess.run(["jq", "-r", expr, path], capture_output=True, text=True)
sys.stdout.write(out.stdout)
sys.exit(out.returncode)
"""

needs_tools = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="the gate script needs bash and jq",
)


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _gate_step() -> dict:
    steps = _workflow()["jobs"]["gallery-changes"]["steps"]
    return next(s for s in steps if s.get("id") == "decide")


def _ci_ok():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ci_ok_gate_pin", _CI_OK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_gate(
    tmp_path: pathlib.Path,
    *,
    event: str = "pull_request",
    files: list[dict] | None = None,
    compare: dict | None = None,
    labels: list[str] | None = None,
    draft: bool = False,
    action: str = "synchronize",
    base_changed: bool = False,
) -> dict[str, str]:
    """Run the extracted gate script and return what it wrote to GITHUB_OUTPUT."""
    fake = tmp_path / "bin"
    fake.mkdir()
    gh = fake / "gh"
    gh.write_text(_FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    if files is not None:
        (data / "pr_files.json").write_text(json.dumps(files), encoding="utf-8")
    if compare is not None:
        (data / "compare.json").write_text(json.dumps(compare), encoding="utf-8")
    script = tmp_path / "gate.sh"
    script.write_text(_gate_step()["run"], encoding="utf-8")
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    env = {
        "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "FAKE_GH_DIR": str(data),
        "GITHUB_OUTPUT": str(out),
        "GH_TOKEN": "x",
        "REPO": "o/r",
        "EVENT": event,
        "ACTION": action,
        "PR": "1",
        "DRAFT": "true" if draft else "false",
        "BASE_REF": "main",
        "BASE_CHANGED": "true" if base_changed else "false",
        "LABELS_JSON": json.dumps(labels or []),
    }
    done = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr + done.stdout
    result = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    result["_log"] = done.stdout
    return result


def _f(name: str, previous: str | None = None) -> dict:
    entry = {"filename": name}
    if previous is not None:
        entry["previous_filename"] = previous
    return entry


def _shards(result: dict[str, str]) -> list[str]:
    return json.loads(result["expected"])


CHEAP = ["physics", "regression-c"]
DOCS = [
    "contract",
    "components",
    "regression-a1",
    "regression-a2",
    "regression-a3",
    "regression-d",
]
EVERY = [
    "contract",
    "components",
    "regression-a1",
    "regression-a2",
    "regression-a3",
    "regression-b1",
    "regression-b2",
    "regression-b3",
    "regression-c",
    "regression-d",
    "physics",
]


@needs_tools
@pytest.mark.parametrize(
    ("name", "kwargs", "scope", "gallery", "shards"),
    [
        ("draft", {"draft": True, "files": [_f("src/tengri/a.py")]}, "none", "false", []),
        ("code", {"files": [_f("src/tengri/a.py")]}, "cheap", "false", CHEAP),
        (
            "code and docs",
            {"files": [_f("src/tengri/a.py"), _f("docs/x.md")]},
            "cheap-docs",
            "false",
            CHEAP + DOCS,
        ),
        ("docs only", {"files": [_f("docs/x.md"), _f("README.md")]}, "docs", "false", DOCS),
        (
            "notebooks and reproduction are docs",
            {"files": [_f("notebooks/a.py"), _f("reproduction/p/run.py")]},
            "docs",
            "false",
            DOCS,
        ),
        (
            "full-tests with code",
            {"files": [_f("src/tengri/a.py")], "labels": ["area:ci", "full-tests"]},
            "all",
            "true",
            EVERY,
        ),
        (
            "full-tests with docs only",
            {"files": [_f("docs/x.md")], "labels": ["full-tests"]},
            "all",
            "false",
            EVERY,
        ),
        (
            "a label that only contains the name does not escalate",
            {
                "files": [_f("src/tengri/a.py")],
                "labels": ["full-tests-later", "needs, full-tests"],
            },
            "cheap",
            "false",
            CHEAP,
        ),
        (
            "other workflow file",
            {"files": [_f(".github/workflows/docs.yml")]},
            "all",
            "false",
            EVERY,
        ),
        (
            "tests.yml",
            {"files": [_f(".github/workflows/tests.yml")]},
            "all",
            "true",
            EVERY,
        ),
        ("ci_ok.py", {"files": [_f("tools/ci_ok.py")]}, "all", "false", EVERY),
        ("ci_split_paths.py", {"files": [_f("tools/ci_split_paths.py")]}, "all", "false", EVERY),
        ("pyproject", {"files": [_f("pyproject.toml")]}, "all", "true", EVERY),
        ("tests conftest", {"files": [_f("tests/conftest.py")]}, "all", "false", EVERY),
        ("nested conftest", {"files": [_f("tests/physics/conftest.py")]}, "all", "false", EVERY),
        (
            "rename src to docs",
            {"files": [_f("docs/moved.md", previous="src/tengri/moved.py")]},
            "cheap-docs",
            "false",
            CHEAP + DOCS,
        ),
        (
            "rename docs to docs",
            {"files": [_f("docs/b.md", previous="docs/a.md")]},
            "docs",
            "false",
            DOCS,
        ),
        (
            "markdown under src is not docs",
            {"files": [_f("src/tengri/data/agn_fe2/PROVENANCE.md")]},
            "cheap",
            "false",
            CHEAP,
        ),
        (
            "markdown under data is not docs",
            {"files": [_f("data/NOTES.md")]},
            "cheap",
            "false",
            CHEAP,
        ),
        (
            "markdown under tests is not docs",
            {"files": [_f("tests/README.md")]},
            "cheap",
            "false",
            CHEAP,
        ),
        (
            "edited, base unchanged",
            {"files": [_f("src/tengri/a.py")], "action": "edited", "base_changed": False},
            "cheap",
            "false",
            CHEAP,
        ),
        (
            "edited, base changed",
            {"files": [_f("src/tengri/a.py")], "action": "edited", "base_changed": True},
            "cheap",
            "false",
            CHEAP,
        ),
        ("schedule", {"event": "schedule"}, "all", "true", EVERY),
        ("dispatch", {"event": "workflow_dispatch"}, "all", "true", EVERY),
        (
            "push, source change",
            {"event": "push", "compare": {"files": [_f("src/tengri/a.py")]}},
            "all",
            "true",
            EVERY,
        ),
        (
            "push, docs only still runs the gallery (pushes are coalesced)",
            {"event": "push", "compare": {"files": [_f("docs/a.md")]}},
            "all",
            "true",
            EVERY,
        ),
        (
            "push, rename out of examples",
            {"event": "push", "compare": {"files": [_f("docs/a.md", previous="examples/a.py")]}},
            "all",
            "true",
            EVERY,
        ),
        (
            "push, new branch",
            {"event": "push", "compare": {"files": [_f("docs/a.md")]}},
            "all",
            "true",
            EVERY,
        ),
        ("push, compare unreadable", {"event": "push"}, "all", "true", EVERY),
        (
            "push, compare at the page limit",
            {"event": "push", "compare": {"files": [_f(f"docs/{i}.md") for i in range(300)]}},
            "all",
            "true",
            EVERY,
        ),
        (
            "push, empty compare",
            {"event": "push", "compare": {"files": []}},
            "all",
            "true",
            EVERY,
        ),
        ("files API fails", {}, "all", "true", EVERY),
        ("empty file list", {"files": []}, "all", "true", EVERY),
        ("unrecognized event", {"event": "repository_dispatch"}, "all", "true", EVERY),
    ],
)
def test_the_gate_chooses_the_scope_the_event_owes(tmp_path, name, kwargs, scope, gallery, shards):
    result = _run_gate(tmp_path, **kwargs)
    assert (result["scope"], result["needed"]) == (scope, gallery), (name, result["_log"])
    assert sorted(_shards(result)) == sorted(shards), name


@needs_tools
def test_a_list_at_the_api_limit_runs_everything(tmp_path):
    files = [_f(f"docs/auto_examples/ex_{i}.rst") for i in range(3000)]
    result = _run_gate(tmp_path, files=files)
    assert (result["scope"], result["needed"]) == ("all", "true")
    assert sorted(_shards(result)) == sorted(EVERY)


@needs_tools
def test_a_list_just_under_the_limit_is_classified_normally(tmp_path):
    files = [_f(f"docs/auto_examples/ex_{i}.rst") for i in range(2999)]
    result = _run_gate(tmp_path, files=files)
    assert result["scope"] == "docs"


@needs_tools
@pytest.mark.parametrize("code_first", [True, False])
def test_a_long_list_hides_no_code_file_from_the_classifier(tmp_path, code_first):
    """B2: >= 200 KB of documentation paths must not hide the one code file.

    ``echo "$files" | grep -q`` under ``pipefail`` lost it whenever the writer was
    still writing when ``grep`` exited; here-strings have no writer to kill.
    """
    stem = "docs/auto_examples/agn/composable/long/path/to/an/example_with_a_long_name"
    docs = [_f(f"{stem}_{i:05d}.rst") for i in range(2900)]
    assert sum(len(f["filename"]) + 1 for f in docs) >= 200_000
    code = [_f("src/tengri/x.py")]
    files = code + docs if code_first else docs + code
    for attempt in range(6):
        attempt_dir = tmp_path / str(attempt)
        attempt_dir.mkdir()
        result = _run_gate(attempt_dir, files=files)
        assert result["scope"] == "cheap-docs", (attempt, result["_log"][-300:])


def test_the_gate_pipes_nothing_into_grep():
    run = _gate_step()["run"]
    for line in run.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "| grep" not in stripped, f"pipe into grep: {stripped}"


@needs_tools
@pytest.mark.parametrize("scope", ["all", "docs", "cheap", "cheap-docs", "none"])
def test_the_gate_table_matches_the_policy_in_ci_ok(tmp_path, scope):
    """The table's `scopes` field and `tools/ci_ok.py: SCOPE_SHARDS` say the same."""
    ci_ok = _ci_ok()
    kwargs: dict = {
        "all": {"event": "schedule"},
        "none": {"draft": True, "files": [_f("src/tengri/a.py")]},
        "cheap": {"files": [_f("src/tengri/a.py")]},
        "docs": {"files": [_f("docs/a.md")]},
        "cheap-docs": {"files": [_f("src/tengri/a.py"), _f("docs/a.md")]},
    }[scope]
    result = _run_gate(tmp_path, **kwargs)
    assert result["scope"] == scope
    all_shards = json.loads(result["all_shards"])
    assert set(_shards(result)) == set(ci_ok.scope_shards(scope, all_shards))
    assert set(all_shards) == set(EVERY)


@needs_tools
def test_the_matrix_carries_what_the_test_job_reads(tmp_path):
    result = _run_gate(tmp_path, event="schedule")
    matrix = json.loads(result["shards"])
    assert list(matrix) == ["include"]
    keys = {"shard", "paths", "bug_half", "nproc", "timeout", "cache_gb"}
    for entry in matrix["include"]:
        assert set(entry) == keys, entry
    job = json.dumps(_workflow()["jobs"]["test"])
    for key in keys:
        assert f"matrix.{key}" in job, f"matrix.{key} is defined but never read"
    read = set(re.findall(r"matrix\.(\w+)", job))
    assert read <= keys, f"the test job reads matrix keys the table lacks: {read - keys}"


def test_the_empty_matrix_is_what_the_test_job_stands_down_on():
    condition = _workflow()["jobs"]["test"]["if"]
    assert "!cancelled()" in condition
    assert "needs.smoke.result == 'success'" in condition
    assert "needs.gallery-changes.result == 'success'" in condition
    assert '{"include":[]}' in condition


def test_the_merge_queue_is_gone_from_every_workflow():
    for path in (_ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert "merge_group" not in text, f"merge_group is still named in {path.name}"


def test_pull_request_types_cover_retargeting_and_labels():
    on = _workflow()[True]
    types = set(on["pull_request"]["types"])
    assert {"opened", "synchronize", "reopened", "ready_for_review"} <= types
    assert {"edited", "labeled", "unlabeled"} <= types


def test_ci_ok_reads_the_dynamic_shard_listing():
    ci_ok = _workflow()["jobs"]["ci-ok"]
    assert ci_ok["if"] == "always()"
    assert ci_ok["permissions"]["actions"] == "read"
    env = next(s for s in ci_ok["steps"] if s.get("run") == "python3 tools/ci_ok.py")["env"]
    for key in ("CI_OK_EXPECTED_SHARDS", "CI_OK_ALL_SHARDS", "CI_OK_SHARD_JOBS_FILE"):
        assert key in env


def test_the_cache_key_rotates_weekly():
    jobs = _workflow()["jobs"]
    for name in ("test", "slow", "crossval"):
        text = json.dumps(jobs[name])
        assert "steps.week.outputs.w" in text, f"{name} cache key has no week term"


@needs_tools
def test_a_bug_duration_ledger_change_runs_everything(tmp_path):
    """The ledger moves tests between the bug/ shards, so its edit forces the full run."""
    result = _run_gate(tmp_path, files=[_f("tools/ci_bug_durations.json")])
    assert (result["scope"], result["needed"]) == ("all", "false"), result["_log"]
    assert sorted(_shards(result)) == sorted(EVERY)


@needs_tools
def test_an_inference_duration_ledger_change_runs_everything(tmp_path):
    """The ledger moves tests between the slow inference legs, so its edit forces the full run."""
    result = _run_gate(tmp_path, files=[_f("tools/ci_inference_durations.json")])
    assert (result["scope"], result["needed"]) == ("all", "false"), result["_log"]
    assert sorted(_shards(result)) == sorted(EVERY)


@needs_tools
def test_a_filename_cannot_inject_a_workflow_command(tmp_path):
    """A changed filename is echoed inside a random stop-commands region."""
    result = _run_gate(tmp_path, files=[_f("src/tengri/a.py"), _f("::warning::injected")])
    lines = result["_log"].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("::stop-commands::"))
    token = lines[start].split("::", 2)[2]
    assert len(token) >= 8, token
    end = next(i for i, line in enumerate(lines) if line == f"::{token}::")
    injected = next(i for i, line in enumerate(lines) if "::warning::injected" in line)
    assert start < injected < end, result["_log"]


def test_the_label_context_is_empty_off_pull_requests():
    jobs = _workflow()["jobs"]
    gate_labels = _gate_step()["env"]["LABELS_JSON"]
    assert "github.event_name == 'pull_request'" in gate_labels and "'[]'" in gate_labels
    ci_ok_env = next(s for s in jobs["ci-ok"]["steps"] if s.get("run") == "python3 tools/ci_ok.py")
    labels = ci_ok_env["env"]["CI_OK_LABELS"]
    assert "github.event_name == 'pull_request'" in labels and "'[]'" in labels


def _shard_table() -> list[dict]:
    run = _gate_step()["run"]
    body = re.search(r"table=\$\(cat <<'JSON'\n(.*?)\n\s*JSON\n", run, re.S)
    assert body, "the shard table heredoc moved"
    return json.loads(body.group(1))


def _selects(paths: str, test_file: pathlib.Path) -> bool:
    """Whether pytest, given ``paths`` (words as the step splits them), runs the file."""
    words = paths.split()
    ignored = [w.removeprefix("--ignore=") for w in words if w.startswith("--ignore=")]
    named = [w for w in words if not w.startswith("--")]
    rel = test_file.relative_to(_ROOT).as_posix()
    if any(rel == i or rel.startswith(i.rstrip("/") + "/") for i in ignored):
        return False
    return any(rel == n or rel.startswith(n.rstrip("/") + "/") for n in named)


def test_every_precision_and_parity_file_runs_in_exactly_one_shard():
    """A new file under precision/ or synthesizer_parity/ cannot fall through the shards (#2908).

    The regression-a shards list precision files explicitly; one of them is the
    catch-all (the directory minus the others' files). This pins that every test
    file is selected by exactly one shard.
    """
    table = _shard_table()
    for tree in ("synthesizer_parity", "precision"):
        for test_file in sorted((_ROOT / "tests" / "regression" / tree).glob("test_*.py")):
            owners = [s["shard"] for s in table if _selects(s["paths"], test_file)]
            if test_file.name in (
                "test_nebular_continuum.py",
                "test_nebular_fesc.py",
            ):
                continue  # ignored by the step itself (data-gated)
            assert len(owners) == 1, f"{test_file.name} is run by {owners}"


def test_main_pushes_share_one_uncancelled_concurrency_group():
    """Pushes coalesce: a fixed group, in-progress run finishes, one pending."""
    import yaml

    conc = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["concurrency"]
    group, cancel = str(conc["group"]), str(conc["cancel-in-progress"])
    assert "tests-main-push" in group
    assert "run_id" not in group
    assert "github.event_name != 'push'" in cancel
