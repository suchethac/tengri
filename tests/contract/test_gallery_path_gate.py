# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the CI filter that decides whether the gallery job runs.

The `gallery-changes` job in `.github/workflows/tests.yml` skips the gallery
when a pull request touches nothing the examples depend on. A skip is invisible
in a green run, so a filter that silently stops matching `src/tengri/` would
un-cover all 264 examples with no failing check anywhere -- exactly the class of
silent failure the gallery itself was added to close (#1146).

The pattern is read OUT of the workflow rather than copied here, so these tests
cannot drift away from the thing they guard: edit the filter and they re-check
the new one.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

_WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tests.yml"

# The filter as written in the workflow: `grep -qE '<pattern>'`.
_GREP_LINE = re.compile(r"grep -qE '([^']+)'")

# Paths whose contents can change what a gallery example does, and paths that
# cannot. `docs/auto_examples/` is the generated OUTPUT of the gallery and is
# the reason the pattern is anchored -- unanchored, it matches `examples/`.
_MUST_RUN = [
    "examples/agn/plot_torus.py",
    "src/tengri/forward/sed_model.py",
    "data/ssp_wne.h5",
    "tools/run_gallery_examples.py",
    "pyproject.toml",
    ".github/workflows/tests.yml",
]
_MUST_SKIP = [
    "docs/index.md",
    "docs/auto_examples/agn/plot_torus.py",
    "notebooks/11_catalog_fits.py",
    "scripts/sync_spine_notebooks_for_docs.py",
    "tools/check_british_spelling.py",
    ".github/workflows/docs.yml",
    "tests/contract/test_gallery_path_gate.py",
    "mydata/thing.h5",
    "vendor/pyproject.toml",
]


def _pattern() -> re.Pattern[str]:
    """The gallery filter, extracted from the workflow it actually runs in."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    matches = _GREP_LINE.findall(text)
    assert matches, f"no `grep -qE '...'` filter found in {_WORKFLOW}"
    assert len(matches) == 1, f"expected exactly one filter, found {len(matches)}"
    return re.compile(matches[0])


@pytest.mark.parametrize("path", _MUST_RUN)
def test_gallery_runs_when_its_inputs_change(path: str) -> None:
    assert _pattern().search(path), (
        f"{path!r} can change a gallery example but the filter would SKIP the "
        f"gallery -- a broken example would ship green"
    )


@pytest.mark.parametrize("path", _MUST_SKIP)
def test_gallery_skips_when_nothing_it_reads_changes(path: str) -> None:
    assert not _pattern().search(path), (
        f"{path!r} cannot change a gallery example, but the filter would still "
        f"run all four shards (~8-10 min each)"
    )


def test_the_filter_is_anchored() -> None:
    """Unanchored, `examples/` matches the generated `docs/auto_examples/`."""
    assert _pattern().pattern.startswith("^"), (
        "the gallery filter must be anchored at the start of the path"
    )
