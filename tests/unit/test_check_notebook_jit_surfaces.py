# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_notebook_jit_surfaces.py``.

Both polarities: violations the guard must flag, and safe cases it must accept.
A guard tested only on the clean tree passes just as well when it is blind.

Why the green-side cases assert an output count
-----------------------------------------------

The previous version of this file asserted only ``returncode == 0`` on every
accepting case. Measured, by mutating the guard and re-running:

* Pointing ``git ls-files`` at a directory that does not exist -- a guard that
  scans **nothing** -- left all eleven green-side cases passing, including
  ``test_live_tree_is_clean``. Only the four flagging cases went red.
* Emptying ``_EXCLUDED_PREFIXES`` left ``test_examples_excluded`` and
  ``test_docs_excluded`` green. Neither could observe the removal of the very
  list they were named for: ``examples/`` and ``docs/`` are outside the
  ``git ls-files -- notebooks/ reproduction/`` pathspec to begin with, so they
  are never candidates and the exclusion never runs on them.

So every accepting case here plants a violation at the path under test *and* a
clean file in scope, then asserts the guard reported ``OK -- 1 notebook(s)
clean``. That single assertion fails in both directions: red if the exclusion
stops working (exit 1), red if the scan collapses to nothing (count 0).

``reproduction/`` was never covered at all -- the old fixture created the
directory but only ever wrote to ``reproduction/archive/``, so nothing pinned
that the non-archive half is in scope.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_notebook_jit_surfaces import (
    _check_violations_in_span,
    _drop_comments,
    _extract_balanced_span,
    _find_jax_transforms,
)

pytestmark = pytest.mark.contract


def _ids(cases):
    """Case tables are (id, *args); split them for parametrize."""
    return {"argvalues": [c[1:] for c in cases], "ids": [c[0] for c in cases]}


# ---------------------------------------------------------------------------
# _extract_balanced_span
# ---------------------------------------------------------------------------

_SPAN_CASES = [
    ("simple", "func(a, b)", 4, "(a, b)"),
    ("nested_parens", "func(foo(x), bar(y))", 4, "(foo(x), bar(y))"),
    ("paren_inside_string_literal", 'func("hello (world)")', 4, '("hello (world)")'),
    ("escaped_quote_in_string", r'func("he\"llo")', 4, r'("he\"llo")'),
    ("multiline", "func(\n    a,\n    b\n)", 4, "(\n    a,\n    b\n)"),
    ("unbalanced_yields_empty", "func(a, b", 4, ""),
    ("start_pos_not_a_paren", "func(a)", 0, ""),
]


@pytest.mark.parametrize(("text", "start", "expected"), **_ids(_SPAN_CASES))
def test_balanced_span(text, start, expected):
    """A span is returned verbatim, or empty when it cannot be balanced."""
    assert _extract_balanced_span(text, start) == expected


# ---------------------------------------------------------------------------
# _find_jax_transforms
# ---------------------------------------------------------------------------

_TRANSFORM_CASES = [
    ("jax_jit", "jax.jit(func)", ["(func)"]),
    ("jax_vmap", "jax.vmap(func)", ["(func)"]),
    ("bare_jit", "jit(func)", ["(func)"]),
    ("bare_vmap", "vmap(func)", ["(func)"]),
    ("nested_keeps_outermost_only", "jax.jit(jax.vmap(m.predict))", ["(jax.vmap(m.predict))"]),
    ("multiline", "jax.jit(\n    lambda p: m.predict(p)\n)", ["(\n    lambda p: m.predict(p)\n)"]),
    ("two_calls_on_separate_lines", "jax.jit(a)\njax.vmap(b)", ["(a)", "(b)"]),
]


@pytest.mark.parametrize(("text", "expected"), **_ids(_TRANSFORM_CASES))
def test_find_jax_transforms(text, expected):
    """The exact span list, not its length.

    ``two_calls_on_separate_lines`` replaces an assertion that read
    ``assert len(spans) >= 2``. It returned four: ``jax.jit(`` and the bare
    ``jit(`` pattern both match the same call, and the de-duplication keyed on
    the start of the *transform name* while the span itself begins at the open
    paren, so the covered range fell short and the second match escaped it.
    ``>=`` cannot see a duplicate; the exact list can.
    """
    assert [span for _s, _e, span in _find_jax_transforms(text)] == expected


_UNSCANNED_TRANSFORMS = [
    ("jax_grad", "jax.grad(lambda p: m.predict(p))"),
    ("jax_value_and_grad", "jax.value_and_grad(lambda p: m.predict(p))"),
    ("bare_grad", "grad(m.predict)"),
]


@pytest.mark.parametrize(("text",), **_ids(_UNSCANNED_TRANSFORMS))
def test_gradient_transforms_are_not_scanned(text):
    """``jax.grad`` and ``jax.value_and_grad`` are outside the guard's patterns.

    Pinned because the guard's own failure message tells the reader to use the
    safe surfaces "inside jax.jit / jax.vmap / jax.grad / jax.value_and_grad
    spans" -- naming four transforms where the implementation matches two. The
    module docstring is accurate; only the advice overreaches.

    ``jax.grad`` over the rich ``predict`` surface fails at trace time the same
    way ``jax.jit`` does, so this is a coverage gap and not a design choice --
    but it is the current behavior, and stating it here keeps the gap visible
    rather than leaving it to be rediscovered. Tracked in #2063; widening the
    patterns changes what the guard rejects on the live tree, so it is not a
    drive-by.
    """
    assert _find_jax_transforms(text) == []


# ---------------------------------------------------------------------------
# _drop_comments
# ---------------------------------------------------------------------------

_COMMENT_CASES = [
    ("full_line_becomes_blank", "x = 1\n# comment\ny = 2", "x = 1\n\ny = 2"),
    ("inline_comment_survives", "x = 1  # inline comment", "x = 1  # inline comment"),
    ("indented_full_line_becomes_blank", "   # comment", ""),
    ("blank_lines_untouched", "x = 1\n\ny = 2", "x = 1\n\ny = 2"),
]


@pytest.mark.parametrize(("text", "expected"), **_ids(_COMMENT_CASES))
def test_drop_comments(text, expected):
    """Full-line comments are blanked in place, never deleted.

    Deleting them would renumber every following line. The exact expected
    string is asserted rather than a substring check, which is what pins the
    "in place" half of that contract.
    """
    assert _drop_comments(text) == expected


# ---------------------------------------------------------------------------
# _check_violations_in_span
# ---------------------------------------------------------------------------

_VIOLATION_CASES = [
    ("bare_predict_call", "(model.predict(p))", [".predict"]),
    ("predict_then_open_paren", "model.predict(", [".predict"]),
    ("predict_then_comma", "model.predict,", [".predict"]),
    ("predict_photometry_is_safe", "jax.jit(model.predict_photometry)", []),
    ("predict_spectrum_is_safe", "jax.jit(model.predict_spectrum)", []),
    ("predict_properties_is_safe", "jax.jit(model.predict_properties)", []),
    ("predict_sfh_is_not_safe", "jax.vmap(model.predict_sfh)", [".predict_sfh"]),
    ("predict_derived_is_not_safe", "jax.jit(model.predict_derived)", [".predict_derived"]),
    ("properties_accessor", "jax.jit(pred.properties)", [".properties"]),
    (
        "safe_name_must_match_exactly",
        "jax.jit(model.predict_photometry_custom)",
        [".predict_photometry_custom"],
    ),
]


@pytest.mark.parametrize(("span", "expected"), **_ids(_VIOLATION_CASES))
def test_violations_in_span(span, expected):
    """Which surface was flagged, not merely how many.

    Four of the cases this replaces asserted only ``len(violations) == 1``. A
    count cannot distinguish flagging ``.predict_derived`` from flagging some
    other token on the same line, and ``safe_name_must_match_exactly`` is
    precisely the case where the distinction carries the meaning: the name
    contains ``photometry`` but is not it.
    """
    violations = _check_violations_in_span(span, Path("notebooks/test.py"), 0, span)
    assert [token for _line, token, _excerpt in violations] == expected


_LINE_NUMBER_CASES = [
    ("flush_to_the_margin", "jax.jit(\nx.predict\n)\n", 2),
    ("indented_four", "jax.jit(\n    x.predict\n)\n", 2),
    ("indented_eight", "jax.jit(\n        x.predict\n)\n", 2),
    ("bare_jit_flush", "jit(\nx.predict\n)\n", 2),
    ("vmap_flush", "jax.vmap(\np.predict\n)\n", 2),
    ("after_blank_lines", "x = 1\n\n\n\njax.jit(\n\n\n\n    m.predict\n)\n", 9),
]


@pytest.mark.parametrize(("text", "expected_line"), **_ids(_LINE_NUMBER_CASES))
def test_reported_line_is_the_line_the_violation_is_on(text, expected_line):
    """Regression: the reported line survives the span/text coordinate change.

    ``_check_violations_in_span`` adds an offset *into the span text* to the
    span's recorded start, so that start has to be the position of the open
    paren -- where the span text begins. It used to be the position of the
    transform name, seven characters earlier for ``jax.jit``, which walked the
    measured point backwards across the newline whenever the violation sat
    within seven characters of its line's start.

    Measured before the fix: ``indented_eight`` passed and every other case
    here reported the line above. Eight spaces of indent is enough to absorb
    the shift; four is not, and four is the common layout. The one line-number
    test this file used to carry was single-line, where the shift cannot cross
    a newline and so cannot be observed.
    """
    violations = []
    for start, _end, span in _find_jax_transforms(text):
        violations += _check_violations_in_span(span, Path("notebooks/test.py"), start, text)
    assert [line for line, _token, _excerpt in violations] == [expected_line]


def test_line_numbers_survive_comment_stripping():
    """Regression: comment lines are blanked, so they still occupy their lines.

    Five comment/blank lines precede the violation; it must report line 6, not
    line 1.
    """
    text = "# Comment 1\n# Comment 2\n\n# Comment 3\n\nbad = jax.jit(jax.vmap(model.predict))"
    stripped = _drop_comments(text)
    spans = _find_jax_transforms(stripped)
    assert len(spans) == 1

    start, _end, span = spans[0]
    violations = _check_violations_in_span(span, Path("notebooks/test.py"), start, stripped)
    assert [line for line, _t, _e in violations] == [6]


# ---------------------------------------------------------------------------
# End to end, against throwaway git trees
# ---------------------------------------------------------------------------

#: The #1255 founding bug, violation on line 2.
_BUG = "# %%\njax.jit(jax.vmap(model.predict))\n"

#: A clean in-scope notebook, used to prove the guard scanned anything at all.
_CLEAN = "# %%\npred = model.predict_photometry(p)\n"

_FLAGGED_CASES = [
    ("founding_bug", {"notebooks/nb.py": _BUG}, "notebooks/nb.py:2: .predict"),
    (
        "multiline_unsafe_span",
        {"notebooks/nb.py": "# %%\njax.vmap(\n    lambda p: model.predict_sfh(p)\n)\n"},
        "notebooks/nb.py:3: .predict_sfh",
    ),
    (
        "violation_at_the_line_margin",
        {"notebooks/nb.py": "# %%\njax.jit(\n    x.predict\n)\n"},
        "notebooks/nb.py:3: .predict",
    ),
    (
        "bare_vmap",
        {"notebooks/nb.py": "# %%\nvmap(model.predict)\n"},
        "notebooks/nb.py:2: .predict",
    ),
    (
        "bare_jit",
        {"notebooks/nb.py": "# %%\njit(model.predict(p))\n"},
        "notebooks/nb.py:2: .predict",
    ),
    ("reproduction_is_in_scope", {"reproduction/run.py": _BUG}, "reproduction/run.py:2: .predict"),
]

_ACCEPTED_CASES = [
    ("safe_photometry", {"notebooks/nb.py": "# %%\njax.jit(model.predict_photometry(p))\n"}, 1),
    ("safe_spectrum", {"notebooks/nb.py": "# %%\njax.jit(model.predict_spectrum(p))\n"}, 1),
    ("safe_properties", {"notebooks/nb.py": "# %%\njax.jit(model.predict_properties(p))\n"}, 1),
    (
        "multiline_safe_span",
        {
            "notebooks/nb.py": "# %%\njax.jit(\n"
            "    jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))\n)\n"
        },
        1,
    ),
    (
        "eager_predict_outside_a_transform",
        {"notebooks/nb.py": "# %%\npred = model.predict(p)\n"},
        1,
    ),
    (
        "violation_inside_a_comment",
        {"notebooks/nb.py": "# a comment: jax.jit(jax.vmap(model.predict))\n"},
        1,
    ),
    # Each exclusion plants the founding bug at the excluded path AND a clean
    # file in scope. Green alone would not distinguish "excluded" from "never
    # looked"; the scanned count does.
    ("excluded_examples", {"examples/bad.py": _BUG, "notebooks/nb.py": _CLEAN}, 1),
    ("excluded_docs", {"docs/bad.py": _BUG, "notebooks/nb.py": _CLEAN}, 1),
    (
        "excluded_notebooks_archive",
        {"notebooks/archive/bad.py": _BUG, "notebooks/nb.py": _CLEAN},
        1,
    ),
    (
        "excluded_reproduction_archive",
        {"reproduction/archive/bad.py": _BUG, "notebooks/nb.py": _CLEAN},
        1,
    ),
    # All six clean bodies in one tree: proves they do not interact, and that
    # the scanned count tracks the tree rather than being a constant.
    (
        "every_clean_body_in_one_tree",
        {
            "notebooks/a.py": "# %%\njax.jit(model.predict_photometry(p))\n",
            "notebooks/b.py": "# %%\njax.jit(model.predict_spectrum(p))\n",
            "notebooks/c.py": "# %%\njax.jit(model.predict_properties(p))\n",
            "notebooks/d.py": "# %%\npred = model.predict(p)\n",
            "notebooks/e.py": "# a comment: jax.jit(jax.vmap(model.predict))\n",
            "reproduction/f.py": _CLEAN,
        },
        6,
    ),
]


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A throwaway git repo holding exactly ``files``, all tracked."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def _run_guard(root: Path) -> subprocess.CompletedProcess:
    """Run the guard with its REPO_ROOT pointed at a throwaway tree."""
    script = (REPO_ROOT / "tools" / "check_notebook_jit_surfaces.py").read_text(encoding="utf-8")
    script = script.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]",
        f"REPO_ROOT = Path({str(root)!r})",
    )
    tmp_script = root / "_guard.py"
    tmp_script.write_text(script, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(tmp_script)], cwd=root, capture_output=True, text=True
    )


@pytest.mark.parametrize(("files", "expected_report"), **_ids(_FLAGGED_CASES))
def test_guard_flags(tmp_path, files, expected_report):
    """Exit 1, and the report names the offending file, line and surface.

    The exit code alone would accept a guard that failed for an unrelated
    reason -- or that flagged the right file at the wrong line, which is the
    defect ``violation_at_the_line_margin`` covers end to end.
    """
    proc = _run_guard(_make_repo(tmp_path, files))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert expected_report in proc.stdout, proc.stdout


@pytest.mark.parametrize(("files", "expected_clean"), **_ids(_ACCEPTED_CASES))
def test_guard_accepts(tmp_path, files, expected_clean):
    """Exit 0, and the guard says how many in-scope notebooks it scanned.

    The count is the part that matters: a guard scanning nothing exits 0 on
    every case in this table, and used to pass all of them.
    """
    proc = _run_guard(_make_repo(tmp_path, files))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"OK -- {expected_clean} notebook(s) clean." in proc.stdout, proc.stdout


def test_live_tree_is_clean():
    """The guard passes on the real repository, having scanned it.

    ``returncode == 0`` on its own is also what a guard that enumerates no
    files returns, so the scanned count is asserted too.
    """
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "check_notebook_jit_surfaces.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Guard failed:\n{proc.stdout}\n{proc.stderr}"

    match = re.search(r"OK -- (\d+) notebook\(s\) clean\.", proc.stdout)
    assert match is not None, proc.stdout
    assert int(match.group(1)) > 0, proc.stdout
