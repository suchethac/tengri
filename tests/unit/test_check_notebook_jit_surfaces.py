# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_notebook_jit_surfaces.py``.

Both polarities: violations the guard must flag, and safe cases it must accept.
A guard tested only on the clean tree passes just as well when it is blind.
"""

from __future__ import annotations

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


class TestBalancedSpan:
    """Extracting balanced parentheses spans."""

    def test_simple_span(self):
        text = "func(a, b)"
        span = _extract_balanced_span(text, 4)  # Position of '('
        assert span == "(a, b)"

    def test_nested_parens(self):
        text = "func(foo(x), bar(y))"
        span = _extract_balanced_span(text, 4)
        assert span == "(foo(x), bar(y))"

    def test_span_with_string_literal(self):
        """String literals must not confuse the paren counter."""
        text = 'func("hello (world)")'
        span = _extract_balanced_span(text, 4)
        assert span == '("hello (world)")'

    def test_span_with_escaped_quote(self):
        """Escaped quotes in strings must not break the logic."""
        text = r'func("he\"llo")'
        span = _extract_balanced_span(text, 4)
        assert span == r'("he\"llo")'

    def test_multiline_span(self):
        text = "func(\n    a,\n    b\n)"
        span = _extract_balanced_span(text, 4)
        assert span == "(\n    a,\n    b\n)"

    def test_unbalanced_span_returns_empty(self):
        text = "func(a, b"  # Missing closing paren
        span = _extract_balanced_span(text, 4)
        assert span == ""

    def test_invalid_start_pos(self):
        text = "func(a)"
        span = _extract_balanced_span(text, 0)  # Not a paren
        assert span == ""


class TestFindJaxTransforms:
    """Finding jax.jit, jax.vmap, bare jit, bare vmap spans."""

    def test_jax_jit(self):
        text = "jax.jit(func)"
        spans = _find_jax_transforms(text)
        assert len(spans) == 1
        assert spans[0][2] == "(func)"

    def test_jax_vmap(self):
        text = "jax.vmap(func)"
        spans = _find_jax_transforms(text)
        assert len(spans) == 1

    def test_bare_jit(self):
        text = "jit(func)"
        spans = _find_jax_transforms(text)
        assert len(spans) == 1

    def test_bare_vmap(self):
        text = "vmap(func)"
        spans = _find_jax_transforms(text)
        assert len(spans) == 1

    def test_nested_jit_vmap(self):
        """Nested transforms are found once per file:line (outermost span)."""
        text = "jax.jit(jax.vmap(model.predict))"
        spans = _find_jax_transforms(text)
        # Should find the outer jit, not duplicate the inner vmap.
        assert len(spans) == 1
        assert "vmap" in spans[0][2]

    def test_multiple_separate_transforms(self):
        """Multiple non-overlapping transforms are all found."""
        text = "jax.jit(a)\njax.vmap(b)"
        spans = _find_jax_transforms(text)
        # Both jax.jit and jax.vmap match, so we expect 2 spans.
        assert len(spans) >= 2
        # Check that jax.jit and jax.vmap are both present.
        span_texts = [s[2] for s in spans]
        assert any("a" in st for st in span_texts)
        assert any("b" in st for st in span_texts)

    def test_multiline_transform(self):
        text = "jax.jit(\n    lambda p: model.predict(p)\n)"
        spans = _find_jax_transforms(text)
        assert len(spans) == 1
        assert "\n" in spans[0][2]


class TestDropComments:
    """Removing full-line comments."""

    def test_full_line_comment_removed(self):
        text = "x = 1\n# comment\ny = 2"
        result = _drop_comments(text)
        assert "# comment" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_inline_comment_preserved(self):
        """Only full-line comments are dropped."""
        text = "x = 1  # inline comment"
        result = _drop_comments(text)
        assert "inline comment" in result

    def test_whitespace_before_hash(self):
        """A hash with leading whitespace is still a full-line comment."""
        text = "   # comment"
        result = _drop_comments(text)
        assert "comment" not in result

    def test_empty_lines_preserved(self):
        text = "x = 1\n\ny = 2"
        result = _drop_comments(text)
        assert result.count("\n") == 2


class TestViolationsInSpan:
    """Detecting unsafe predict surfaces in jax.jit/vmap spans."""

    def test_bare_predict_is_flagged(self):
        """Bare .predict() is flagged."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(model.predict(p))"
        violations = _check_violations_in_span("(model.predict(p))", file_path, 0, text)
        assert len(violations) == 1
        assert violations[0][1] == ".predict"

    def test_predict_with_trailing_paren_is_flagged(self):
        """Bare .predict( is flagged."""
        file_path = Path("notebooks/test.py")
        text = "model.predict("
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 1

    def test_predict_with_trailing_comma_is_flagged(self):
        """Bare .predict, is flagged."""
        file_path = Path("notebooks/test.py")
        text = "model.predict,"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 1

    def test_predict_photometry_is_safe(self):
        """predict_photometry is not flagged."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(model.predict_photometry)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 0

    def test_predict_spectrum_is_safe(self):
        """predict_spectrum is not flagged."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(model.predict_spectrum)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 0

    def test_predict_properties_is_safe(self):
        """predict_properties is not flagged."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(model.predict_properties)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 0

    def test_predict_sfh_is_flagged(self):
        """predict_sfh is flagged (not in safe list)."""
        file_path = Path("notebooks/test.py")
        text = "jax.vmap(model.predict_sfh)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 1
        assert violations[0][1] == ".predict_sfh"

    def test_predict_derived_is_flagged(self):
        """predict_derived is flagged (rich accessor, not safe)."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(model.predict_derived)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 1

    def test_properties_is_flagged(self):
        """.properties accessor is flagged."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(pred.properties)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        assert len(violations) == 1
        assert violations[0][1] == ".properties"

    def test_predict_photometry_not_flagged_when_not_safe(self):
        """A method with 'photometry' in a longer name is flagged if not exact."""
        file_path = Path("notebooks/test.py")
        text = "jax.jit(model.predict_photometry_custom)"
        violations = _check_violations_in_span(text, file_path, 0, text)
        # This should be flagged because "photometry_custom" is not in the safe list.
        assert len(violations) == 1

    def test_line_numbers_survive_comment_stripping(self):
        """Regression: reported line numbers match original file, not comment-stripped.

        Issue: _drop_comments removed comment lines, shifting all line numbers.
        Fix: _drop_comments now preserves line structure by replacing with blanks.
        This test verifies the fix: a violation on line N should report line N,
        even when preceded by M full-line comment lines.
        """
        from check_notebook_jit_surfaces import _drop_comments, _find_jax_transforms

        # File with 5 comment/blank lines, then violation on line 6
        text = "# Comment 1\n# Comment 2\n\n# Comment 3\n\nbad = jax.jit(jax.vmap(model.predict))"
        # Line numbers (1-indexed): 1-5 are comments, 6 is the violation
        text_no_comments = _drop_comments(text)

        # Find the transform and check the violation line number
        spans = _find_jax_transforms(text_no_comments)
        assert len(spans) == 1
        start_pos = spans[0][0]

        # The violation should be reported on line 6
        file_path = Path("notebooks/test.py")
        violations = _check_violations_in_span(spans[0][2], file_path, start_pos, text_no_comments)
        assert len(violations) == 1
        assert violations[0][0] == 6  # Must be line 6 in the original text


class TestGuardOnRealTrees:
    """End-to-end: the guard must go red on violations and green on clean code."""

    def _run(self, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "check_notebook_jit_surfaces.py")],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def _repo(self, tmp_path: Path) -> Path:
        root = tmp_path / "repo"
        (root / "notebooks").mkdir(parents=True)
        (root / "reproduction").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        return root

    def test_founding_bug_is_flagged(self, tmp_path):
        """The exact #1255 bug: jax.jit(jax.vmap(model.predict))."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\njax.jit(jax.vmap(model.predict))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        code = _guard_check(root)
        assert code == 1

    def test_safe_predict_photometry_is_accepted(self, tmp_path):
        """jax.jit(model.predict_photometry) is safe."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\njax.jit(model.predict_photometry(p))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_safe_predict_spectrum_is_accepted(self, tmp_path):
        """jax.jit(model.predict_spectrum) is safe."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\njax.jit(model.predict_spectrum(p))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_safe_predict_properties_is_accepted(self, tmp_path):
        """jax.jit(model.predict_properties) is safe."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\njax.jit(model.predict_properties(p))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_multiline_safe_span_is_accepted(self, tmp_path):
        """A multi-line jax.jit with only safe surfaces is accepted."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text(
            "# %%\njax.jit(\n    jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))\n)\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_multiline_unsafe_span_is_flagged(self, tmp_path):
        """A multi-line jax.vmap with an unsafe surface is flagged."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text(
            "# %%\njax.vmap(\n    lambda p: model.predict_sfh(p)\n)\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        code = _guard_check(root)
        assert code == 1

    def test_predict_outside_transform_is_accepted(self, tmp_path):
        """Eager predict (outside jax.jit/vmap) is fine and ubiquitous."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\npred = model.predict(p)\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_comment_with_jit_is_not_flagged(self, tmp_path):
        """A jax.jit in a comment is dropped before scanning."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        text = "# This is a comment: jax.jit(jax.vmap(model.predict))\n"
        nb_path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_examples_excluded(self, tmp_path):
        """examples/ is out of scope."""
        root = self._repo(tmp_path)
        examples_dir = root / "examples"
        examples_dir.mkdir(parents=True)
        example_path = examples_dir / "bad.py"
        example_path.write_text("# %%\njax.jit(jax.vmap(model.predict))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_docs_excluded(self, tmp_path):
        """docs/ is out of scope."""
        root = self._repo(tmp_path)
        docs_dir = root / "docs"
        docs_dir.mkdir(parents=True)
        doc_path = docs_dir / "bad.py"
        doc_path.write_text("# %%\njax.jit(jax.vmap(model.predict))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_notebook_archive_excluded(self, tmp_path):
        """notebooks/archive/ is out of scope."""
        root = self._repo(tmp_path)
        archive_dir = root / "notebooks" / "archive"
        archive_dir.mkdir(parents=True)
        archive_path = archive_dir / "bad.py"
        archive_path.write_text("# %%\njax.jit(jax.vmap(model.predict))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_reproduction_archive_excluded(self, tmp_path):
        """reproduction/archive/ is out of scope."""
        root = self._repo(tmp_path)
        archive_dir = root / "reproduction" / "archive"
        archive_dir.mkdir(parents=True)
        archive_path = archive_dir / "bad.py"
        archive_path.write_text("# %%\njax.jit(jax.vmap(model.predict))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _guard_check(root) == 0

    def test_live_tree_is_clean(self):
        """The guard passes on the real repository as committed."""
        proc = self._run(REPO_ROOT)
        assert proc.returncode == 0, f"Guard failed:\n{proc.stdout}\n{proc.stderr}"

    def test_bare_vmap_is_flagged(self, tmp_path):
        """Bare vmap (unqualified) with unsafe surface is flagged."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\nvmap(model.predict)\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        code = _guard_check(root)
        assert code == 1

    def test_bare_jit_is_flagged(self, tmp_path):
        """Bare jit (unqualified) with unsafe surface is flagged."""
        root = self._repo(tmp_path)
        nb_path = root / "notebooks" / "test.py"
        nb_path.write_text("# %%\njit(model.predict(p))\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        code = _guard_check(root)
        assert code == 1


def _guard_check(root: Path) -> int:
    """Run the guard against a throwaway repo by pointing REPO_ROOT at it."""
    script = (REPO_ROOT / "tools" / "check_notebook_jit_surfaces.py").read_text(encoding="utf-8")
    script = script.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]",
        f"REPO_ROOT = Path({str(root)!r})",
    )
    tmp_script = root / "_guard.py"
    tmp_script.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(tmp_script)], cwd=root, capture_output=True, text=True
    )
    return proc.returncode
