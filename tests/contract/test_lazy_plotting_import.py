# SPDX-License-Identifier: BSD-3-Clause
"""``import tengri`` must not import matplotlib, and must not break plotting.

Making the plotting re-exports lazy is worth ~0.4-0.8 s on every ``import
tengri`` -- inference runs, CI shards and slurm tasks that never draw anything
all paid it. The saving is only worth having if the lazy path is transparent,
and one half of that was missed: ``tengri.analysis.plotting`` existed *only* as
a side effect of ``tengri/__init__.py`` doing ``from tengri.analysis.plotting
import (...)``. Importing a submodule binds it on its parent package. Drop the
import and the attribute goes with it, so
``tengri.analysis.plotting.setup_style()`` -- the idiom ``examples/_STYLE.md``
prescribes and 45 gallery examples use -- raised ``AttributeError``.

CI could not see it. ``tools/run_gallery_examples.py`` runs every example in one
process, so the first example to touch ``tengri.plot_sed_fit`` imported the
submodule and bound the attribute for all the rest; the affected example ran
sixth and passed. It fails only when it runs first.

Hence every check here spawns a **fresh interpreter**: in-process, the test
session has already imported matplotlib and half of tengri, so all of these
would pass no matter what the source said.
"""

from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys
import warnings

import pytest

pytestmark = pytest.mark.contract

REPO = pathlib.Path(__file__).resolve().parents[2]


def _run(code: str) -> str:
    """Execute *code* in a clean interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"fresh interpreter failed:\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return result.stdout.strip()


def test_importing_tengri_does_not_import_matplotlib():
    """The whole point of the lazy path. Structural, not a stopwatch claim."""
    out = _run("import sys, tengri; print('matplotlib' in sys.modules)")
    assert out == "False", "import tengri pulled in matplotlib"


@pytest.mark.parametrize(
    "symbol",
    ["plot_sed_fit", "plot_sfh", "plot_spectrum_fit", "setup_style", "COLORS", "safe_corner"],
)
def test_the_re_exported_plotting_names_still_resolve(symbol):
    """Lazy must be invisible to callers: ``tengri.<name>`` works as before."""
    out = _run(f"import tengri; print(tengri.{symbol} is not None)")
    assert out == "True"


def test_the_plot_submodule_still_resolves():
    """``tengri.plot`` is the canonical user-facing plotting path."""
    out = _run("import tengri; print(tengri.plot.__name__)")
    assert out == "tengri.plot"


def test_analysis_plotting_resolves_after_a_bare_import():
    """The regression: this attribute must not depend on something else first.

    Ordered deliberately so nothing touches ``tengri.plot*`` beforehand -- that
    would import the submodule and bind the attribute, which is exactly the
    accident that hid the break in CI.
    """
    out = _run("import tengri; print(tengri.analysis.plotting.setup_style.__name__)")
    assert out == "setup_style"


def test_analysis_plotting_is_not_imported_until_it_is_asked_for():
    """Restoring the attribute must not restore the import cost with it."""
    out = _run(
        "import sys, tengri; "
        "print('tengri.analysis.plotting' in sys.modules, 'matplotlib' in sys.modules)"
    )
    assert out == "False False", "the plotting submodule loaded before anyone asked"


def test_an_unknown_analysis_attribute_still_raises_attribute_error():
    """A lazy ``__getattr__`` must not turn typos into silent successes."""
    out = _run(
        "import tengri\n"
        "try:\n"
        "    tengri.analysis.no_such_module\n"
        "except AttributeError:\n"
        "    print('raised')\n"
    )
    assert out == "raised"


def _attribute_paths_used_by_examples() -> list[str]:
    """Every ``tengri.a.b.c`` chain the shipped scripts actually evaluate.

    Parsed, not grepped. A regex over the same files reports
    ``tengri.predict_photometry`` from a sentence in a module docstring, and a
    guard with a known false positive is one people learn to skip.
    """
    found: set[str] = set()
    for root in ("examples", "notebooks"):
        directory = REPO / root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            try:
                with warnings.catch_warnings():
                    # Compiling someone else's file re-raises their lint as ours
                    # (a stray "\A" in a docstring). Not this test's finding.
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                parts: list[str] = []
                cur: ast.expr = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == "tengri":
                    found.add(".".join(reversed(parts)))
    return sorted(found)


def test_every_attribute_the_examples_use_resolves_after_a_bare_import():
    """Generalize the regression: no shipped script may need a warm-up import.

    ``tengri.analysis.plotting`` broke because it was bound as a side effect of
    an unrelated import. Nothing said which *other* attributes rested on the
    same accident, and the gallery cannot answer it -- one process, so the first
    example to import something binds it for all the rest.

    A fresh interpreter that has done nothing but ``import tengri`` can answer
    it. ``ast`` is imported here rather than in the child so the child's only
    import is the one under test.
    """
    paths = _attribute_paths_used_by_examples()
    assert len(paths) > 20, f"collector found only {len(paths)} paths; it has stopped working"

    code = (
        "import json, sys, tengri\n"
        f"paths = json.loads({json.dumps(json.dumps(paths))})\n"
        "bad = []\n"
        "for dotted in paths:\n"
        "    obj = tengri\n"
        "    for part in dotted.split('.'):\n"
        "        try:\n"
        "            obj = getattr(obj, part)\n"
        "        except AttributeError:\n"
        "            bad.append(dotted)\n"
        "            break\n"
        "print(json.dumps(bad))\n"
    )
    unresolved = json.loads(_run(code))
    assert not unresolved, (
        "these resolve only if something imports them first, so a script using "
        "one on its own line 1 raises AttributeError: "
        + ", ".join("tengri." + p for p in unresolved)
    )
