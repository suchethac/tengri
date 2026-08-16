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

import subprocess
import sys

import pytest

pytestmark = pytest.mark.contract


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
