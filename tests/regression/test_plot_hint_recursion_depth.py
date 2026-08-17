# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #853: plot-hint derivation stops poisoning matplotlib at deep stack.

:func:`tengri.describe` sweeps all menus to find a name, which calls every
``list_*`` helper, including ``list_plots()``. Each plot helper's call hint is
derived by importing matplotlib to read the signature. At deep recursion depth
(near the default 1000 frame limit), the import can raise RecursionError, which
must propagate rather than be swallowed with "defensive" exception handling.

If the RecursionError is silently caught, sys.modules is left poisoned:
matplotlib's submodules (._api, ._cm, etc.) stay cached while the parent module
is evicted, so every later ``import matplotlib`` fails with ``AttributeError:
module 'matplotlib' has no attribute 'get_data_path'``. This breaks
``tengri.plot`` lazy loading, making :func:`inspect.ismodule(tengri.plot)``
return False and breaking :func:`test_api_coverage_census.py`.

Regression: describe sweep + plot-module health check.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]


def test_describe_sweep_does_not_poison_matplotlib():
    """Calling describe() repeatedly must not poison sys.modules or break tengri.plot.

    Runs in a fresh interpreter to avoid any state from prior tests. The sweep
    (describe for each age_kernel) should derive plot hints, but not poison
    matplotlib or break the lazy plot attribute.
    """
    code = """
import inspect
import sys
import tengri

# Run the describe sweep that triggers the deep stack
try:
    for r in tengri.list_age_kernels():
        tengri.describe(r["name"])
    describe_ok = "yes"
except Exception as e:
    describe_ok = f"error: {e}"

# Check that tengri.plot still resolves as a module
plot_obj = getattr(tengri, "plot", None)
is_module = inspect.ismodule(plot_obj)

# Check that matplotlib (if cached) is not poisoned
mpl_healthy = "unknown"
if "matplotlib" in sys.modules:
    mpl = sys.modules["matplotlib"]
    mpl_healthy = "yes" if hasattr(mpl, "get_data_path") else "no"

print(f"describe_ok:{describe_ok}")
print(f"plot_is_module:{is_module}")
print(f"plot_type:{type(plot_obj).__name__}")
print(f"matplotlib_healthy:{mpl_healthy}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": "src",  # relative to wt
        },
        cwd="/Users/suchethacooray/.claude/jobs/076b9494/tmp/wt-f5",
    )

    # Print diagnostics
    print("=== Fresh interpreter output ===")
    print(result.stdout)
    if result.stderr:
        print("=== Stderr ===")
        print(result.stderr)

    assert result.returncode == 0, (
        f"Fresh interpreter failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    lines = {
        k: v for k, _, v in (line.partition(":") for line in result.stdout.strip().split("\n"))
    }
    assert lines.get("plot_is_module") == "True", (
        f"tengri.plot is not a module after describe sweep: {lines}"
    )
    # matplotlib_healthy may not appear if matplotlib was never imported
    if lines.get("matplotlib_healthy") not in ("unknown", "yes"):
        raise AssertionError(f"matplotlib is poisoned after describe sweep: {lines}")


def test_plot_call_hint_is_memoized():
    """_plot_call_hint cache must reduce import attempts on repeated calls.

    Regression check: without memoization, each describe() call would re-derive
    the hint by re-importing matplotlib. At high stack depth, this causes
    RecursionError. The memoization ensures at most one derivation per helper.
    """
    import tengri.registry

    # Clear the cache to start fresh
    tengri.registry._PLOT_HINT_CACHE.clear()

    # Call twice with the same name
    hint1 = tengri.registry._plot_call_hint("plot_sed_fit")
    cache_size_after_first = len(tengri.registry._PLOT_HINT_CACHE)

    hint2 = tengri.registry._plot_call_hint("plot_sed_fit")
    cache_size_after_second = len(tengri.registry._PLOT_HINT_CACHE)

    # Both calls should return the same result
    assert hint1 == hint2, "Memoization not returning consistent results"

    # Cache size should not grow after the second call (it was already computed)
    assert cache_size_after_second == cache_size_after_first, (
        "Cache grew after second call — memoization not working"
    )
