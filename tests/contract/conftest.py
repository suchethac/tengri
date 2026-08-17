# SPDX-License-Identifier: BSD-3-Clause
"""Fixtures and hooks for contract tests.

Ensures test isolation for import-sensitive tests like test_api_coverage_census,
which checks that the lazy plot attribute resolves as a module.
"""

import pytest


@pytest.fixture(autouse=True)
def _cleanup_tengri_plot_pollution():
    """Reload tengri.plot module to ensure consistent state for later tests.

    Several test modules import submodules or call functions that perturb
    sys.modules (e.g., test_age_kernel_knob.py's describe() calls). Even if
    tengri.plot remains in sys.modules, the cached entry in tengri.__dict__
    may become inconsistent. Reloading the module ensures the next test sees
    a fresh, consistent import for test_api_coverage_census.py (regression guard
    for #853).

    This is a module-level fixture in tests/contract/conftest.py so it applies
    to all contract tests.
    """
    yield
    # After each test, reload tengri.plot and other lazy-imported names to ensure
    # they're in a consistent state for later tests
    try:
        import importlib
        import sys

        import tengri

        # Reload tengri.plot if it's in sys.modules
        if "tengri.plot" in sys.modules:
            importlib.reload(sys.modules["tengri.plot"])

        # Re-cache it in tengri's globals via __getattr__
        # Access it to trigger the cache (but don't delete and re-import, which
        # can cause circular import issues)
        if "plot" not in tengri.__dict__:
            # Only access if it's not already cached; accessing will trigger __getattr__
            # which will cache it properly
            _ = getattr(tengri, "plot", None)
    except Exception:
        # If cleanup fails, don't break the test run
        pass
