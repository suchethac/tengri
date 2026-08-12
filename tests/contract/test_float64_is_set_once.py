# SPDX-License-Identifier: BSD-3-Clause
"""Double precision is a suite-wide policy, set in exactly one place.

356 test files each carried their own module-level
``jax.config.update("jax_enable_x64", True)``. All 357 calls were no-ops:
``tests/conftest.py`` sets the same flag, and pytest imports conftest before any
test module, so the flag was already on every time one of those lines ran.

They were not merely redundant, they were misleading in two ways:

* They read as if each file controlled its own precision. It does not — the
  flag is process-global, and under ``pytest-xdist`` a worker runs many files
  in one process. Whichever file set it first set it for all of them.
* 143 of them sat *between* the stdlib imports and the ``tengri`` imports,
  contorting the import block (and tripping E402) to "make sure x64 is on
  before tengri is imported" — which the call could not achieve either, since
  conftest had already imported tengri by then.

Deleting 357 copies of a policy is only safe if the surviving copy is guarded,
so this file guards it. Without these tests a future edit to conftest would
silently drop the whole suite to float32: nothing would fail, the numbers would
just quietly get worse — which is the failure mode double precision exists to
prevent.
"""

from __future__ import annotations

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CONFTEST = _TESTS_ROOT / "conftest.py"


class TestDoublePrecisionIsActuallyOn:
    """Behavioural, not a flag read: a flag can be true while some other dtype
    policy still demotes."""

    def test_a_float64_array_stays_float64(self):
        assert jnp.zeros(3, dtype=jnp.float64).dtype == np.float64

    def test_python_floats_weakly_promote_to_float64(self):
        """The case that matters for the physics: an unannotated literal in a
        kernel must not silently pin the computation to single precision."""
        assert (jnp.asarray(1.0) * 2.0).dtype == np.float64

    def test_the_flag_agrees_with_the_behaviour(self):
        assert jax.config.jax_enable_x64 is True

    def test_precision_is_enough_to_separate_neighbouring_doubles(self):
        """float32 could not represent this difference at all, so this fails
        loudly rather than subtly if the policy is lost."""
        one_plus_eps = jnp.asarray(1.0) + jnp.asarray(1e-12)
        assert float(one_plus_eps) != 1.0


def _module_level_x64_calls(path: pathlib.Path) -> list[int]:
    """Line numbers of module-scope jax_enable_x64 updates in ``path``.

    Module scope only: the one legitimate nested use is the try/finally in
    ``tests/inference/test_devices.py::test_warns_x64_disabled``, which flips
    the flag to exercise a warning and restores it. That is a test *of* the
    policy, not a duplicate of it.
    """
    try:
        mod = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for node in mod.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and "config.update" in ast.unparse(node.value.func)
            and "jax_enable_x64" in ast.unparse(node.value)
        ):
            out.append(node.lineno)
    return out


class TestItIsSetInExactlyOnePlace:
    def test_conftest_sets_it(self):
        """The one place. If this moves, the test below will find it and this
        one will say what happened."""
        assert _module_level_x64_calls(_CONFTEST), (
            f"{_CONFTEST} no longer sets jax_enable_x64 at module scope. Either "
            f"restore it or move the policy somewhere this test can see."
        )

    def test_no_test_module_sets_it_again(self):
        """Scanned rather than allow-listed, so a file added tomorrow is
        covered without editing this one."""
        offenders = {}
        for path in sorted(_TESTS_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts or path == _CONFTEST:
                continue
            lines = _module_level_x64_calls(path)
            if lines:
                offenders[str(path.relative_to(_TESTS_ROOT))] = lines
        assert not offenders, (
            f"{len(offenders)} test module(s) set jax_enable_x64 at module "
            f"scope: {sorted(offenders)[:5]}. conftest.py already sets it "
            f"before any test module is imported, so the call is a no-op that "
            f"reads as if the file controlled its own precision. Delete it."
        )

    def test_the_scan_can_actually_find_a_call(self):
        """A scanner that matched nothing would make the test above pass
        vacuously forever."""
        assert _module_level_x64_calls(_CONFTEST), "the AST scan matches nothing"
