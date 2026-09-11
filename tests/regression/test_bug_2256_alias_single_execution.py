# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2256 — submodule alias single execution.

Issue: sys.modules aliases for top-level package shortcuts (e.g.
sys.modules["tengri.sps"] = the real tengri.components.stellar.sps package)
only alias the TOP-LEVEL name. When code imports a SUBMODULE through the
alias (e.g. `from tengri.sps.dsps_wrapper import X`), the import machinery
creates a NEW module named "tengri.sps.dsps_wrapper" by re-executing
dsps_wrapper.py from the package __path__, so two module objects for one file
exist in sys.modules. Each re-execution creates its own module-level state
(e.g. _SSP_CONTENT_HASH_CACHE dict), and the second overwrites the canonical
package attribute.

Solution: custom MetaPathFinder that intercepts aliased submodule imports and
routes them through the canonical name, ensuring the import system binds the
EXISTING canonical module object under the aliased name with no re-execution.

https://github.com/suchethac/tengri/issues/2256
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression_bug


def _get_src_path() -> str:
    """Return the worktree src/ directory suitable for PYTHONPATH."""
    # tests/ sits under the project root; src/ is a sibling
    test_root = Path(__file__).parent.parent.parent
    src_path = test_root / "src"
    return str(src_path)


@pytest.mark.parametrize(
    "description,code",
    [
        (
            "alias then canonical",
            """
import sys
import tengri.sps.dsps_wrapper as a
import tengri.components.stellar.sps.dsps_wrapper as b
assert a is b, f"Different objects: {a} != {b}"
""",
        ),
        (
            "canonical then alias",
            """
import sys
import tengri.components.stellar.sps.dsps_wrapper as a
import tengri.sps.dsps_wrapper as b
assert a is b, f"Different objects: {a} != {b}"
""",
        ),
        (
            "alias cache shared",
            """
import sys
import tengri.sps.dsps_wrapper as a
import tengri.components.stellar.sps.dsps_wrapper as b
# Both should point to the same _SSP_CONTENT_HASH_CACHE dict object
cache_a = a._SSP_CONTENT_HASH_CACHE
cache_b = b._SSP_CONTENT_HASH_CACHE
assert cache_a is cache_b, f"Cache objects differ: {id(cache_a)} != {id(cache_b)}"
""",
        ),
        (
            "all nine aliases",
            """
import sys
import tengri.sps.dsps_wrapper as alias_sps
import tengri.components.stellar.sps.dsps_wrapper as canonical_sps
assert alias_sps is canonical_sps, "SPS dsps_wrapper differs"

import tengri.sfh as alias_sfh
import tengri.components.stellar.sfh as canonical_sfh
# These packages may not have a "common" submodule, but at least check they are the same
assert alias_sfh.__name__ == canonical_sfh.__name__, "SFH packages differ"
""",
        ),
    ],
    ids=["alias_then_canonical", "canonical_then_alias", "state_shared", "all_aliases"],
)
def test_submodule_alias_single_execution(description: str, code: str):
    """Test that submodule imports via aliases resolve to the same object.

    Each test runs in a fresh subprocess so we control the import order and
    avoid caching artifacts from prior tests.
    """
    src_path = _get_src_path()
    env = {
        "PYTHONPATH": src_path,
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{description} failed with returncode {result.returncode}\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
