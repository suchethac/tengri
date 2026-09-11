# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the host-table use guard (tools/check_host_tables.py).

Guards the detector's two failure modes:

1. False negatives: a module-scope table used bare inside a function (operand,
   subscript, argument to a jnp function, a derived table, a table bound by a
   tuple-unpacking loader) must be flagged.
2. False positives: ``device_table(T)``, ``np.asarray(T)``, ``len(T)``,
   ``T.shape``, a default-argument value, and module-scope arithmetic must not.

And the live tree: zero violations, with a census large enough to mean it.
"""

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_host_tables.py"
_spec = importlib.util.spec_from_file_location("check_host_tables", _TOOL)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

pytestmark = pytest.mark.unit

_MODULE = """
import numpy as np
import jax.numpy as jnp
from tengri.utils.host_array import device_table, host_array

_LAM = host_array([0.1, 0.2, 0.3])
_A = host_array([[1.0, 2.0], [3.0, 4.0]])
_COL = _A[:, 0]                      # derived at module scope: fine
_RATIO = _LAM * 2.0                  # module-scope arithmetic: fine
_W, _F = (host_array(x) for x in (np.ones(3), np.ones(3)))

def bad_operand(x):
    return x[..., None] / _LAM

def bad_subscript(x):
    return _A[:, 0:1] * x

def bad_jnp_arg(x):
    return jnp.interp(x, _LAM, _RATIO)

def bad_derived(x):
    return _COL + x

def bad_loader_bound(x):
    return jnp.sum(_W) + x

def good_device(x):
    return x[..., None] / device_table(_LAM)

def good_host(x):
    n = len(_LAM)
    edges = np.asarray(_A[:, 0])
    return n, edges, _LAM.shape

def good_default(edges=_LAM):
    return np.asarray(edges)
"""


def _verdicts(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(_MODULE)
    return {(line, name): verdict for _, line, name, verdict in mod.scan([p])}


def test_bare_uses_are_violations(tmp_path):
    v = _verdicts(tmp_path)
    bad = {name: verdict for (_, name), verdict in v.items() if verdict.startswith("VIOLATION")}
    assert set(bad) == {"_LAM", "_A", "_RATIO", "_COL", "_W"}, bad


def test_stated_uses_are_not_violations(tmp_path):
    v = _verdicts(tmp_path)
    stated = {verdict for (_, name), verdict in v.items() if not verdict.startswith("VIOLATION")}
    assert "device: device_table" in stated
    assert "host: np.*" in stated
    assert "host: len()" in stated
    assert "host: .shape" in stated
    assert "host: default value" in stated


def test_violation_count_matches_the_bad_functions(tmp_path):
    v = _verdicts(tmp_path)
    n_bad = sum(1 for verdict in v.values() if verdict.startswith("VIOLATION"))
    # bad_operand(1) + bad_subscript(1) + bad_jnp_arg(2) + bad_derived(1) + bad_loader_bound(1)
    assert n_bad == 6, v


def test_the_live_tree_has_no_violations_and_a_real_census():
    rows = mod.scan(sorted(mod.SRC.rglob("*.py")))
    bad = [r for r in rows if r[3].startswith("VIOLATION")]
    assert not bad, "\n".join(f"{p}:{ln}: {n} {v}" for p, ln, n, v in bad)
    # Non-vacuity: the guard measured something. 122 uses at the time of writing.
    assert len(rows) > 80, f"census found only {len(rows)} table uses; it has gone blind"
