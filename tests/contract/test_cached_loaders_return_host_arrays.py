# SPDX-License-Identifier: BSD-3-Clause
"""Contract for #2827: a process-lifetime cache must hold host data, not JAX arrays.

``functools.cache`` / ``lru_cache`` keeps its return value for the life of the
process. If the first call runs inside a ``jax.jit`` trace and the function
builds a ``jnp`` array, the cache keeps a ``DynamicJaxprTracer`` that leaks into
every later model (``UnexpectedTracerError``). The loader must cache numpy
arrays and convert at the call site.

The check is an AST scan of ``src/tengri``: a cached function that calls a
``jnp`` function outside a ``with jax.ensure_compile_time_eval():`` block is a
violation (inside that block every ``jnp`` call is concrete, which is the guard
``fritz.py`` and ``skirtor.py`` use). ``ALLOWLIST`` names exceptions with the
reason they are safe; it must stay empty unless that reason is written down.

Marker: contract
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

SRC = Path(__file__).resolve().parents[2] / "src" / "tengri"

#: ``"relative/path.py:function_name": "why this is safe"``.
ALLOWLIST: dict[str, str] = {}

_CACHE_NAMES = {"cache", "lru_cache"}


def _decorator_is_cache(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr in _CACHE_NAMES
    if isinstance(target, ast.Name):
        return target.id in _CACHE_NAMES
    return False


def _is_jnp_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    root = node.func.value
    if isinstance(root, ast.Name) and root.id == "jnp":
        return True
    return (
        isinstance(root, ast.Attribute)
        and root.attr == "numpy"
        and isinstance(root.value, ast.Name)
        and root.value.id == "jax"
    )


def _is_compile_time_eval(node: ast.AST) -> bool:
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "ensure_compile_time_eval"
        ):
            return True
    return False


def _unguarded_jnp_calls(func: ast.FunctionDef) -> list[int]:
    """Line numbers of jnp calls not lexically inside a compile-time-eval block."""
    guarded: set[int] = set()
    for sub in ast.walk(func):
        if _is_compile_time_eval(sub):
            guarded.update(id(inner) for inner in ast.walk(sub))
    return [sub.lineno for sub in ast.walk(func) if _is_jnp_call(sub) and id(sub) not in guarded]


def _violations() -> list[str]:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(SRC.parent.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not any(_decorator_is_cache(d) for d in node.decorator_list):
                continue
            if _unguarded_jnp_calls(node) and f"{rel}:{node.name}" not in ALLOWLIST:
                found.append(f"{rel}:{node.lineno} {node.name}")
    return found


def test_cached_loaders_build_no_jax_arrays():
    assert _violations() == []


def test_scan_flags_unguarded_jnp_and_accepts_guarded(tmp_path, monkeypatch):
    """Planted cases: unguarded jnp in a cache is flagged; guarded and host-only are not."""
    import tests.contract.test_cached_loaders_return_host_arrays as mod

    (tmp_path / "planted.py").write_text(
        "import functools\nimport jax\nimport jax.numpy as jnp\n"
        "@functools.cache\ndef bad():\n    return jnp.array([1.0])\n"
        "@functools.cache\ndef guarded():\n"
        "    with jax.ensure_compile_time_eval():\n"
        "        return jnp.array([1.0])\n"
        "@functools.lru_cache(maxsize=1)\ndef host_only():\n    return [1.0]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "SRC", tmp_path)
    monkeypatch.setattr(mod, "ALLOWLIST", {})
    hits = mod._violations()
    assert len(hits) == 1
    assert hits[0].endswith(" bad")


_COLD_JIT_SCRIPT = """
import jax, jax.numpy as jnp, numpy as np
from tengri.components.agn import _nthcomp
from tengri.components.agn.disc import load_relagn_default_grid, relagn_disc_from_grid
from tengri.components.agn.unified import kubota_done_full_agn

wave = jnp.geomspace(1.0e3, 1.0e5, 48)

def kd(w, lb):
    return kubota_done_full_agn(w, agn_log_lbol=lb)

def rel(w, lb):
    return relagn_disc_from_grid(
        load_relagn_default_grid(), w, agn_log_lbol=lb, agn_log_mbh=8.0,
        agn_log_mdot=-1.0, agn_astar=0.5, agn_cos_inc=0.8,
    )

def norm(x):
    grid = _nthcomp.nthcomp_norm_grid(16)
    return x * jnp.asarray(grid).sum()

# Every cached grid loader's FIRST call happens under a jit trace; a second
# model in the same process must then still work.
for fn in (kd, rel):
    jit = jax.jit(fn)
    a, b = np.asarray(jit(wave, 44.0)), np.asarray(jit(wave, 43.0))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), fn.__name__
assert np.isfinite(float(jax.jit(norm)(1.0)))
print("OK")
"""


def test_cached_grid_loaders_survive_a_cold_first_call_under_jit():
    """Behavioral twin of the AST scan: host-side helpers that read a cached grid.

    The scan above only sees ``jnp`` calls written in the cached function itself; a helper
    such as ``_get_nu_jax`` (device array) later ``np.asarray``-ed on the host slips past it.
    A fresh interpreter makes the jit call the first call of every loader.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", _COLD_JIT_SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-1500:]
