# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the one shared compiled-FLOP helper.

``tengri.profiling.compiled_flops`` is the single path every bench and test
reads FLOP counts through. It must report a real count for real work, return
``None`` (never raise) when the backend gives no count, and accept both return
shapes ``cost_analysis()`` has used across JAX releases.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.profiling import compiled_flops

pytestmark = pytest.mark.contract


def _matmul(n: int):
    a = jnp.ones((n, n), dtype=jnp.float32)
    return lambda x: x @ x, a


class _FakeCompiled:
    def __init__(self, analysis):
        self._analysis = analysis

    def cost_analysis(self):
        return self._analysis


class _FakeLowered:
    def __init__(self, analysis):
        self._analysis = analysis

    def compile(self):
        return _FakeCompiled(self._analysis)


class _FakeJit:
    def __init__(self, analysis):
        self._analysis = analysis

    def lower(self, *args):
        return _FakeLowered(self._analysis)


def _patch_analysis(monkeypatch, analysis):
    monkeypatch.setattr(jax, "jit", lambda fn: _FakeJit(analysis))


def test_matmul_reports_positive_flops_scaling_cubically():
    fn_small, a_small = _matmul(128)
    fn_big, a_big = _matmul(256)
    small = compiled_flops(fn_small, a_small)
    big = compiled_flops(fn_big, a_big)
    assert isinstance(small, int) and small > 0
    assert isinstance(big, int) and big > 0
    # Doubling every dimension of a matmul multiplies its work by 8.
    assert 6.0 <= big / small <= 10.0


def test_trivial_function_returns_int_or_none_without_raising():
    value = compiled_flops(lambda x: x + 1.0, jnp.float32(1.0))
    assert value is None or isinstance(value, int)


def test_list_shape_cost_analysis_is_handled(monkeypatch):
    _patch_analysis(monkeypatch, [{"flops": 1234.0, "bytes accessed": 8.0}])
    assert compiled_flops(lambda x: x, 0.0) == 1234


def test_dict_shape_cost_analysis_is_handled(monkeypatch):
    _patch_analysis(monkeypatch, {"flops": 4321.0})
    assert compiled_flops(lambda x: x, 0.0) == 4321


def test_empty_analysis_returns_none(monkeypatch):
    _patch_analysis(monkeypatch, [])
    assert compiled_flops(lambda x: x, 0.0) is None


def test_analysis_without_flops_key_returns_none(monkeypatch):
    _patch_analysis(monkeypatch, {"bytes accessed": 8.0})
    assert compiled_flops(lambda x: x, 0.0) is None
