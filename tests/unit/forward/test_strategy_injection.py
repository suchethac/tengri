"""Tests for the public ``strategy=`` injection point on ``SEDModel``.

These tests verify the public surface added in PR3 without constructing a
full SEDModel (which requires SSP data on disk). They exercise the
``_get_strategy`` accessor — the single attachment point.
"""

from __future__ import annotations

import tengri
from tengri.forward._kernels import (
    COMPOSITIONAL_ONLY,
    DEFAULT,
    EXACT_ONLY,
    LOW_MEMORY,
    KernelStrategy,
)
from tengri.forward.sed_model import SEDModel


class _Fake:
    """Minimal stand-in exposing ``_strategy`` only."""

    def __init__(self, strategy):
        self._strategy = strategy


def test_default_strategy_when_unset():
    class _Pristine:
        pass

    obj = _Pristine()
    assert SEDModel._get_strategy(obj) is DEFAULT


def test_low_memory_strategy_round_trip():
    obj = _Fake(LOW_MEMORY)
    assert SEDModel._get_strategy(obj) is LOW_MEMORY


def test_exact_only_strategy_round_trip():
    obj = _Fake(EXACT_ONLY)
    assert SEDModel._get_strategy(obj) is EXACT_ONLY


def test_compositional_only_strategy_round_trip():
    obj = _Fake(COMPOSITIONAL_ONLY)
    assert SEDModel._get_strategy(obj) is COMPOSITIONAL_ONLY


def test_custom_strategy_round_trip():
    custom = KernelStrategy(preferred=("hybrid_photometry", "exact_rest_sed"))
    obj = _Fake(custom)
    assert SEDModel._get_strategy(obj) is custom


def test_public_reexports_from_tengri():
    """KernelStrategy and the built-in policies are reachable from
    ``import tengri``."""
    assert tengri.KernelStrategy is KernelStrategy
    assert tengri.DEFAULT_KERNEL_STRATEGY is DEFAULT
    assert tengri.LOW_MEMORY_KERNEL_STRATEGY is LOW_MEMORY
    assert tengri.EXACT_ONLY_KERNEL_STRATEGY is EXACT_ONLY
    assert tengri.COMPOSITIONAL_ONLY_KERNEL_STRATEGY is COMPOSITIONAL_ONLY
    assert isinstance(tengri.NoCompatibleKernelError, type)
    assert issubclass(tengri.NoCompatibleKernelError, RuntimeError)


def test_init_signature_accepts_strategy():
    import inspect

    sig = inspect.signature(SEDModel.__init__)
    assert "strategy" in sig.parameters
    assert sig.parameters["strategy"].default is None
