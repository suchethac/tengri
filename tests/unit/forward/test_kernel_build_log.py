"""Tests for the kernel build log + warn-on-failure behaviour.

PR2 replaces ``contextlib.suppress(Exception)`` around kernel builds with
``SEDModel._try_build_kernel``, which emits a UserWarning and records the
failure in ``_kernel_build_log``. These tests exercise the helper in
isolation so we don't need a real SSPData fixture.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.forward.sed_model import SEDModel


class _Bare:
    """Bare object satisfying just enough of SEDModel for the helper."""

    def __init__(self):
        self._kernel_build_log = {}


def _bound(method, instance):
    return method.__get__(instance, instance.__class__)


def test_try_build_kernel_records_ok_on_success():
    obj = _Bare()
    fn = _bound(SEDModel._try_build_kernel, obj)
    sentinel = object()
    result = fn("compositional_photometry", lambda: sentinel)
    assert result is sentinel
    assert obj._kernel_build_log["compositional_photometry"] == "ok"


def test_try_build_kernel_records_build_returned_none():
    obj = _Bare()
    fn = _bound(SEDModel._try_build_kernel, obj)
    assert fn("hybrid_spectrum", lambda: None) is None
    assert obj._kernel_build_log["hybrid_spectrum"] == "build_returned_none"


def test_try_build_kernel_warns_and_records_on_exception():
    obj = _Bare()
    fn = _bound(SEDModel._try_build_kernel, obj)

    def boom():
        raise RuntimeError("simulated XLA blowup")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn("hybrid_photometry", boom)

    assert result is None
    assert obj._kernel_build_log["hybrid_photometry"].startswith("build_failed: RuntimeError")
    assert "simulated XLA blowup" in obj._kernel_build_log["hybrid_photometry"]
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "hybrid_photometry" in str(caught[0].message)


def test_try_build_kernel_creates_log_if_missing():
    # Mirrors the case where __init__ hasn't yet set _kernel_build_log.
    class _Pristine:
        pass

    obj = _Pristine()
    fn = _bound(SEDModel._try_build_kernel, obj)
    assert fn("exact_rest_sed", lambda: object()) is not None
    assert obj._kernel_build_log["exact_rest_sed"] == "ok"


def test_list_available_kernels_returns_copy():
    obj = _Bare()
    obj._kernel_build_log = {"compositional_photometry": "ok"}
    listed = _bound(SEDModel.list_available_kernels, obj)()
    assert listed == {"compositional_photometry": "ok"}
    listed["compositional_photometry"] = "tampered"
    assert obj._kernel_build_log["compositional_photometry"] == "ok"


def test_list_available_kernels_returns_empty_if_uninit():
    class _Pristine:
        pass

    obj = _Pristine()
    listed = _bound(SEDModel.list_available_kernels, obj)()
    assert listed == {}


def test_get_strategy_returns_default_when_unset():
    from tengri.forward._kernels import DEFAULT

    class _Pristine:
        pass

    obj = _Pristine()
    strategy = _bound(SEDModel._get_strategy, obj)()
    assert strategy is DEFAULT


def test_get_strategy_returns_assigned_strategy():
    from tengri.forward._kernels import LOW_MEMORY

    obj = _Bare()
    obj._strategy = LOW_MEMORY
    strategy = _bound(SEDModel._get_strategy, obj)()
    assert strategy is LOW_MEMORY


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
