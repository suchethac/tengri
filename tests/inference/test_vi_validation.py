"""Unit tests for run_native_vi parameter validation.

These tests exercise the ValueError and UserWarning paths that fire before
any fitter state is accessed, so fitter=None is safe to pass.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.contract


def test_n_samples_zero_raises() -> None:
    """run_native_vi raises ValueError when n_samples=0."""
    from tengri.inference.backends.vi.native import run_native_vi

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        run_native_vi(None, key=None, n_samples=0)


def test_n_samples_negative_raises() -> None:
    """run_native_vi raises ValueError when n_samples is negative."""
    from tengri.inference.backends.vi.native import run_native_vi

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        run_native_vi(None, key=None, n_samples=-5)


def test_n_iterations_zero_raises() -> None:
    """run_native_vi raises ValueError when n_iterations=0."""
    from tengri.inference.backends.vi.native import run_native_vi

    with pytest.raises(ValueError, match="n_iterations must be >= 1"):
        run_native_vi(None, key=None, n_iterations=0)


def test_n_iterations_negative_raises() -> None:
    """run_native_vi raises ValueError when n_iterations is negative."""
    from tengri.inference.backends.vi.native import run_native_vi

    with pytest.raises(ValueError, match="n_iterations must be >= 1"):
        run_native_vi(None, key=None, n_iterations=-10)


def test_high_n_samples_warns() -> None:
    """run_native_vi emits UserWarning when n_samples > 12 (before fitter access)."""
    from tengri.inference.backends.vi.native import run_native_vi

    # n_samples=0 + n_samples=13 would hit the warning first, then ValueError.
    # Use n_samples=13 with n_iterations=0 so ValueError fires after warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_iterations must be >= 1"):
            run_native_vi(None, key=None, n_samples=13, n_iterations=0)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("n_samples=13" in m for m in messages), (
        f"Expected UserWarning about n_samples=13, got: {messages}"
    )


def test_high_iterations_no_rtol_warns() -> None:
    """run_native_vi emits UserWarning for n_iterations>100 and kl_rtol<=0."""
    from tengri.inference.backends.vi.native import run_native_vi

    # n_samples=0 fires ValueError before fitter access, after both warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            run_native_vi(None, key=None, n_samples=0, n_iterations=200, kl_rtol=0.0)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("n_iterations=200" in m for m in messages), (
        f"Expected UserWarning about n_iterations=200, got: {messages}"
    )
