# SPDX-License-Identifier: BSD-3-Clause
"""Best-effort guards must leave a trace when they fire.

Four of the ten blanket ``contextlib.suppress(Exception)`` sites keep their
broad catch on purpose. The three ``Fitter`` prewarm paths and the profiler's
gradient timing all run code that can raise anything a real fit can raise, so
narrowing the exception type would let some failures escape and abort a run
that was going to succeed. Breadth was never the defect.

The defect was that failure left no trace. A warmup that silently stopped
working is indistinguishable from one that ran; the only symptom is compile
cost reappearing where the warmup existed to pay it. A profiler step whose
gradient failed to compile reports ``grad_us=None``, which reads as "not
measured" rather than "measurement failed".

So for these four the entire change is *it now logs*. That makes these tests
load-bearing in a way the narrowing tests are not: without them, a change that
did nothing would look identical to a change that worked.

Each test asserts both halves — the call still succeeds (the guard still
guards), and the reason is recorded.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.regression_bug


# ── the profiler's gradient timing ──────────────────────────────────────


def test_a_failed_gradient_timing_is_logged_and_does_not_abort(caplog):
    from tengri.profiling.pipeline import _time_step

    def _boom():
        raise RuntimeError("gradient did not compile")

    with caplog.at_level(logging.DEBUG, logger="tengri.profiling.pipeline"):
        timing = _time_step("forward", lambda: 1.0, n=1, grad_fn=_boom)

    assert timing.grad_us is None, "the guard must still swallow the failure"
    assert timing.mean_us is not None, "and the forward timing must survive it"
    assert any("gradient timing" in r.getMessage() for r in caplog.records), (
        f"the failure left no trace; records={[r.getMessage() for r in caplog.records]}"
    )


def test_the_log_names_the_step_and_the_exception(caplog):
    """A log line that does not say which step or why is barely better than silence."""
    from tengri.profiling.pipeline import _time_step

    def _boom():
        raise ValueError("a very specific reason")

    with caplog.at_level(logging.DEBUG, logger="tengri.profiling.pipeline"):
        _time_step("the_step_that_failed", lambda: 1.0, n=1, grad_fn=_boom)

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "the_step_that_failed" in joined
    assert "ValueError" in joined
    assert "a very specific reason" in joined


def test_a_working_gradient_is_timed_and_logs_nothing(caplog):
    """The direction that fails if the guard started swallowing successes."""
    from tengri.profiling.pipeline import _time_step

    with caplog.at_level(logging.DEBUG, logger="tengri.profiling.pipeline"):
        timing = _time_step("forward", lambda: 1.0, n=1, grad_fn=lambda: 2.0)

    assert timing.grad_us is not None, "a gradient that works must still be timed"
    assert not caplog.records, f"nothing should be logged on success; got {caplog.records}"


# ── the Fitter prewarm paths ────────────────────────────────────────────


class _StubFitter:
    """Duck-typed stand-in for the attributes ``_auto_prewarm`` touches.

    Building a real ``Fitter`` would drag in an SSP grid and a forward model to
    test three lines of logging. ``_auto_prewarm`` only reaches for these five
    attributes, so an unbound call with a stub exercises the real code path.
    """

    _data_args = ()

    def __init__(self, *, grad_raises: bool, predict_raises: bool):
        self._grad_raises = grad_raises
        self._predict_raises = predict_raises
        self.model = object()
        self.spec = self

    def _get_or_build_grad_fn(self):
        if self._grad_raises:
            raise RuntimeError("grad build failed")
        return lambda p, d: 0.0

    def _initialize_unbounded(self, key):
        return {}

    def sample(self, key):
        if self._predict_raises:
            raise RuntimeError("spec sample failed")
        return {}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"grad_raises": True, "predict_raises": False}, "gradient prewarm skipped"),
        ({"grad_raises": False, "predict_raises": True}, "predict-surface prewarm skipped"),
    ],
    ids=["gradient", "predict-surface"],
)
def test_a_failed_prewarm_step_is_logged_and_never_raises(caplog, kwargs, expected):
    from tengri.inference.fitter import Fitter

    stub = _StubFitter(**kwargs)
    with caplog.at_level(logging.DEBUG, logger="tengri.inference.fitter"):
        Fitter._auto_prewarm(stub, key=None)  # must not raise — prewarm is best-effort

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert expected in joined, f"expected {expected!r} in the log; got {joined!r}"
    assert "RuntimeError" in joined, "the log must name the exception type"


def test_the_prewarm_logger_is_the_fitter_module_logger():
    """Pinned because the log is useless if it cannot be switched on by name.

    The docstring on ``_prewarm_logger`` tells people to enable
    ``logging.getLogger("tengri.inference.fitter")``. If the logger were named
    anything else that instruction would be wrong, and silently so.
    """
    from tengri.inference.fitter import _prewarm_logger

    assert _prewarm_logger().name == "tengri.inference.fitter"
