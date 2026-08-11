# SPDX-License-Identifier: BSD-3-Clause
"""The inert-file detector must not fire on an honest data gate.

`tests/components/dust/test_dust_emission_traceable.py` skipped 6 of 6 on a
stale `filter_waves` kwarg — an `except Exception: pytest.skip(...)` converting
a `TypeError` into a skip — and stayed green while verifying nothing (#1615).

A naive rule ("flag any file that skips 100%") is unusable here: CI does not
ship the CLOUDY, Cue, CB19 or MAPPINGS grids, so whole files skip entirely and
correctly, every run. Measured on `test_nebular_gradients.py`: 2 ran, 8 skipped,
all eight on absent grids.

The discriminator is *when* the skip happened.

===================================  =============  ==================
skip source                          report phase   verdict
===================================  =============  ==================
``@pytest.mark.skipif`` / module      ``setup``      honest data gate
``pytest.skip()`` in a test body      ``call``       test started, then
                                                     something converted
                                                     a failure to a skip
===================================  =============  ==================

So the rule is: nothing ran **and** at least one skip came from a test body.
A file gated entirely by markers can never trip it, however many tests it
skips. These tests pin that property — without them the rule looks identical
to the naive one that would fire on half of CI.
"""

from __future__ import annotations

import pytest

from tests.conftest import _FILE_OUTCOMES, inert_test_files, pytest_runtest_logreport

pytestmark = pytest.mark.contract


class _Report:
    """Minimal stand-in for a pytest TestReport.

    Only the four attributes the hook reads. Building real reports would mean
    running real tests, which is what this test exists to avoid.
    """

    def __init__(self, nodeid, when, outcome, wasxfail=None):
        self.nodeid = nodeid
        self.when = when
        self.outcome = outcome
        if wasxfail is not None:
            self.wasxfail = wasxfail

    @property
    def skipped(self):
        return self.outcome == "skipped"


@pytest.fixture(autouse=True)
def _isolate_tally():
    """The hook writes to module state shared with the live session."""
    saved = dict(_FILE_OUTCOMES)
    _FILE_OUTCOMES.clear()
    yield
    _FILE_OUTCOMES.clear()
    _FILE_OUTCOMES.update(saved)


def _passing_test(path, name="test_a"):
    pytest_runtest_logreport(_Report(f"{path}::{name}", "setup", "passed"))
    pytest_runtest_logreport(_Report(f"{path}::{name}", "call", "passed"))


def _marker_skipped_test(path, name="test_a"):
    """A skipif marker: pytest reports skipped at setup and never calls."""
    pytest_runtest_logreport(_Report(f"{path}::{name}", "setup", "skipped"))


def _body_skipped_test(path, name="test_a"):
    """`pytest.skip()` reached inside the test: setup passes, call skips."""
    pytest_runtest_logreport(_Report(f"{path}::{name}", "setup", "passed"))
    pytest_runtest_logreport(_Report(f"{path}::{name}", "call", "skipped"))


# ── the case the detector exists for ────────────────────────────────────


def test_a_file_skipping_entirely_from_test_bodies_is_flagged():
    """The dust file's exact signature."""
    for i in range(6):
        _body_skipped_test("tests/x/test_inert.py", f"test_{i}")

    flagged = dict(inert_test_files())
    assert "tests/x/test_inert.py" in flagged
    assert flagged["tests/x/test_inert.py"]["skip_call"] == 6


# ── and the cases it must stay silent on ────────────────────────────────


def test_a_file_gated_entirely_by_markers_is_not_flagged():
    """CI lacks the CLOUDY/Cue/CB19/MAPPINGS grids; those files skip wholesale.

    This is the property that makes the rule usable at all. Without it the
    detector would fire every run on files that are behaving correctly.
    """
    for i in range(8):
        _marker_skipped_test("tests/x/test_data_gated.py", f"test_{i}")

    assert not inert_test_files()


def test_a_file_that_mixes_a_data_gate_with_real_tests_is_not_flagged():
    """`test_nebular_gradients.py`: 2 ran, 8 marker-skipped."""
    _passing_test("tests/x/test_mixed.py", "test_ran_1")
    _passing_test("tests/x/test_mixed.py", "test_ran_2")
    for i in range(8):
        _marker_skipped_test("tests/x/test_mixed.py", f"test_gated_{i}")

    assert not inert_test_files()


def test_one_body_skip_among_passing_tests_is_not_flagged():
    """An occasional in-body skip is ordinary; only a file of *nothing else* is not."""
    _passing_test("tests/x/test_mostly_fine.py", "test_ran")
    _body_skipped_test("tests/x/test_mostly_fine.py", "test_skipped")

    assert not inert_test_files()


def test_an_xfail_counts_as_having_run():
    """xfail arrives as a call-phase skip carrying ``wasxfail``.

    A strict xfail is a test that ran and behaved as declared — the opposite of
    an absent test. Counting it as a skip would flag any file whose tests are
    all xfail, such as a fully-quarantined regression file.
    """
    for i in range(3):
        pytest_runtest_logreport(_Report(f"tests/x/test_xfail.py::test_{i}", "setup", "passed"))
        pytest_runtest_logreport(
            _Report(f"tests/x/test_xfail.py::test_{i}", "call", "skipped", wasxfail="known")
        )

    assert not inert_test_files()


def test_a_file_where_everything_passes_is_not_flagged():
    """The trivial direction, which fails if the rule inverted."""
    for i in range(4):
        _passing_test("tests/x/test_healthy.py", f"test_{i}")

    assert not inert_test_files()
