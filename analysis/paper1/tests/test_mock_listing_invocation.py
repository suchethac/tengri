# SPDX-License-Identifier: BSD-3-Clause
"""The paper prints a listing; something has to execute its last three lines.

``verify_mock_listing`` is named for the listing and checks that the model
block builds and predicts on both channels. It never touched what comes after
the model: the SSP name, the precompute spelling, and the fit call. Those three
lines drifted, and because the verifier re-implements the model block rather
than reading the printed one, every drift passed:

* ``load_ssp("fsps_mist_c3k")`` -- not a grid on disk; ``FileNotFoundError``.
* ``approx=(WavePrecomp(), SpectrumPrecomp())`` -- the figure was produced with
  ``WavePrecomp()`` alone, and ``approx="auto"`` tops a build-time config up
  with ``FeaturePrecomp`` only, never ``SpectrumPrecomp``, so the printed pair
  had never run.
* ``Fitter(forward).run(data, method="nuts", seed=0)`` -- three defects in one
  line: ``run``'s first positional is ``method``, so ``data`` there collides
  with the keyword; ``"nuts"`` is not a method name; and ``run`` takes ``key``,
  not ``seed``.

A reader copying that block got an exception on the first and the last of
them. These tests bind the documented call rather than running a fit, so they
cost milliseconds and still observe the thing that broke.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

pytestmark = pytest.mark.unit


def test_the_method_name_the_listing_prints_resolves():
    """The defect: "nuts" is not a name, and nothing executed it to find out."""
    from paper1.verify_mock_listing import FIT_METHOD

    from tengri.inference.fitter import resolve_method

    assert resolve_method(FIT_METHOD) == FIT_METHOD


def test_the_name_that_shipped_in_the_listing_is_still_rejected():
    """Guards the guard: if "nuts" ever became valid this test is vacuous."""
    from tengri.config.exceptions import ParameterError
    from tengri.inference.fitter import resolve_method

    with pytest.raises(ParameterError):
        resolve_method("nuts")


def test_the_documented_fit_call_binds():
    """``forward.fit(data, method=..., key=...)`` -- the canonical surface."""
    from paper1.verify_mock_listing import FIT_METHOD

    from tengri.forward.forward_model import ForwardModel

    sig = inspect.signature(ForwardModel.fit)
    bound = sig.bind(None, "DATA", method=FIT_METHOD, key="KEY")
    assert bound.arguments["data"] == "DATA"
    assert bound.arguments["method"] == FIT_METHOD


def test_the_call_shape_that_shipped_does_not_bind():
    """The exact line the paper carried, kept as the thing being excluded.

    Without this the test above passes on a signature that would also have
    accepted the broken call, and the file would prove nothing about it.
    """
    from tengri.inference.fitter import Fitter

    sig = inspect.signature(Fitter.run)
    with pytest.raises(TypeError, match="multiple values"):
        sig.bind(None, "DATA", method="nuts", seed=0)


def test_run_takes_key_not_seed():
    """``seed=0`` was swallowed by ``**kwargs`` and handed to the backend."""
    from tengri.forward.forward_model import ForwardModel

    params = inspect.signature(ForwardModel.fit).parameters
    assert "key" in params
    assert "seed" not in params


def test_the_ssp_name_is_the_full_spelling():
    """The short name is not a file; ``load_ssp`` raises on it."""
    from paper1.verify_mock_listing import SSP_NAME

    assert SSP_NAME == "fsps_mist_c3k_a_chabrier"
    assert SSP_NAME != "fsps_mist_c3k", "the short spelling is not a grid on disk"


def test_the_verifier_itself_refuses_a_bad_method_name(monkeypatch):
    """Observe the guard, not only the facts it checks.

    Every test above re-derives what ``verify_fit_invocation`` asserts, so all
    of them stay green if the verifier stops asserting -- which is exactly how
    a guard goes quiet without anyone noticing. This one fails if it does.
    """
    from paper1 import verify_mock_listing as V

    from tengri.config.exceptions import ParameterError

    monkeypatch.setattr(V, "FIT_METHOD", "nuts")
    # Either spelling of the refusal counts: resolve_method raises before the
    # assert can compare, and a future guard might assert first instead.
    with pytest.raises((ParameterError, AssertionError)):
        V.verify_fit_invocation()


def test_main_actually_calls_the_verifier():
    """A verifier nothing invokes is not a check.

    Asserted against the source because reaching the real call site means
    loading the SSP grid and building the kitchen-sink model, minutes of work
    for a one-line fact.
    """
    source = (PAPER1 / "verify_mock_listing.py").read_text()
    body = source[source.index("def main(") :]
    assert "verify_fit_invocation()" in body, (
        "verify_mock_listing.main() never runs the fit-invocation check, so the "
        "listing's last line is unverified again"
    )
