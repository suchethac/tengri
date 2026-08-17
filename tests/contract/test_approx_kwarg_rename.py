# SPDX-License-Identifier: BSD-3-Clause
"""The call-time precompute flag is spelled ``approx=``, matching build time.

Until 2026-08 the runtime flag was ``fast=`` while the object it selects is
installed as ``SEDModel.build(..., approx=WavePrecomp(...))`` — one mechanism
under two names. ``fast`` also named the benefit and hid the cost: its own
docstring claimed exact-vs-LUT agreement to ``rtol=1e-12``, where the measured
figure is 8.5e-4 at z = 0.05 and 1.4e-2 for sdss_g at z = 3
(``bench/reports/2026-08-17_wave_precomp_accuracy.md``).

These tests pin the rename and its deprecation window (NAMING_CONTRACT §4b.4).
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from tengri._deprecated import UNSET, resolve_renamed_flag
from tengri.forward.prediction import Prediction
from tengri.forward.sed_model import SEDModel
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


# Every surface that takes the call-time precompute flag, and the class it
# lives on. Adding a surface without adding it here is the failure mode this
# list exists to catch.
FLAG_SURFACES = [
    (Prediction, "photometry"),
    (Prediction, "magnitudes"),
    (Prediction, "spectrum"),
    (Posterior, "observables"),
    (Posterior, "spectra"),
    (SEDModel, "predict_spectral_indices"),
    (SEDModel, "measure_line_fluxes"),
]


@pytest.mark.parametrize(
    ("cls", "method"), FLAG_SURFACES, ids=[f"{c.__name__}.{m}" for c, m in FLAG_SURFACES]
)
def test_surface_takes_approx_and_still_accepts_fast(cls, method):
    """Both spellings are present, and ``approx`` defaults to exact."""
    sig = inspect.signature(getattr(cls, method))
    assert "approx" in sig.parameters, f"{cls.__name__}.{method} lost `approx=`"
    assert "fast" in sig.parameters, f"{cls.__name__}.{method} dropped `fast=` too early"
    assert sig.parameters["approx"].default is False, "the default must be the exact path"
    assert sig.parameters["fast"].default is UNSET, (
        "`fast` must default to the sentinel, not to False -- otherwise "
        "'not passed' and 'passed False' are indistinguishable and the "
        "contradiction check below cannot work"
    )


@pytest.mark.parametrize(
    ("cls", "method"), FLAG_SURFACES, ids=[f"{c.__name__}.{m}" for c, m in FLAG_SURFACES]
)
def test_fast_is_keyword_only(cls, method):
    """``fast`` cannot be passed positionally into ``approx``'s slot.

    ``photometry(filters, fast)`` was a legal positional call before the
    rename. Leaving ``fast`` positional would let an old positional call bind
    silently to the new parameter, which is the quiet half of a rename.
    """
    sig = inspect.signature(getattr(cls, method))
    assert sig.parameters["fast"].kind is inspect.Parameter.KEYWORD_ONLY


class TestResolveRenamedFlag:
    """The shim itself, without needing an SSP grid or a built model."""

    def test_absent_old_returns_new(self):
        assert (
            resolve_renamed_flag(True, UNSET, old_name="fast", new_name="approx", caller="f")
            is True
        )

    def test_absent_old_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            resolve_renamed_flag(False, UNSET, old_name="fast", new_name="approx", caller="f")

    def test_old_spelling_warns_and_is_honored(self):
        with pytest.warns(DeprecationWarning, match=r"f\(fast=\.\.\.\).*approx"):
            got = resolve_renamed_flag(False, True, old_name="fast", new_name="approx", caller="f")
        assert got is True

    def test_old_spelling_false_is_honored(self):
        """``fast=False`` must resolve to False, not to the default-by-accident."""
        with pytest.warns(DeprecationWarning):
            got = resolve_renamed_flag(
                False, False, old_name="fast", new_name="approx", caller="f"
            )
        assert got is False

    def test_contradiction_raises(self):
        with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="contradict"):
            resolve_renamed_flag(True, False, old_name="fast", new_name="approx", caller="f")

    def test_agreement_does_not_raise(self):
        with pytest.warns(DeprecationWarning):
            got = resolve_renamed_flag(True, True, old_name="fast", new_name="approx", caller="f")
        assert got is True

    def test_message_names_the_caller(self):
        """A warning that does not say which call to fix is a worse warning."""
        with pytest.warns(DeprecationWarning, match="Prediction.photometry"):
            resolve_renamed_flag(
                False,
                True,
                old_name="fast",
                new_name="approx",
                caller="Prediction.photometry",
            )


def test_legacy_tutorial_name_still_resolves():
    """``tutorial('fast_vs_exact')`` warns and shows the renamed topic."""
    import tengri

    with pytest.warns(DeprecationWarning, match="approx_vs_exact"):
        tengri.tutorial("fast_vs_exact")


def test_no_src_reference_left_on_the_old_spelling():
    """tengri itself must neither call nor teach ``fast=``.

    Two failure modes, one check. An internal *caller* left on the old
    spelling emits a DeprecationWarning the user cannot act on, from a call
    they did not write. A stale *docstring* is worse in a quieter way: it
    teaches the removed name with nothing to warn about. This guard found one
    of each when it was written.

    A line naming *both* spellings is documenting the rename itself and is
    exempt — that is the one legitimate reason for `fast=` to survive here.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "tengri"
    offenders = []
    for path in src.rglob("*.py"):
        if path.name == "_deprecated.py":
            continue  # defines the shim; names both spellings by construction
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "fast=UNSET" in line or "old_name=" in line:
                continue  # the shim's own parameter declarations
            if not re.search(r"\bfast=(?!UNSET)", line):
                continue
            if "approx=" in line:
                continue  # documents the rename, e.g. "`fast=` became `approx=`"
            offenders.append(f"{path.relative_to(src)}:{i}: {line.strip()}")
    assert not offenders, "src/ still references the deprecated `fast=` spelling:\n" + "\n".join(
        offenders
    )
