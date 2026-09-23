# SPDX-License-Identifier: BSD-3-Clause
"""``verify_mock_precompute.py`` must not report a channel it never approximated.

The guard compares the mock as fit (``approx=WavePrecomp()``) against the same
model forced to ``approx=None``, and reports the difference in units of the
measurement error. It reported two things: a per-band photometric bias, and a
spectrum bias of ``0.000`` sigma.

The spectrum number was zero for a structural reason. ``WavePrecomp`` is the
*photometric* LUT; the spectrum channel is approximated only by
``SpectrumPrecomp``, which this mock does not pass. Both arms therefore
evaluated the spectrum on the exact path, and their difference could not have
been anything but zero. The report did not say so, and ``chi2_added`` summed
the identically-zero term alongside a measured one, so neither the line nor the
aggregate could reveal that the spectrum had never been checked.

A zero that cannot come out otherwise is not evidence. These pin the
distinction:

1. with photometry-only approximation the spectrum bias is ``None``, not a
   number, *even when the two spectra differ* -- the guard must decline to
   report rather than report a difference it has no standing to interpret;
2. with both LUTs present it is measured;
3. a model carrying no photometric LUT is refused outright, because then the
   band table would be structurally zero too and the guard would pass having
   measured nothing at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYSIS_DIR))

from paper1.verify_mock_precompute import (
    approximated_channels,
    spectrum_bias,
)

from tengri import FeaturePrecomp, SpectrumPrecomp, WavePrecomp

pytestmark = pytest.mark.unit


def test_a_photometric_lut_alone_does_not_cover_the_spectrum():
    """The mock's own configuration. It approximates one channel, not two."""
    assert approximated_channels(WavePrecomp()) == {"photometry"}


def test_both_luts_cover_both_channels():
    """The pairing the SpectrumPrecomp docstring recommends for a joint fit."""
    assert approximated_channels((WavePrecomp(), SpectrumPrecomp())) == {
        "photometry",
        "spectrum",
    }


def test_a_spectrum_lut_alone_does_not_cover_the_photometry():
    assert approximated_channels(SpectrumPrecomp()) == {"spectrum"}


def test_no_approximation_covers_nothing():
    assert approximated_channels(None) == set()


def test_an_unrelated_precompute_claims_neither_channel():
    """``FeaturePrecomp`` accelerates a component, not an observation channel.

    If it were counted, a model passing it would look like it had a LUT on a
    channel it does not, which is the same error in the other direction.
    """
    assert approximated_channels(FeaturePrecomp()) == set()


def test_a_spectrum_difference_is_not_reported_when_the_spectrum_is_not_approximated():
    """The defect, in its sharpest form.

    The two spectra here differ by 5 sigma. With photometry-only approximation
    that difference cannot have come from the LUT, so the guard must return
    ``None`` rather than dress it up as a measured bias. Returning ``0.0`` --
    or returning the 5.0 -- both claim knowledge the comparison does not have.
    """
    s_lut = np.array([1.0, 2.0, 3.0])
    s_exact = np.array([1.0, 2.0, 8.0])
    spec_sig = np.array([1.0, 1.0, 1.0])

    worst, chi2, note = spectrum_bias({"photometry"}, s_lut, s_exact, spec_sig)

    assert worst is None, f"reported {worst} for a channel that carries no LUT"
    assert chi2 == 0.0
    assert "not approximated" in note


def test_a_spectrum_difference_is_reported_when_the_spectrum_is_approximated():
    """The same arrays, now with standing to interpret them."""
    s_lut = np.array([1.0, 2.0, 3.0])
    s_exact = np.array([1.0, 2.0, 8.0])
    spec_sig = np.array([1.0, 1.0, 1.0])

    worst, chi2, note = spectrum_bias({"photometry", "spectrum"}, s_lut, s_exact, spec_sig)

    assert worst == pytest.approx(5.0)
    assert chi2 == pytest.approx(25.0)
    assert "max |err|/sigma" in note


def test_the_committed_sidecar_does_not_carry_a_structural_zero():
    """The recorded result must not say ``0.0`` where it means "not measured".

    This is the artifact the paper would quote. A ``null`` says the spectrum
    was not approximated; a ``0.0`` says it was approximated and came out
    perfect, and only one of those is true.
    """
    payload = ANALYSIS_DIR / "paper1" / "results" / "mock_precompute_bias.json"
    if not payload.exists():
        pytest.skip(f"no recorded result at {payload}")

    import json

    recorded = json.loads(payload.read_text())
    channels = set(recorded.get("approximated_channels", []))
    assert channels, (
        "the recorded result does not say which channels were approximated, so "
        "its spectrum entry cannot be read either way"
    )
    if "spectrum" not in channels:
        assert recorded["spectrum_max_sigma"] is None, (
            f"recorded spectrum_max_sigma is {recorded['spectrum_max_sigma']!r} "
            f"but the approximated channels are {sorted(channels)}; a spectrum "
            "that was never approximated cannot have a measured bias"
        )
