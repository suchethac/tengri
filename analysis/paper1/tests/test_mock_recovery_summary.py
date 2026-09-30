# SPDX-License-Identifier: BSD-3-Clause
"""Recovery must be reportable in a way a wide posterior cannot pass.

Section 3 quotes how closely the joint fit recovers the mock's truth. Its own
source comment states the hazard: report the offset in sigma "so a wide
posterior cannot be read as a good recovery". Sigma alone does the opposite --
widen a posterior far enough and every truth falls inside a fraction of one,
which is why the width is reported beside it rather than divided away.

The other half is the pairing. ``stellar_mass`` is the SFH's time-integral and
``stellar_mass_surviving`` subtracts what the population returned, so one can
never exceed the other; on this mock's truth they differ by 0.195 dex, which is
larger than the offsets being measured. A script that paired draws with
properties wrongly would show that constraint violated, and that is the cheapest
signal it is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.derive_mock_properties import _properties, _thin, summarize

pytestmark = pytest.mark.unit


class _Model:
    """Publishes exactly the properties it is given."""

    def __init__(self, published):
        self._published = published

    def predict_properties(self, params, names):
        return {k: v for k, v in self._published.items() if k in names}


def test_a_sharp_posterior_centered_on_truth_recovers_it():
    draws = np.random.default_rng(0).normal(10.0, 0.01, 4000)
    row = summarize(10.0, draws)
    assert abs(row["offset"]) < 0.005
    assert abs(row["offset_in_sigma"]) < 0.5
    assert row["width"] < 0.02


def test_a_wide_posterior_reports_its_width_beside_a_small_sigma_offset():
    """The hazard, stated as a number.

    Two posteriors, both centered 0.5 dex off the truth. The wide one lands
    within a third of a sigma and would read as a fine recovery from that
    figure alone; its width is 1.5 dex, which is the fact that stops it.
    """
    rng = np.random.default_rng(1)
    sharp = summarize(10.0, rng.normal(10.5, 0.05, 4000))
    wide = summarize(10.0, rng.normal(10.5, 1.5, 4000))

    assert abs(sharp["offset_in_sigma"]) > 5, "a sharp posterior 0.5 dex off must look bad"
    assert abs(wide["offset_in_sigma"]) < 1.0, "fixture: the wide one must look fine in sigma"

    # Both are 0.5 dex off, and only the width distinguishes them.
    assert abs(sharp["offset"] - wide["offset"]) < 0.1
    assert wide["width"] > 10 * sharp["width"]


def test_a_degenerate_posterior_reports_no_sigma_rather_than_infinity():
    """Every draw identical. "Recovered to inf sigma" is not a measurement."""
    row = summarize(10.0, np.full(100, 10.25))
    assert row["offset_in_sigma"] is None
    assert row["width"] == 0.0
    assert row["offset"] == pytest.approx(0.25)


def test_thinning_spans_the_whole_chain():
    """Taking the first N draws would report the warmup's tail, not the chain."""
    index = _thin(4000, 400)
    assert len(index) == 400
    assert index[0] == 0
    assert index[-1] == 3999
    assert np.all(np.diff(index) > 0)


def test_thinning_keeps_every_draw_when_the_chain_is_short():
    index = _thin(120, 400)
    assert np.array_equal(index, np.arange(120))


def test_thinning_is_deterministic():
    assert np.array_equal(_thin(4000, 400), _thin(4000, 400))


def test_a_missing_property_refuses_rather_than_defaulting():
    """The backend sweep's defect, not repeated here."""
    model = _Model({"stellar_mass": 1e10})
    with pytest.raises(SystemExit) as excinfo:
        _properties(model, {}, ("stellar_mass", "sfr_100myr"))
    message = str(excinfo.value)
    assert "sfr_100myr" in message
    assert "stellar_mass" in message, "the refusal should say what was available"


def test_published_properties_are_returned_as_floats():
    model = _Model({"stellar_mass": 1e10, "sfr_100myr": 0.0})
    got = _properties(model, {}, ("stellar_mass", "sfr_100myr"))
    assert got == {"stellar_mass": 1e10, "sfr_100myr": 0.0}
    # A quiescent galaxy publishes SFR 0.0; a truthiness check would drop it.
    assert isinstance(got["sfr_100myr"], float)
