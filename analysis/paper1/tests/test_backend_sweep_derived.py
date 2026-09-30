# SPDX-License-Identifier: BSD-3-Clause
"""A missing derived property must refuse, not resolve to a plausible galaxy.

``run_backend_sweep.py`` read its two headline quantities as
``props.get("stellar_mass", 1e10)`` and ``props.get("sfr_100myr", 1.0)``. Both
defaults are the dangerous kind. They do not reach the figure as a sentinel, a
NaN or a zero flux; they reach it as ``log M* = 10.0`` and ``log SFR = 0.0``
exactly, which is an unremarkable star-forming galaxy. A default that is
indistinguishable from a measurement cannot be caught on the plot, in the npz,
or by a reader.

Neither has ever fired: the five committed backend rows carry log M* between
10.55 and 10.64 and log SFR between 1.36 and 1.47. That is the argument for
removing them now, while the data is clean, rather than after a model stops
publishing one of the two and a backend quietly lands on the round number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
# ``run_backend_sweep`` is a package module (relative imports) but reaches
# ``configs.py``, which imports ``config_metadata`` by bare name -- so both
# directories have to be importable.
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.run_backend_sweep import _derived

pytestmark = pytest.mark.unit


def test_a_published_property_is_returned():
    assert _derived({"stellar_mass": 1.5e10, "sfr_100myr": 12.0}, "stellar_mass") == 1.5e10


def test_a_missing_mass_refuses_rather_than_returning_the_round_number():
    """The defect. ``1e10`` is log 10.0 exactly -- a normal galaxy."""
    with pytest.raises(SystemExit) as excinfo:
        _derived({"sfr_100myr": 12.0}, "stellar_mass")
    assert "stellar_mass" in str(excinfo.value)


def test_a_missing_sfr_refuses_rather_than_returning_the_round_number():
    """``1.0`` is log 0.0 exactly -- also a normal galaxy."""
    with pytest.raises(SystemExit) as excinfo:
        _derived({"stellar_mass": 1.5e10}, "sfr_100myr")
    assert "sfr_100myr" in str(excinfo.value)


def test_the_refusal_names_what_was_available():
    """So the operator can see whether the property was renamed or dropped."""
    with pytest.raises(SystemExit) as excinfo:
        _derived({"stellar_mass_surviving": 8e9}, "sfr_100myr")
    message = str(excinfo.value)
    assert "stellar_mass_surviving" in message, (
        f"the refusal does not say what the model did publish: {message}"
    )


def test_an_empty_property_set_still_refuses_by_name():
    """The shape a model that published nothing at all would take."""
    with pytest.raises(SystemExit) as excinfo:
        _derived({}, "stellar_mass")
    assert "stellar_mass" in str(excinfo.value)


def test_a_falsy_but_real_value_is_returned_not_treated_as_absent():
    """Zero SFR is a measurement, and a presence check must not confuse the two.

    A quiescent galaxy can legitimately publish ``sfr_100myr = 0.0``. Written
    as a truthiness test rather than a membership test, the refusal would fire
    on exactly the galaxies the paper most wants to report.
    """
    assert _derived({"sfr_100myr": 0.0}, "sfr_100myr") == 0.0
