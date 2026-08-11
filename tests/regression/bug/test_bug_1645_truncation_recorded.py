# SPDX-License-Identifier: BSD-3-Clause
"""Regression: pre-Big-Bang truncation must be recorded, not only warned (#1645).

``make_population`` samples redshift and the SFH age parameters independently, so
a mock can request an SFH that does not fit inside the age of the universe at its
own redshift. The mass is then truncated and the prediction stops representing
the requested SFH.

Measured on the ESS sweep's own fixed-key population
(``PRNGKey(42)``, ``n_galaxies=4``): **all four** galaxies truncate — 3%, 5%, 9%
and **69%** at z = 4.05, 10.09, 3.69 and 10.67. The 69% galaxy is not a usable
fixture, and nothing downstream could tell.

The only signal was ``SFHBeforeBigBangWarning`` (#683) on stderr. Two properties
made that unusable as data:

* the exact fraction is computed at the raise site and then **discarded** — the
  message keeps ``{frac:.0%}``, so "69%" could be anything in 0.685-0.695;
* recovering it meant regex-parsing prose, which breaks the moment the sentence
  is reworded.

So the exact value now rides on the warning instance, and ``MockPopulation``
records it per galaxy — following the ``n_halpha_absorption`` precedent already
in that class: detect, report, never silently drop.

Note the raise site's own threshold: ``if frac_pre_bb > 0.01``. A galaxy losing
1% or less is truncated with **no warning at all**, so a recorded 0.0 means "at
most 1%", not "exactly none".
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri.analysis.population_mocks import MockPopulation, _max_truncated_fraction
from tengri.components.stellar.component import SFHBeforeBigBangWarning

pytestmark = pytest.mark.regression_bug


def _warn(fraction=None, message="truncated"):
    """A captured-warning stand-in carrying an optional exact fraction."""
    w = SFHBeforeBigBangWarning(message)
    if fraction is not None:
        w.truncated_fraction = fraction
    return warnings.WarningMessage(w, SFHBeforeBigBangWarning, __file__, 0)


class TestTheWarningCarriesTheExactValue:
    def test_the_class_defaults_to_none(self):
        """A bare instance must not pretend to know a fraction it was not given."""
        assert SFHBeforeBigBangWarning("x").truncated_fraction is None

    def test_the_value_is_exact_not_the_rounded_percent(self):
        """The message rounds to whole percent; the attribute must not. This is
        the whole reason for carrying it structurally."""
        w = _warn(0.6873)
        assert _max_truncated_fraction([w]) == 0.6873


class TestTheReducer:
    def test_no_warnings_means_zero(self):
        assert _max_truncated_fraction([]) == 0.0

    def test_it_takes_the_worst_not_the_first_or_the_sum(self):
        """A galaxy predicted more than once must report its worst truncation,
        and fractions must never accumulate past 1.0."""
        assert _max_truncated_fraction([_warn(0.05), _warn(0.69), _warn(0.09)]) == 0.69

    def test_a_warning_without_the_attribute_does_not_crash(self):
        """Warnings of this category may arrive from a path that predates the
        attribute. Degrade to 0.0 rather than raising inside a mock builder."""
        assert _max_truncated_fraction([_warn(None)]) == 0.0

    def test_unrelated_warnings_are_ignored(self):
        other = warnings.WarningMessage(UserWarning("unrelated"), UserWarning, __file__, 0)
        assert _max_truncated_fraction([other, _warn(0.42)]) == 0.42


class TestMockPopulationRecordsIt:
    def test_the_field_exists_and_defaults_safely(self):
        """Older callers construct MockPopulation positionally; the new field
        must be optional and must not shift any existing argument."""
        pop = MockPopulation(table=np.zeros(1), truth_params=[{}], n_halpha_absorption=0)
        assert pop.truncated_fraction is None

    def test_it_holds_one_value_per_galaxy(self):
        pop = MockPopulation(
            table=np.zeros(3),
            truth_params=[{}, {}, {}],
            n_halpha_absorption=0,
            truncated_fraction=np.array([0.03, 0.05, 0.69]),
        )
        assert pop.truncated_fraction.shape == (3,)
        assert float(np.max(pop.truncated_fraction)) == 0.69
