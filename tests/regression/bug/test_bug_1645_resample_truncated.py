# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a truncated mock can be redrawn instead of kept or refused (#1645).

``make_population`` draws redshift and the SFH age parameters independently, so a
mock can request an SFH that does not fit inside cosmic time at its own redshift.
The first half of #1645 made that **visible** — the exact fraction rides on the
warning and is recorded per galaxy on :class:`MockPopulation` — and added
``max_truncated_fraction`` to **refuse** a population containing one.

Refusing is the right default for a fixture you did not generate. It is the wrong
tool when you are generating one: a mock builder that raises leaves the caller to
re-run with a different key until it gets lucky, which is rejection sampling done
by hand and without a record.

``resample_truncated=True`` does it in the loop and says so. The result is a draw
from the prior **conditioned on being physically realizable**, which is a
different distribution from the prior — hence opt-in, and hence
:attr:`MockPopulation.n_resampled` records how much conditioning happened.

Why not the default: with ``PRNGKey(42)`` and ``n_galaxies=4`` the existing
fixtures contain a galaxy at 69.2%. Redrawing by default would silently change
every mock in the repository, including the population behind the banked
ESS-vs-breadth curve in the handoff doc's §4j.

These tests drive the resampling loop through a stub rather than a real model, so
they run in the PR gate rather than the SSP-gated slow tier.
"""

from __future__ import annotations

import pytest

from tengri.analysis.population_mocks import _resample_until_realizable

pytestmark = pytest.mark.regression_bug


def _draws(sequence):
    """A draw function returning the given truncated fractions in order."""
    calls = {"n": 0}

    def draw(attempt):
        value = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        return value

    draw.calls = calls
    return draw


class TestItRedrawsOnlyWhatItMust:
    def test_an_acceptable_first_draw_is_kept_untouched(self):
        """No redraw means no conditioning — the common case must cost nothing."""
        draw = _draws([0.0])
        value, attempts = _resample_until_realizable(draw, max_fraction=0.1, max_attempts=8)
        assert (value, attempts) == (0.0, 1)
        assert draw.calls["n"] == 1

    def test_it_stops_at_the_first_acceptable_draw(self):
        """Three bad draws then a good one: four calls, not eight."""
        draw = _draws([0.69, 0.4, 0.2, 0.05, 0.0])
        value, attempts = _resample_until_realizable(draw, max_fraction=0.1, max_attempts=8)
        assert (value, attempts) == (0.05, 4)
        assert draw.calls["n"] == 4

    def test_the_threshold_is_inclusive(self):
        """A galaxy exactly at the limit is acceptable, matching the refuse path,
        which raises only on ``> max_truncated_fraction``."""
        draw = _draws([0.1])
        value, attempts = _resample_until_realizable(draw, max_fraction=0.1, max_attempts=4)
        assert (value, attempts) == (0.1, 1)


class TestItRefusesToLoopForever:
    def test_an_unsatisfiable_prior_raises_rather_than_spinning(self):
        """If every draw truncates, the prior and the threshold are incompatible.
        Silently looping would hang a mock builder; quietly keeping the last draw
        would defeat the point of asking."""
        draw = _draws([0.69])
        with pytest.raises(ValueError, match="max_attempts"):
            _resample_until_realizable(draw, max_fraction=0.1, max_attempts=5)
        assert draw.calls["n"] == 5

    def test_the_error_reports_what_it_actually_saw(self):
        """'Could not find a draw' is unactionable. The best fraction reached says
        whether the threshold is slightly too tight or wildly unreachable."""
        draw = _draws([0.69, 0.55, 0.51])
        with pytest.raises(ValueError) as excinfo:
            _resample_until_realizable(draw, max_fraction=0.1, max_attempts=3)
        message = str(excinfo.value)
        assert "0.51" in message
        assert "0.1" in message
