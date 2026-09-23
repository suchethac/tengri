# SPDX-License-Identifier: BSD-3-Clause
"""A divergence count does not say which geometry produced it.

The two shapes have different remedies, so calling one the other sends the
next run at the wrong knob. A funnel concentrates divergent draws in a single
coordinate's neck and is fixed by that parameter; curvature spreads them
across coordinates, is invisible to a diagonal mass matrix, and is fixed by
the metric or a shorter step. Reading "79 divergences" alone cannot tell them
apart, and the refused 11 h mock run is diffuse.

The third case matters as much as the two: zero divergences means there is
nothing to locate, which is not the same as a diffuse result and must not be
reported as one.
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

from paper1.divergence_geometry import (
    FUNNEL_SIGMA,
    MIN_DIVERGENT_FOR_SHAPE,
    offsets,
    verdict,
)

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(0)
NAMES = ["a", "b", "c"]


def _posterior(divergent: dict[str, np.ndarray], bulk_sd=1.0, n=2000):
    post = {}
    for name in NAMES:
        post[name] = RNG.normal(0.0, bulk_sd, n)
    for name, values in divergent.items():
        post[f"divergent_{name}"] = values
    return post


def test_a_funnel_names_the_parameter_it_is_in():
    """Divergent draws four sigma out in one coordinate, ordinary in the rest."""
    post = _posterior(
        {
            "a": RNG.normal(4.0, 0.1, 50),
            "b": RNG.normal(0.0, 1.0, 50),
            "c": RNG.normal(0.0, 1.0, 50),
        }
    )
    rows = offsets(post, NAMES)
    shape = verdict(rows, n_divergent=50)

    assert shape["shape"] == "funnel"
    assert shape["parameter"] == "a"
    assert abs(shape["offset_sd"]) >= FUNNEL_SIGMA
    assert "not the step size" in shape["note"]


def test_diffuse_divergences_refuse_to_name_a_parameter():
    """The refused mock run's shape. No coordinate is singled out."""
    post = _posterior(
        {
            "a": RNG.normal(0.5, 1.0, 79),
            "b": RNG.normal(-0.4, 1.0, 79),
            "c": RNG.normal(0.3, 1.0, 79),
        }
    )
    rows = offsets(post, NAMES)
    shape = verdict(rows, n_divergent=79)

    assert shape["shape"] == "diffuse"
    assert abs(shape["offset_sd"]) < FUNNEL_SIGMA
    assert "diagonal mass matrix" in shape["note"]


def test_no_divergences_is_absent_not_diffuse():
    """The distinction this file exists for.

    A clean fit and a fit whose geometry is spread out both produce a flat
    ranking. Calling the clean one "diffuse" would report a geometry problem
    that is not there.
    """
    shape = verdict([], n_divergent=0)

    assert shape["shape"] is None
    assert "absent one" in shape["note"]


def test_the_ranking_is_ordered_by_absolute_offset():
    """A large negative offset is as much a funnel as a large positive one."""
    post = _posterior(
        {
            "a": RNG.normal(0.2, 1.0, 40),
            "b": RNG.normal(-3.5, 0.1, 40),
            "c": RNG.normal(0.1, 1.0, 40),
        }
    )
    rows = offsets(post, NAMES)

    assert rows[0]["parameter"] == "b"
    assert rows[0]["offset_sd"] < 0


def test_a_frozen_parameter_reports_no_scale_rather_than_an_infinite_offset():
    """Dividing by a zero spread would manufacture a funnel out of a constant."""
    post = {
        "a": np.full(500, 2.0),
        "divergent_a": np.full(20, 2.5),
    }
    rows = offsets(post, ["a"])

    assert rows[0]["offset_sd"] is None
    assert rows[0]["sd"] == 0.0


def test_a_parameter_with_no_divergent_column_is_skipped():
    """Not every array in the file is a sampled parameter."""
    post = _posterior({"a": RNG.normal(0.0, 1.0, 30)})
    rows = offsets(post, NAMES)

    assert [r["parameter"] for r in rows] == ["a"]


def test_a_ranking_of_only_frozen_parameters_declines_a_verdict():
    """Nothing measurable is not the same as nothing wrong.

    The count here is above :data:`MIN_DIVERGENT_FOR_SHAPE` deliberately. This
    test is about a parameter with no spread to measure an offset against, and
    an earlier version used five divergent draws, which was incidental to what
    it checks and would now be declined for the count instead -- passing for
    the wrong reason and never touching the branch it names.
    """
    n = MIN_DIVERGENT_FOR_SHAPE * 2
    post = {"a": np.full(100, 1.0), "divergent_a": np.full(n, 1.0)}
    rows = offsets(post, ["a"])
    shape = verdict(rows, n_divergent=n)

    assert shape["shape"] is None
    assert "spread" in shape["note"]


# ---------------------------------------------------------------------------
# A handful of divergences cannot support a shape.
#
# The ta095 mock fit ended with two divergent draws. Two is a median of two
# numbers, and the module reports its offset in units of the parameter's own
# standard deviation, where the median's uncertainty is 1.253/sqrt(n) -- 0.89 sd
# at n=2, against a funnel threshold of 2.0. Without a floor the module would
# have answered "funnel" or "diffuse" with the same confidence it uses for a
# run with hundreds, and a weak verdict reads exactly like a strong one once it
# is quoted.


def test_two_divergences_are_unmeasurable_rather_than_diffuse():
    """The defect. A shape read off a median of two draws."""
    post = _posterior(
        {
            "a": RNG.normal(4.0, 0.1, 2),
            "b": RNG.normal(0.0, 1.0, 2),
            "c": RNG.normal(0.0, 1.0, 2),
        }
    )
    rows = offsets(post, NAMES)

    shape = verdict(rows, n_divergent=2)

    assert shape["shape"] is None
    assert "too few" in shape["note"]
    # The offsets themselves would have said "funnel" -- that is the point.
    assert abs(rows[0]["offset_sd"]) >= FUNNEL_SIGMA


def test_the_too_few_refusal_is_not_the_no_divergences_refusal():
    """Both return None and they mean different things.

    Zero divergences is a clean fit with nothing to locate. Two is a fit whose
    geometry is simply not measurable from what it produced. Collapsing them
    would let "no verdict" be read as "no problem".
    """
    none_at_all = verdict([], n_divergent=0)
    too_few = verdict(offsets(_posterior({"a": RNG.normal(0.0, 1.0, 3)}), NAMES), n_divergent=3)

    assert none_at_all["shape"] is too_few["shape"] is None
    assert "absent one" in none_at_all["note"]
    assert "unmeasurable one" in too_few["note"]
    assert none_at_all["note"] != too_few["note"]


def test_at_the_floor_a_verdict_is_given_again():
    """The floor must not be a permanent refusal on a run that clears it."""
    n = MIN_DIVERGENT_FOR_SHAPE
    post = _posterior(
        {
            "a": RNG.normal(4.0, 0.1, n),
            "b": RNG.normal(0.0, 1.0, n),
            "c": RNG.normal(0.0, 1.0, n),
        }
    )

    shape = verdict(offsets(post, NAMES), n_divergent=n)

    assert shape["shape"] == "funnel"
    assert shape["parameter"] == "a"
