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

from paper1.divergence_geometry import FUNNEL_SIGMA, offsets, verdict

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
    """Nothing measurable is not the same as nothing wrong."""
    post = {"a": np.full(100, 1.0), "divergent_a": np.full(5, 1.0)}
    rows = offsets(post, ["a"])
    shape = verdict(rows, n_divergent=5)

    assert shape["shape"] is None
    assert "spread" in shape["note"]
