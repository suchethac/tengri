# SPDX-License-Identifier: BSD-3-Clause
"""A condition number is only as good as the chain that produced it.

This module exists to decide whether a dense mass matrix is worth its cost, and
its whole output is one number. That makes two failure modes expensive rather
than merely untidy: a number computed from a chain too short to estimate a
correlation matrix, and a number driven to infinity by a parameter that never
moved. Both produce a large, confident-looking figure that argues for the same
remedy as a genuine one.
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

from paper1.posterior_conditioning import MIN_ESS_PER_PARAM, conditioning

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(0)


def _write(tmp_path, columns, ess, name="post.npz"):
    """An npz shaped like a saved posterior: named columns plus ess_<name>."""
    payload = {"free_params": np.array(list(columns), dtype=object)}
    for key, values in columns.items():
        payload[key] = values
        payload[f"ess_{key}"] = np.array(float(ess))
    path = tmp_path / name
    np.savez(path, **payload)
    return path


def _independent(n, p):
    return {f"p{i}": RNG.normal(0.0, 1.0, n) for i in range(p)}


def test_an_uncorrelated_posterior_is_well_conditioned(tmp_path):
    """The control. A diagonal metric has nothing left to fix here."""
    cols = _independent(4000, 6)
    report = conditioning(_write(tmp_path, cols, ess=4000))

    assert report["condition_number"] < 5.0
    assert report["n_above_0.9"] == 0


def test_a_correlated_posterior_is_badly_conditioned(tmp_path):
    """What the mock looks like: a shared direction across every coordinate."""
    n = 4000
    shared = RNG.normal(0.0, 1.0, n)
    cols = {f"p{i}": shared + RNG.normal(0.0, 0.15, n) for i in range(6)}
    report = conditioning(_write(tmp_path, cols, ess=4000))

    assert report["condition_number"] > 50.0
    # The reported remedy scale is the square root, not the number itself.
    assert report["trajectory_factor"] == pytest.approx(np.sqrt(report["condition_number"]))


def test_too_few_effective_samples_per_parameter_is_refused(tmp_path):
    """The defect. A correlation matrix from a chain that did not mix.

    Twelve parameters at 3 effective samples each cannot support a 12-by-12
    correlation matrix, and the number that comes out is noise -- but it is a
    large number, so it argues for a dense metric exactly as a real one would.
    """
    cols = _independent(2000, 12)
    ess = 3.0 * 12  # 3 per parameter, under the floor

    with pytest.raises(SystemExit) as excinfo:
        conditioning(_write(tmp_path, cols, ess=ess))

    assert "per parameter" in str(excinfo.value)
    assert "noise wearing a condition number" in str(excinfo.value)


def test_at_the_floor_the_number_is_reported(tmp_path):
    """The floor must not refuse a chain that cleared it."""
    p = 8
    cols = _independent(2000, p)
    report = conditioning(_write(tmp_path, cols, ess=MIN_ESS_PER_PARAM * p))

    assert report["condition_number"] > 0
    assert report["n_param"] == p


def test_a_frozen_parameter_is_excluded_rather_than_made_infinite(tmp_path):
    """A column that never moved has no correlation, and keeping it would make
    the matrix singular -- reporting an enormous condition number for a reason
    that is not geometry and that no metric can fix."""
    cols = _independent(4000, 5)
    cols["pinned"] = np.full(4000, 2.5)
    report = conditioning(_write(tmp_path, cols, ess=4000))

    assert report["frozen"] == ["pinned"]
    assert report["n_param"] == 5
    assert np.isfinite(report["condition_number"])
    assert report["condition_number"] < 5.0
