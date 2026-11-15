# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for tengri.inference.backends._psis.

The Pareto-smoothed importance sampling utilities.

Surface protected: ``fit_generalized_pareto`` (shape and scale recovery on
known generalized Pareto samples), ``psis`` (proposal-equals-target
degeneracy, well-behaved and ill-behaved importance sampling diagnostics,
tail truncation, input validation), and ``pareto_k_threshold``.

Expected values come from ``scipy.stats`` samples with known parameters and
from the closed-form normalizing constants of Gaussian densities, never from
re-typed copies of the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from tengri.inference.backends._psis import (
    fit_generalized_pareto,
    pareto_k_threshold,
    psis,
)

pytestmark = pytest.mark.contract

_SEEDS = (0, 1, 2)
# Bad-IS proposal scale. The brief's N(0, 0.5**2) gives k-hat of 0.58 to
# 0.74 at n=4000, which straddles the 0.7 cut; 0.25 gives 0.72 to 0.92.
_BAD_PROPOSAL_SCALE = 0.25
# Slack for the largest-weight check: normalization of the smoothed weights
# can lift the largest normalized weight slightly when tail weights shrink.
_NORMALIZATION_SLACK = 0.05


def _gaussian_log_ratio(x: np.ndarray, proposal_scale: float) -> np.ndarray:
    """Log target over proposal density for a N(0, 1) target."""
    return stats.norm.logpdf(x) - stats.norm.logpdf(x, scale=proposal_scale)


@pytest.mark.parametrize("seed", _SEEDS)
def test_fit_recovers_shape_and_scale_for_half_tail(seed: int) -> None:
    samples = stats.genpareto.rvs(
        c=0.5, scale=2.0, size=20000, random_state=np.random.default_rng(seed)
    )
    k, sigma = fit_generalized_pareto(samples)
    assert abs(k - 0.5) < 0.05
    assert abs(sigma - 2.0) < 0.15


@pytest.mark.parametrize("seed", _SEEDS)
def test_fit_recovers_shape_for_light_tail(seed: int) -> None:
    samples = stats.genpareto.rvs(
        c=0.1, scale=1.0, size=20000, random_state=np.random.default_rng(seed)
    )
    k, _ = fit_generalized_pareto(samples)
    assert abs(k - 0.1) < 0.05


def test_proposal_equals_target_is_degenerate() -> None:
    n = 1000
    result = psis(np.full(n, 3.7))
    np.testing.assert_allclose(result.log_weights, np.full(n, -math.log(n)), atol=1e-12)
    assert result.ess == pytest.approx(n, abs=1e-9)
    assert result.log_mean_raw_weight == pytest.approx(3.7, abs=1e-12)
    assert math.isnan(result.pareto_k)


@pytest.mark.parametrize("seed", _SEEDS)
def test_wider_proposal_is_well_behaved(seed: int) -> None:
    x = np.random.default_rng(seed).normal(0.0, 1.5, 4000)
    result = psis(_gaussian_log_ratio(x, 1.5))
    assert result.pareto_k < 0.5
    # Both densities are normalized, so log Z = 0.
    assert abs(result.log_mean_raw_weight) < 0.05


@pytest.mark.parametrize("seed", _SEEDS)
def test_narrow_proposal_is_flagged_unreliable(seed: int) -> None:
    x = np.random.default_rng(seed).normal(0.0, _BAD_PROPOSAL_SCALE, 4000)
    result = psis(_gaussian_log_ratio(x, _BAD_PROPOSAL_SCALE))
    assert result.pareto_k > 0.7


@pytest.mark.parametrize("seed", _SEEDS)
def test_smoothing_does_not_raise_largest_weight(seed: int) -> None:
    x = np.random.default_rng(seed).normal(0.0, _BAD_PROPOSAL_SCALE, 4000)
    log_ratios = _gaussian_log_ratio(x, _BAD_PROPOSAL_SCALE)
    raw_normalized = log_ratios - logsumexp(log_ratios)
    result = psis(log_ratios)
    assert np.max(result.log_weights) <= np.max(raw_normalized) + _NORMALIZATION_SLACK


def test_pareto_k_threshold_values() -> None:
    assert pareto_k_threshold(1000) == pytest.approx(min(1 - 1 / 3, 0.7), abs=1e-12)
    assert pareto_k_threshold(1000) == pytest.approx(0.6667, abs=1e-4)
    assert pareto_k_threshold(10**8) == 0.7


def test_nan_entry_raises() -> None:
    log_ratios = np.zeros(20)
    log_ratios[3] = np.nan
    with pytest.raises(ValueError, match="1 NaN"):
        psis(log_ratios)


def test_positive_inf_entry_raises() -> None:
    log_ratios = np.zeros(20)
    log_ratios[5] = np.inf
    with pytest.raises(ValueError, match="1 \\+inf"):
        psis(log_ratios)


def test_too_few_samples_raises() -> None:
    with pytest.raises(ValueError):
        psis(np.zeros(9))


def test_negative_inf_entries_get_zero_weight() -> None:
    log_ratios = np.random.default_rng(0).normal(size=200)
    log_ratios[:5] = -np.inf
    result = psis(log_ratios)
    assert np.all(np.isneginf(result.log_weights[:5]))
    assert np.all(np.isfinite(result.log_weights[5:]))
    assert logsumexp(result.log_weights) == pytest.approx(0.0, abs=1e-12)
