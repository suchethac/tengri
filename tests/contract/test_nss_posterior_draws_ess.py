# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the NSS posterior draw count (#715).

Surface protected: ``_resolve_posterior_draws`` in the nested-sampling evidence
backend. With ``n_posterior_samples=None`` (the default) the number of
resampled posterior draws is ``ceil`` of the Kish effective sample size of the
normalized dead-point weights; an explicit integer is honored exactly; a
weight ESS below the warning floor warns with its value and is never padded.

The toy weighted set uses the real ``nested.utils`` ESS and resampler, so the
draw count is checked against the weights the backend actually uses.
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.backends import evidence
from tengri.inference.backends.nested.base import NSInfo, StateWithLogLikelihood
from tengri.inference.backends.nested.utils import ess as ns_ess, sample as ns_sample

pytestmark = pytest.mark.contract


class TestDefaultDrawsFollowEss:
    """With no explicit count, the draw count is ceil(Kish ESS)."""

    @pytest.mark.parametrize(
        ("ess_value", "expected"),
        [(237.2, 238), (1000.0, 1000), (1000.0000001, 1001), (5.5, 6)],
    )
    @pytest.mark.filterwarnings("ignore:NSS weight ESS")
    def test_default_draws_ceil_of_ess(self, ess_value, expected):
        """The default count is ceil(ESS) for fractional and exact ESS values."""
        assert evidence._resolve_posterior_draws(None, ess_value) == expected

    def test_default_draws_on_toy_weighted_set_match_ess(self):
        """ns_sample on a toy dead set returns ceil(Kish ESS) rows."""
        n_dead = 40
        loglike = jnp.linspace(-3.0, 0.0, n_dead)
        particles = StateWithLogLikelihood(
            position={"x": jnp.linspace(0.0, 1.0, n_dead)},
            logdensity=jnp.zeros(n_dead),
            loglikelihood=loglike,
            loglikelihood_birth=loglike - 0.5,
        )
        dead = NSInfo(particles, None)
        ess_val = float(ns_ess(jax.random.PRNGKey(0), dead))
        n_draws = evidence._resolve_posterior_draws(None, ess_val)
        assert n_draws == math.ceil(ess_val)
        assert 1 <= n_draws <= n_dead + 1
        drawn = ns_sample(jax.random.PRNGKey(1), dead, n_draws)
        assert drawn.position["x"].shape == (n_draws,)

    def test_low_ess_warns_with_value_and_is_not_padded(self):
        """An ESS below the floor warns with its value and draws ceil(ESS), no padding."""
        with pytest.warns(UserWarning, match=r"ESS = 0\.42"):
            n = evidence._resolve_posterior_draws(None, 0.42)
        assert n == 1

    def test_healthy_ess_does_not_warn(self):
        """No warning when the ESS is at or above the floor."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            evidence._resolve_posterior_draws(None, 250.0)


class TestExplicitDrawsHonored:
    """An explicit integer is used exactly, whatever the ESS says."""

    @pytest.mark.parametrize("n", [1, 7, 1000, 5000])
    def test_explicit_count_is_returned_unchanged(self, n):
        """Explicit n is returned exactly, independent of the ESS."""
        assert evidence._resolve_posterior_draws(n, 37.3) == n

    @pytest.mark.parametrize("bad", [0, -4, 2.5, True])
    def test_invalid_explicit_count_raises(self, bad):
        """Zero, negative, fractional and boolean counts are rejected."""
        with pytest.raises(ValueError, match="n_posterior_samples"):
            evidence._resolve_posterior_draws(bad, 100.0)

    @pytest.mark.parametrize("ess_value", [float("nan"), float("inf"), 0.0, -1.0])
    def test_non_finite_or_nonpositive_ess_raises_for_default(self, ess_value):
        """A default draw count needs a finite, positive ESS."""
        with pytest.raises(ValueError, match="ESS"):
            evidence._resolve_posterior_draws(None, ess_value)
