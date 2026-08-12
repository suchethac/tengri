# SPDX-License-Identifier: BSD-3-Clause
import pytest

pytestmark = pytest.mark.contract
"""Tests for distribution objects used in Parameters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.parameters.priors import (
    Fixed,
    Gaussian,
    LogUniform,
    Uniform,
    resolve_shorthand,
)
from tests._jit_parity import assert_vmap_matches_loop


class TestUniform:
    def test_bounds(self):
        d = Uniform(0.1, 5.0)
        assert d.bounds == (0.1, 5.0)
        assert d.lo == 0.1
        assert d.hi == 5.0

    def test_not_fixed(self):
        assert not Uniform(0.0, 1.0).is_fixed

    def test_sample_in_bounds(self):
        d = Uniform(0.1, 5.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 1000)
        samples = jax.vmap(d.sample)(keys)
        assert jnp.all(samples >= 0.1)
        assert jnp.all(samples <= 5.0)

    def test_vmap_sampling_matches_a_loop(self):
        """A bound holds just as well for a batch that ignored its keys.

        ``vmap`` that mapped the wrong axis, or a sampler that dropped its key
        and returned one value broadcast 1000 times, still lands inside
        [0.1, 5.0] — so ``test_sample_in_bounds`` above cannot see it. Checked
        on a slice rather than the full batch: 8 looped calls cost nothing,
        where looping all 1000 would make a millisecond test a slow one for no
        extra signal.
        """
        d = Uniform(0.1, 5.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 1000)
        assert_vmap_matches_loop(d.sample, keys[:8])

    def test_log_prob_flat_inside(self):
        d = Uniform(0.0, 1.0)
        lp1 = d.log_prob(jnp.array(0.3))
        lp2 = d.log_prob(jnp.array(0.7))
        np.testing.assert_allclose(float(lp1), float(lp2))

    def test_log_prob_outside(self):
        d = Uniform(0.0, 1.0)
        assert d.log_prob(jnp.array(-0.1)) == -jnp.inf
        assert d.log_prob(jnp.array(1.5)) == -jnp.inf

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="lo < hi"):
            Uniform(5.0, 0.1)

    def test_repr(self):
        assert repr(Uniform(0.1, 5.0)) == "Uniform(0.1, 5.0)"

    def test_jit_compatible(self):
        d = Uniform(0.0, 1.0)
        key = jax.random.PRNGKey(0)
        val = jax.jit(d.sample)(key)
        assert jnp.isfinite(val)
        lp = jax.jit(d.log_prob)(jnp.array(0.5))
        assert jnp.isfinite(lp)


class TestGaussian:
    def test_bounds(self):
        d = Gaussian(0.0, 1.0, lo=-2.0, hi=2.0)
        assert d.bounds == (-2.0, 2.0)

    def test_unbounded_gaussian(self):
        d = Gaussian(0.0, 1.0)
        assert d.bounds == (float("-inf"), float("inf"))

    def test_not_fixed(self):
        assert not Gaussian(0.0, 1.0).is_fixed

    def test_sample_mean_close(self):
        d = Gaussian(3.0, 0.5)
        keys = jax.random.split(jax.random.PRNGKey(0), 10000)
        samples = jax.vmap(d.sample)(keys)
        np.testing.assert_allclose(float(jnp.mean(samples)), 3.0, atol=0.05)

    def test_clipped_sample_in_bounds(self):
        d = Gaussian(0.0, 1.0, lo=-2.0, hi=2.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 1000)
        samples = jax.vmap(d.sample)(keys)
        assert jnp.all(samples >= -2.0)
        assert jnp.all(samples <= 2.0)

    def test_log_prob_peak_at_mu(self):
        d = Gaussian(3.0, 0.5)
        lp_mu = d.log_prob(jnp.array(3.0))
        lp_off = d.log_prob(jnp.array(4.0))
        assert float(lp_mu) > float(lp_off)

    def test_log_prob_outside_bounds(self):
        d = Gaussian(0.0, 1.0, lo=-2.0, hi=2.0)
        assert d.log_prob(jnp.array(-3.0)) == -jnp.inf

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma > 0"):
            Gaussian(0.0, -1.0)

    def test_repr(self):
        r = repr(Gaussian(0.0, 1.0, lo=-2.0, hi=2.0))
        assert "mu=0.0" in r
        assert "sigma=1.0" in r
        assert "lo=-2.0" in r

    def test_jit_compatible(self):
        d = Gaussian(0.0, 1.0, lo=-3.0, hi=3.0)
        val = jax.jit(d.sample)(jax.random.PRNGKey(0))
        assert jnp.isfinite(val)
        lp = jax.jit(d.log_prob)(jnp.array(0.5))
        assert jnp.isfinite(lp)


class TestLogUniform:
    def test_bounds(self):
        d = LogUniform(0.01, 100.0)
        assert d.bounds == (0.01, 100.0)

    def test_not_fixed(self):
        assert not LogUniform(0.01, 100.0).is_fixed

    def test_sample_in_bounds(self):
        d = LogUniform(0.01, 100.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 1000)
        samples = jax.vmap(d.sample)(keys)
        assert jnp.all(samples >= 0.01)
        assert jnp.all(samples <= 100.0)

    def test_log_uniform_in_log_space(self):
        d = LogUniform(0.01, 100.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 10000)
        samples = jax.vmap(d.sample)(keys)
        log_samples = jnp.log10(samples)
        # Should be uniform in [-2, 2]
        assert float(jnp.mean(log_samples)) == pytest.approx(0.0, abs=0.1)

    def test_log_prob_formula(self):
        d = LogUniform(1.0, 10.0)
        x = jnp.array(5.0)
        expected = -jnp.log(5.0 * jnp.log(10.0))
        np.testing.assert_allclose(float(d.log_prob(x)), float(expected), rtol=1e-6)

    def test_log_prob_outside(self):
        d = LogUniform(1.0, 10.0)
        assert d.log_prob(jnp.array(0.5)) == -jnp.inf

    def test_invalid_lo(self):
        with pytest.raises(ValueError, match="lo > 0"):
            LogUniform(-1.0, 10.0)

    def test_repr(self):
        assert repr(LogUniform(0.01, 100.0)) == "LogUniform(0.01, 100.0)"

    def test_jit_compatible(self):
        d = LogUniform(0.01, 100.0)
        val = jax.jit(d.sample)(jax.random.PRNGKey(0))
        assert jnp.isfinite(val)


class TestFixed:
    def test_is_fixed(self):
        assert Fixed(0.3).is_fixed

    def test_value(self):
        assert Fixed(0.3).value == 0.3

    def test_bounds(self):
        assert Fixed(0.3).bounds == (0.3, 0.3)

    def test_sample_returns_value(self):
        d = Fixed(0.3)
        val = d.sample(jax.random.PRNGKey(0))
        np.testing.assert_allclose(float(val), 0.3)

    def test_log_prob_zero(self):
        d = Fixed(0.3)
        np.testing.assert_allclose(float(d.log_prob(jnp.array(0.3))), 0.0)

    def test_repr(self):
        assert repr(Fixed(0.3)) == "Fixed(0.3)"


class TestResolveShorthand:
    def test_scalar_to_fixed(self):
        d = resolve_shorthand(0.3)
        assert isinstance(d, Fixed)
        assert d.value == 0.3

    def test_int_to_fixed(self):
        d = resolve_shorthand(1)
        assert isinstance(d, Fixed)
        assert d.value == 1.0

    def test_tuple_to_uniform(self):
        d = resolve_shorthand((0.1, 5.0))
        assert isinstance(d, Uniform)
        assert d.lo == 0.1
        assert d.hi == 5.0

    def test_distribution_passthrough(self):
        g = Gaussian(0.0, 1.0)
        assert resolve_shorthand(g) is g

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Cannot resolve"):
            resolve_shorthand("bad")

    def test_invalid_tuple_length(self):
        with pytest.raises(TypeError, match="Cannot resolve"):
            resolve_shorthand((1, 2, 3))
