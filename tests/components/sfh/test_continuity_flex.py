# SPDX-License-Identifier: BSD-3-Clause
"""Tests for continuity_flex SFH model (Leja+2019).

Tests cover shape, non-negativity, mass conservation, custom anchors,
JIT/gradient compatibility, and prior log-probability.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sfh.nonparametric import (
    continuity_flex,
    continuity_flex_prior_logp,
)

pytestmark = pytest.mark.bounds


class TestContinuityFlexSFH:
    """Tests for continuity_flex and continuity_flex_prior_logp."""

    def _age_grid(self):
        return jnp.logspace(6.0, 10.14, 256)

    def test_shape(self):
        t = self._age_grid()
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            ratio_young=0.0,
            flex_0=0.0,
            flex_1=0.0,
            flex_2=0.0,
            ratio_old=0.0,
        )
        chex.assert_shape(sfr, (256,))

    def test_non_negative(self):
        t = self._age_grid()
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            ratio_young=2.0,
            flex_0=-1.0,
            flex_1=1.5,
            ratio_old=-2.0,
        )
        assert jnp.all(sfr >= 0.0)

    def test_mass_conservation(self):
        """Integrated SFR * dt should equal 10^log_total_mass."""
        log_m = 10.5
        t = jnp.linspace(0.0, 13.7e9, 100_000)
        sfr = continuity_flex(
            t,
            log_total_mass=log_m,
            ratio_young=0.3,
            flex_0=0.1,
            flex_1=-0.2,
            flex_2=0.0,
            ratio_old=-0.5,
        )
        dt = t[1] - t[0]
        mass_integrated = jnp.sum(sfr) * dt
        assert abs(float(mass_integrated) / 10.0**log_m - 1.0) < 0.01

    def test_flat_sfh_from_zero_ratios(self):
        """All-zero ratios should give a flat SFH (constant SFR)."""
        t = jnp.array([1e8, 1e9, 3e9, 8e9])
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            ratio_young=0.0,
            flex_0=0.0,
            ratio_old=0.0,
        )
        # SFR in the interior flex region should be constant
        assert jnp.allclose(sfr[1], sfr[2], rtol=1e-5)

    def test_zero_ratios_n_flex_0(self):
        """With no flex_* kwargs, n_flex_ratios=0 and we still get a valid SFH."""
        t = self._age_grid()
        sfr = continuity_flex(t, log_total_mass=10.0, ratio_young=0.0, ratio_old=0.0)
        chex.assert_equal_shape([sfr, t])
        assert jnp.all(sfr >= 0.0)

    def test_custom_anchor_edges(self):
        """Custom bin_edges_gyr should be accepted and yield finite SFR."""
        anchors = jnp.array([0.05, 4.0, 12.0])
        t = self._age_grid()
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            bin_edges_gyr=anchors,
            ratio_young=0.0,
            flex_0=0.2,
            ratio_old=0.0,
        )
        chex.assert_tree_all_finite(sfr)
        assert jnp.any(sfr > 0)

    def test_jit_compatible(self):
        t = self._age_grid()

        def _fn(ry, f0, f1, ro):
            return continuity_flex(t, 10.0, ratio_young=ry, flex_0=f0, flex_1=f1, ratio_old=ro)

        sfr = jax.jit(_fn)(0.3, 0.1, -0.2, -0.4)
        chex.assert_tree_all_finite(sfr)

    def test_gradient_through_ratio_young(self):
        """Gradient w.r.t. ratio_young should be finite."""
        t = self._age_grid()

        def total_sfr(ratio_young):
            return jnp.sum(
                continuity_flex(t, 10.0, ratio_young=ratio_young, flex_0=0.0, ratio_old=0.0)
            )

        g = jax.grad(total_sfr)(0.5)
        assert jnp.isfinite(g)

    def test_gradient_through_flex_ratio(self):
        """Gradient w.r.t. a flex bin ratio should be finite."""
        t = self._age_grid()

        def total_sfr(flex_0):
            return jnp.sum(continuity_flex(t, 10.0, ratio_young=0.0, flex_0=flex_0, ratio_old=0.0))

        g = jax.grad(total_sfr)(0.3)
        assert jnp.isfinite(g)

    def test_prior_logp_zero_ratios(self):
        """All-zero ratios should give maximum log-probability.

        Moved here from ``tests/components/sfh/test_dense_basis.py``, which
        carried a ``TestContinuityFlexSFH`` class that was otherwise a strict
        subset of this one. This was the single test that lived only in that
        copy — the file docstring above already promised prior-log-probability
        coverage that was not actually here.
        """
        logp_zero = continuity_flex_prior_logp(0.0, jnp.array([0.0, 0.0]), 0.0)
        logp_nonzero = continuity_flex_prior_logp(1.0, jnp.array([1.0, 1.0]), 1.0)
        assert float(logp_zero) > float(logp_nonzero)
