# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Suess+2021 PSB nonparametric SFH.

Tests cover positivity, finiteness, mass scaling, and JIT compatibility.
"""

import functools

import chex
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sfh.nonparametric import psb_continuity
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds


class TestPSBContinuitySFH:
    """Tests for Suess+2021 PSB nonparametric SFH."""

    @pytest.fixture
    def age_yr(self):
        return jnp.linspace(1e6, 10e9, 200)

    @pytest.fixture
    def default_edges(self):
        return jnp.array([0.1, 1.0, 3.0, 6.0, 13.7])

    def test_non_negative(self, age_yr, default_edges):
        sfr = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        assert jnp.all(sfr >= 0.0)

    def test_finite(self, age_yr, default_edges):
        sfr = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        chex.assert_tree_all_finite(sfr)

    def test_mass_scales_with_log_total_mass(self, age_yr, default_edges):
        sfr10 = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        sfr11 = psb_continuity(
            age_yr,
            11.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        ratio = float(jnp.sum(sfr11) / jnp.sum(sfr10))
        assert abs(ratio - 10.0) < 0.5, f"10x mass increase should give ~10x SFR, got {ratio:.2f}"

    def test_jit_compatible(self, age_yr, default_edges):
        # bin_edges_gyr is a fixed structural arg — bake it in via partial before JIT
        sfr = assert_jit_matches_eager(
            functools.partial(psb_continuity, bin_edges_gyr=default_edges),
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        chex.assert_tree_all_finite(sfr)
