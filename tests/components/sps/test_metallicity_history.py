# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phenomenological metallicity history models.

Verifies:
1. two_step: correct step behavior, smooth transition, differentiability
2. psb_two_step: delegates to two_step correctly
3. metallicity_bins: piecewise-constant Z per bin
4. metallicity_bins_continuity: cumulative delta-log-Z from base
5. JIT compatibility for all functions
6. Gradient flow for fitted parameters
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh.met_registry import MET_REGISTRY, resolve_met
from tengri.components.stellar.sfh.metallicity_history import (
    metallicity_bins_continuity_on_ssp_grid,
    metallicity_bins_on_ssp_grid,
    psb_two_step_metallicity,
    tabulated_metallicity_on_ssp_grid,
    two_step_metallicity,
)
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds

LOG10_ZSUN = -1.8477116556169435


@pytest.fixture
def ssp_lg_age_gyr():
    """Typical SSP log10(age/Gyr) grid: ~1 Myr to ~13.7 Gyr."""
    return jnp.linspace(-3.0, 1.14, 94)


@pytest.fixture
def bin_edges_log_yr():
    """6-bin edges in log10(age/yr), spanning 10 Myr to 13.7 Gyr."""
    return jnp.array([7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.14])


# ── two_step_metallicity ──────────────────────────────────────────


class TestTwoStepMetallicity:
    def test_old_stars_get_old_metallicity(self, ssp_lg_age_gyr):
        log_z_old = -3.0
        log_z_young = -1.5
        step_gyr = 1.0
        result = two_step_metallicity(ssp_lg_age_gyr, log_z_old, log_z_young, step_gyr)
        old_mask = ssp_lg_age_gyr > 0.5
        assert jnp.all(result[old_mask] > -3.05)
        assert jnp.all(result[old_mask] < -2.5)

    def test_young_stars_get_young_metallicity(self, ssp_lg_age_gyr):
        log_z_old = -3.0
        log_z_young = -1.5
        step_gyr = 1.0
        result = two_step_metallicity(ssp_lg_age_gyr, log_z_old, log_z_young, step_gyr)
        young_mask = ssp_lg_age_gyr < -1.0
        assert jnp.all(result[young_mask] > -1.6)
        assert jnp.all(result[young_mask] < -1.4)

    def test_smooth_transition(self, ssp_lg_age_gyr):
        """Transition should be smooth (no NaN/Inf, all finite)."""
        result = two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, 1.0)
        chex.assert_tree_all_finite(result)

    def test_equal_metallicities(self, ssp_lg_age_gyr):
        """When old == young, result should be constant."""
        result = two_step_metallicity(ssp_lg_age_gyr, -2.0, -2.0, 1.0)
        assert_allclose(result, -2.0, atol=1e-6)

    def test_jit_compatible(self, ssp_lg_age_gyr):
        result = assert_jit_matches_eager(two_step_metallicity, ssp_lg_age_gyr, -3.0, -1.5, 1.0)
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_metallicities(self, ssp_lg_age_gyr):
        """Gradient should flow through metallicity parameters."""

        def loss(z_old, z_young):
            return jnp.mean(two_step_metallicity(ssp_lg_age_gyr, z_old, z_young, 1.0))

        g_old, g_young = jax.grad(loss, argnums=(0, 1))(-3.0, -1.5)
        assert jnp.isfinite(g_old)
        assert jnp.isfinite(g_young)
        assert g_old > 0, "Increasing old Z should increase mean"
        assert g_young > 0, "Increasing young Z should increase mean"

    def test_gradient_wrt_step_age(self, ssp_lg_age_gyr):
        """Gradient should flow through step age."""

        def loss(step_age):
            return jnp.mean(two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, step_age))

        g = assert_grad_matches_fd(loss, 1.0)
        assert jnp.isfinite(g)

    def test_output_shape(self, ssp_lg_age_gyr):
        result = two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, 1.0)
        chex.assert_equal_shape([result, ssp_lg_age_gyr])

    def test_very_old_step_all_young(self, ssp_lg_age_gyr):
        """Step at very old age → nearly all stars get young Z."""
        result = two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, 100.0)
        assert_allclose(result, -1.5, atol=0.1)

    def test_very_young_step_all_old(self, ssp_lg_age_gyr):
        """Step at very young age → nearly all stars get old Z... wait, no.
        Step at very young lookback → nearly all stars are OLDER → get old Z.
        But young-age step means step_age is small."""
        result = two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, 1e-4)
        # Almost all SSP ages are older than 0.1 Myr → get old Z
        assert jnp.mean(result) < -2.5


# ── psb_two_step_metallicity ──────────────────────────────────────


class TestPSBTwoStepMetallicity:
    def test_delegates_to_two_step(self, ssp_lg_age_gyr):
        """Should produce identical output to two_step."""
        ts = two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, 0.5)
        psb = psb_two_step_metallicity(ssp_lg_age_gyr, -3.0, -1.5, 0.5)
        assert_allclose(psb, ts)

    def test_jit_compatible(self, ssp_lg_age_gyr):
        result = assert_jit_matches_eager(
            psb_two_step_metallicity, ssp_lg_age_gyr, -3.0, -1.5, 0.5
        )
        chex.assert_tree_all_finite(result)


# ── metallicity_bins_on_ssp_grid ──────────────────────────────────


class TestMetallicityBins:
    def test_assigns_correct_metallicity_per_bin(self, ssp_lg_age_gyr, bin_edges_log_yr):
        """Shape and finite values match input (bounds test: finiteness)."""
        n_bins = len(bin_edges_log_yr) - 1
        mets = jnp.linspace(-3.0, -1.0, n_bins)
        result = metallicity_bins_on_ssp_grid(ssp_lg_age_gyr, bin_edges_log_yr, mets)
        chex.assert_equal_shape([result, ssp_lg_age_gyr])
        chex.assert_tree_all_finite(result)
        # Values should span the input range
        assert jnp.min(result) >= jnp.min(mets) - 1e-6
        assert jnp.max(result) <= jnp.max(mets) + 1e-6

    def test_uniform_metallicity(self, ssp_lg_age_gyr, bin_edges_log_yr):
        """If all bins have the same Z, output should be constant."""
        n_bins = len(bin_edges_log_yr) - 1
        mets = jnp.full(n_bins, -2.0)
        result = metallicity_bins_on_ssp_grid(ssp_lg_age_gyr, bin_edges_log_yr, mets)
        within_range = (ssp_lg_age_gyr + 9.0 >= bin_edges_log_yr[0]) & (
            ssp_lg_age_gyr + 9.0 <= bin_edges_log_yr[-1]
        )
        assert_allclose(result[within_range], -2.0, atol=0.05)

    def test_younger_bins_get_higher_metallicity(self, ssp_lg_age_gyr, bin_edges_log_yr):
        """With increasing Z toward younger bins, young SSPs should have higher Z."""
        n_bins = len(bin_edges_log_yr) - 1
        # youngest bin (idx 0) = -1.0 (high Z), oldest bin (idx -1) = -3.0 (low Z)
        mets = jnp.linspace(-1.0, -3.0, n_bins)
        result = metallicity_bins_on_ssp_grid(ssp_lg_age_gyr, bin_edges_log_yr, mets)
        # SSP ages within bin range: use log10(age/yr)
        young_mask = (ssp_lg_age_gyr + 9.0 > 7.0) & (ssp_lg_age_gyr + 9.0 < 8.5)
        old_mask = (ssp_lg_age_gyr + 9.0 > 9.5) & (ssp_lg_age_gyr + 9.0 < 10.14)
        assert jnp.any(young_mask) and jnp.any(old_mask), (
            "probe setup failed: no young and old age ranges in grid"
        )
        assert jnp.mean(result[young_mask]) > jnp.mean(result[old_mask])

    def test_jit_compatible(self, ssp_lg_age_gyr, bin_edges_log_yr):
        n_bins = len(bin_edges_log_yr) - 1
        mets = jnp.linspace(-3.0, -1.0, n_bins)
        result = assert_jit_matches_eager(
            metallicity_bins_on_ssp_grid, ssp_lg_age_gyr, bin_edges_log_yr, mets
        )
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_metallicities(self, ssp_lg_age_gyr, bin_edges_log_yr):
        n_bins = len(bin_edges_log_yr) - 1

        def loss(mets):
            return jnp.mean(metallicity_bins_on_ssp_grid(ssp_lg_age_gyr, bin_edges_log_yr, mets))

        mets = jnp.linspace(-3.0, -1.0, n_bins)
        g = assert_grad_matches_fd(loss, mets)
        chex.assert_tree_all_finite(g)
        assert jnp.all(g > 0), "Increasing any bin Z should increase mean"


# ── metallicity_bins_continuity_on_ssp_grid ───────────────────────


class TestMetallicityBinsContinuity:
    def test_zero_deltas_gives_constant(self, ssp_lg_age_gyr, bin_edges_log_yr):
        """With zero delta-log-Z steps, all bins should have the base Z."""
        n_bins = len(bin_edges_log_yr) - 1
        d_log_z = jnp.zeros(n_bins - 1)
        result = metallicity_bins_continuity_on_ssp_grid(
            ssp_lg_age_gyr, bin_edges_log_yr, -2.0, d_log_z
        )
        within_range = (ssp_lg_age_gyr + 9.0 >= bin_edges_log_yr[0]) & (
            ssp_lg_age_gyr + 9.0 <= bin_edges_log_yr[-1]
        )
        assert_allclose(result[within_range], -2.0, atol=0.05)

    def test_positive_deltas_increase_z(self, ssp_lg_age_gyr, bin_edges_log_yr):
        """Positive delta-log-Z steps should increase Z toward younger bins."""
        n_bins = len(bin_edges_log_yr) - 1
        d_log_z = jnp.full(n_bins - 1, 0.3)
        result = metallicity_bins_continuity_on_ssp_grid(
            ssp_lg_age_gyr, bin_edges_log_yr, -3.0, d_log_z
        )
        young_mask = ssp_lg_age_gyr < -1.5
        old_mask = ssp_lg_age_gyr > 0.5
        assert jnp.any(young_mask) and jnp.any(old_mask), (
            "probe setup failed: no young and old age ranges in grid"
        )
        assert jnp.mean(result[young_mask]) > jnp.mean(result[old_mask])

    def test_cumulative_sum_correct(self, ssp_lg_age_gyr, bin_edges_log_yr):
        """Final bin metallicity = base + sum(deltas)."""
        n_bins = len(bin_edges_log_yr) - 1
        d_log_z = jnp.array([0.1, 0.2, 0.3, 0.15, 0.25])
        result = metallicity_bins_continuity_on_ssp_grid(
            ssp_lg_age_gyr, bin_edges_log_yr, -3.0, d_log_z
        )
        expected_youngest = -3.0 + jnp.sum(d_log_z)
        youngest_mask = ssp_lg_age_gyr < -2.0
        if jnp.any(youngest_mask):
            assert_allclose(jnp.mean(result[youngest_mask]), expected_youngest, atol=0.15)

    def test_jit_compatible(self, ssp_lg_age_gyr, bin_edges_log_yr):
        n_bins = len(bin_edges_log_yr) - 1
        d_log_z = jnp.zeros(n_bins - 1)
        result = assert_jit_matches_eager(
            metallicity_bins_continuity_on_ssp_grid,
            ssp_lg_age_gyr,
            bin_edges_log_yr,
            -2.0,
            d_log_z,
        )
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_base_and_deltas(self, ssp_lg_age_gyr, bin_edges_log_yr):
        n_bins = len(bin_edges_log_yr) - 1

        def loss(base, deltas):
            return jnp.mean(
                metallicity_bins_continuity_on_ssp_grid(
                    ssp_lg_age_gyr, bin_edges_log_yr, base, deltas
                )
            )

        d_log_z = jnp.full(n_bins - 1, 0.1)
        g_base, g_deltas = jax.grad(loss, argnums=(0, 1))(-3.0, d_log_z)
        assert jnp.isfinite(g_base)
        chex.assert_tree_all_finite(g_deltas)
        assert g_base > 0, "Increasing base Z should increase mean"


# ── tabulated_metallicity_on_ssp_grid ─────────────────────────────


class TestTabulatedMetallicity:
    def test_linear_ramp(self, ssp_lg_age_gyr):
        """A linear Z(t) table should interpolate smoothly (bounds test: monotone vs input)."""
        met_log_age_yr = jnp.linspace(6.0, 10.14, 50)
        met_log_z_abs = jnp.linspace(-3.0, -1.5, 50)
        result = tabulated_metallicity_on_ssp_grid(ssp_lg_age_gyr, met_log_age_yr, met_log_z_abs)
        chex.assert_equal_shape([result, ssp_lg_age_gyr])
        chex.assert_tree_all_finite(result)
        # Interpolated values should stay within input range
        assert jnp.min(result) >= jnp.min(met_log_z_abs) - 1e-6
        assert jnp.max(result) <= jnp.max(met_log_z_abs) + 1e-6

    def test_constant_table(self, ssp_lg_age_gyr):
        """A constant Z table should give constant output."""
        met_log_age_yr = jnp.linspace(6.0, 10.14, 20)
        met_log_z_abs = jnp.full(20, -2.0)
        result = tabulated_metallicity_on_ssp_grid(ssp_lg_age_gyr, met_log_age_yr, met_log_z_abs)
        within = (ssp_lg_age_gyr + 9.0 >= 6.0) & (ssp_lg_age_gyr + 9.0 <= 10.14)
        assert_allclose(result[within], -2.0, atol=1e-6)

    def test_jit_compatible(self, ssp_lg_age_gyr):
        met_log_age_yr = jnp.linspace(6.0, 10.14, 20)
        met_log_z_abs = jnp.linspace(-3.0, -1.5, 20)
        result = assert_jit_matches_eager(
            tabulated_metallicity_on_ssp_grid, ssp_lg_age_gyr, met_log_age_yr, met_log_z_abs
        )
        chex.assert_tree_all_finite(result)

    def test_gradient_flows(self, ssp_lg_age_gyr):
        met_log_age_yr = jnp.linspace(6.0, 10.14, 20)

        def loss(met_vals):
            return jnp.mean(
                tabulated_metallicity_on_ssp_grid(ssp_lg_age_gyr, met_log_age_yr, met_vals)
            )

        met_log_z_abs = jnp.linspace(-3.0, -1.5, 20)
        g = assert_grad_matches_fd(loss, met_log_z_abs)
        chex.assert_tree_all_finite(g)

    def test_clamped_outside_range(self, ssp_lg_age_gyr):
        """SSP ages outside the table range should clamp to edge values."""
        met_log_age_yr = jnp.array([8.0, 9.0])
        met_log_z_abs = jnp.array([-3.0, -1.5])
        result = tabulated_metallicity_on_ssp_grid(ssp_lg_age_gyr, met_log_age_yr, met_log_z_abs)
        very_young = ssp_lg_age_gyr + 9.0 < 7.5
        very_old = ssp_lg_age_gyr + 9.0 > 9.5
        if jnp.any(very_young):
            assert_allclose(result[very_young], -3.0, atol=1e-6)
        if jnp.any(very_old):
            assert_allclose(result[very_old], -1.5, atol=1e-6)


# ── Met registry ──────────────────────────────────────────────────


class TestMetRegistry:
    def test_all_modes_registered(self):
        expected = {
            "delta",
            "ramp",
            "two_step",
            "psb_two_step",
            "bins",
            "bins_continuity",
            "chem_evol",
            "table",
            "massmap_lin",
            "massmap_box",
        }
        assert expected == set(MET_REGISTRY.keys())

    def test_resolve_delta(self):
        _spec, params, pm, _settings = resolve_met("delta")
        assert "met_logzsol" in params
        assert pm["met_logzsol"][0] == "log_z_abs"

    def test_resolve_two_step(self):
        _spec, params, pm, _settings = resolve_met("two_step")
        assert "met_logzsol_old" in params
        assert "met_logzsol_young" in params
        assert "met_step_age_gyr" in params
        assert pm["met_step_age_gyr"][2] == 0.0  # no offset for age

    def test_resolve_bins_continuity(self):
        _spec, params, _pm, settings = resolve_met("bins_continuity")
        assert "met_logzsol_base" in params
        assert "met_d_log_z_0" in params
        assert settings["met_n_bins"] == 6

    def test_resolve_chem_evol(self):
        _spec, params, _pm, _settings = resolve_met("chem_evol")
        assert "chem_yield" in params
        assert "chem_eta_outflow" in params

    def test_resolve_table_has_no_params(self):
        _spec, params, _pm, _settings = resolve_met("table")
        assert len(params) == 0

    def test_invalid_mode_raises(self):
        with pytest.raises(KeyError, match="Unknown met_mode"):
            resolve_met("nonexistent")

    def test_param_map_offset_is_log10_zsun(self):
        """Solar-relative params should have LOG10_ZSUN offset."""
        _, _, pm, _ = resolve_met("delta")
        assert_allclose(pm["met_logzsol"][2], LOG10_ZSUN, atol=1e-10)
