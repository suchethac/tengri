# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust age weight precomputation and fast dust attenuation.

Validates that precomputing the age-dependent sigmoid weight once
at init gives identical results to recomputing it per call, and that
the precomputed path compiles to strictly less work.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import (
    precompute_dust_age_weights,
    two_component_dust,
    two_component_dust_fast,
)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient of scalar f at x."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def age_grid():
    """Typical SSP age grid (107 points, log-spaced)."""
    return 10.0 ** jnp.linspace(5.5, 10.14, 107)


@pytest.fixture
def filter_wavelengths():
    """Effective wavelengths for 5 SDSS bands (rest-frame Angstrom)."""
    return jnp.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])


@pytest.fixture
def spectral_wavelengths():
    """200 spectral pixel wavelengths for spectroscopy tests."""
    return jnp.linspace(3500.0, 9500.0, 200)


@pytest.fixture
def dust_age_weights(age_grid):
    return precompute_dust_age_weights(age_grid)


# ── precompute_dust_age_weights ───────────────────────────────────
class TestPrecomputeDustAgeWeights:
    """Tests for the age weight precomputation function."""

    def test_output_shape(self, age_grid):
        """Output has same shape as age_grid."""
        weights = precompute_dust_age_weights(age_grid)
        chex.assert_equal_shape([weights, age_grid])

    def test_young_stars_near_one(self, age_grid):
        """Stars younger than t_birth have weight ~1."""
        weights = precompute_dust_age_weights(age_grid)
        young_mask = age_grid < 1e5  # well below 10 Myr
        if jnp.any(young_mask):
            assert jnp.all(weights[young_mask] > 0.99)

    def test_old_stars_near_zero(self, age_grid):
        """Stars older than t_birth have weight ~0."""
        weights = precompute_dust_age_weights(age_grid)
        old_mask = age_grid > 1e9  # well above 10 Myr
        assert jnp.all(weights[old_mask] < 0.01)

    def test_monotonically_decreasing(self, age_grid):
        """Weights decrease with age (more dust for young stars)."""
        weights = precompute_dust_age_weights(age_grid)
        # The grid is sorted ascending in age, weights should be non-increasing
        assert jnp.all(jnp.diff(weights) <= 1e-10)

    def test_values_between_zero_and_one(self, age_grid):
        """Sigmoid output is always in [0, 1]."""
        weights = precompute_dust_age_weights(age_grid)
        assert jnp.all(weights >= 0.0)
        assert jnp.all(weights <= 1.0)

    def test_custom_t_birth(self, age_grid):
        """Changing t_birth shifts the transition."""
        w_default = precompute_dust_age_weights(age_grid, t_birth=1e7)
        w_late = precompute_dust_age_weights(age_grid, t_birth=1e8)
        # With later t_birth, more stars count as "young"
        assert float(jnp.sum(w_late)) > float(jnp.sum(w_default))

    def test_custom_width(self, age_grid):
        """Narrower transition gives sharper step."""
        w_sharp = precompute_dust_age_weights(age_grid, transition_width=0.1)
        w_smooth = precompute_dust_age_weights(age_grid, transition_width=1.0)
        # Sharp transition has more extreme values (more near 0 or 1)
        assert float(jnp.std(w_sharp)) > float(jnp.std(w_smooth))

    def test_is_deterministic(self, age_grid):
        """Same inputs give same outputs."""
        w1 = precompute_dust_age_weights(age_grid)
        w2 = precompute_dust_age_weights(age_grid)
        assert_allclose(w1, w2, atol=0.0)


# ── charlot_fall_at_wavelengths_fast vs charlot_fall_at_wavelengths
_CF_KWARGS = {"law_bc": "power_law", "law_diff": "power_law"}


class TestFastDustAgreement:
    """Fast dust must exactly match the original per-call version."""

    def test_exact_agreement_photometry(self, age_grid, filter_wavelengths, dust_age_weights):
        """Fast and original agree exactly for photometric wavelengths."""
        tau_v1, tau_v2, n_slope = 0.5, 0.3, -0.7
        result_original = two_component_dust(
            filter_wavelengths,
            age_grid,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            n_slope=n_slope,
            **_CF_KWARGS,
        )
        result_fast = two_component_dust_fast(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            n_slope=n_slope,
            **_CF_KWARGS,
        )
        assert_allclose(result_fast, result_original, rtol=1e-12)

    def test_exact_agreement_spectroscopy(self, age_grid, spectral_wavelengths, dust_age_weights):
        """Fast and original agree exactly for spectroscopic wavelengths."""
        tau_v1, tau_v2, n_slope = 1.0, 0.5, -0.7
        result_original = two_component_dust(
            spectral_wavelengths,
            age_grid,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            n_slope=n_slope,
            **_CF_KWARGS,
        )
        result_fast = two_component_dust_fast(
            spectral_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            n_slope=n_slope,
            **_CF_KWARGS,
        )
        assert_allclose(result_fast, result_original, rtol=1e-12)

    @pytest.mark.parametrize(
        "tau_v1,tau_v2,n_slope",
        [
            (0.0, 0.0, -0.7),  # no dust
            (3.0, 1.5, -0.7),  # heavy dust
            (0.5, 0.3, -1.3),  # steep curve (Calzetti-like)
            (0.5, 0.3, -0.3),  # shallow curve
            (0.01, 0.01, -0.7),  # nearly no dust
        ],
    )
    def test_agreement_various_params(
        self,
        age_grid,
        filter_wavelengths,
        dust_age_weights,
        tau_v1,
        tau_v2,
        n_slope,
    ):
        """Agreement holds across diverse dust parameter combinations."""
        result_original = two_component_dust(
            filter_wavelengths,
            age_grid,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            n_slope=n_slope,
            **_CF_KWARGS,
        )
        result_fast = two_component_dust_fast(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            n_slope=n_slope,
            **_CF_KWARGS,
        )
        assert_allclose(result_fast, result_original, rtol=1e-12)

    def test_output_shape(self, filter_wavelengths, dust_age_weights):
        """Output shape is (n_ages, n_filters)."""
        result = two_component_dust_fast(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=0.5,
            tau_v2=0.3,
            **_CF_KWARGS,
        )
        assert result.shape == (len(dust_age_weights), len(filter_wavelengths))


# ── Gradient tests ────────────────────────────────────────────────
class TestFastDustGradients:
    """Gradients through the fast dust function are correct."""

    def test_gradients_finite(self, filter_wavelengths, dust_age_weights):
        """Gradients of two_component_dust_fast match central FD."""

        def loss(tau_v1, tau_v2):
            atten = two_component_dust_fast(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )
            return jnp.sum(atten)

        g1, g2 = jax.grad(loss, argnums=(0, 1))(0.5, 0.3)

        def f1(tau_v1: float) -> float:
            return float(loss(tau_v1, 0.3))

        def f2(tau_v2: float) -> float:
            return float(loss(0.5, tau_v2))

        np.testing.assert_allclose(
            float(g1),
            fd_grad(f1, 0.5),
            rtol=1e-3,
            err_msg="two_component_dust_fast: FD check ∂(∑atten)/∂tau_bc",
        )
        np.testing.assert_allclose(
            float(g2),
            fd_grad(f2, 0.3),
            rtol=1e-3,
            err_msg="two_component_dust_fast: FD check ∂(∑atten)/∂tau_diff",
        )

    def test_gradients_match_original(self, age_grid, filter_wavelengths, dust_age_weights):
        """Autodiff gradients match between fast and original."""

        def loss_original(tau_v1, tau_v2):
            atten = two_component_dust(
                filter_wavelengths,
                age_grid,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )
            return jnp.sum(atten)

        def loss_fast(tau_v1, tau_v2):
            atten = two_component_dust_fast(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )
            return jnp.sum(atten)

        g_orig = jax.grad(loss_original, argnums=(0, 1))(0.5, 0.3)
        g_fast = jax.grad(loss_fast, argnums=(0, 1))(0.5, 0.3)
        assert_allclose(g_fast[0], g_orig[0], rtol=1e-10)
        assert_allclose(g_fast[1], g_orig[1], rtol=1e-10)

    def test_jit_compatible(self, filter_wavelengths, dust_age_weights):
        """Fast dust function works inside jax.jit."""

        @jax.jit
        def fn(tau_v1, tau_v2):
            return two_component_dust_fast(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )

        result = fn(0.5, 0.3)
        chex.assert_tree_all_finite(result)


# ── Precompute must do strictly less work ─────────────────────────
class TestFastDustDoesLessWork:
    """The precomputed path must compile to strictly less work than the exact one."""

    def test_fast_path_compiles_to_fewer_flops(
        self, age_grid, filter_wavelengths, dust_age_weights
    ):
        """The precomputed path compiles to strictly fewer FLOPs than the exact path.

        This guard exists to catch ``two_component_dust_fast`` silently
        degrading into the per-call work it is supposed to skip. It used to
        assert a wall-clock ratio (``t_fast < t_original * 3.0``) and was
        marked ``benchmark`` so it ran serially in the smoke job. That
        formulation could not do the job, for two independent reasons —
        both measured, 2026-08-11:

        **1. It could not detect the regression.** Timed at the benchmark's own
        setup the ratio is 1.00 at every problem size (5, 20, 50, 200, 1000,
        5000 wavelengths). The cause is constant folding: with ``age_grid``
        closed over, it is a compile-time constant, so XLA folds the
        age-weight computation *inside the exact path* — precisely the work
        the precompute exists to skip. Compiled cost makes it exact rather
        than approximate: closed over, both paths compile to **1717 FLOPs**,
        the same number. A fast path that fell back to the exact one would
        move the ratio from 1.00 to 1.00.

        **2. It failed on noise.** Equal-cost arms leave the whole 3x bound to
        absorb runner jitter, and on 2026-08-11 a shared runner returned
        43.4us vs 13.3us (3.26x) on ``main``. Because ``smoke`` gates
        ``test``, that one wobble skipped the entire ~13000-test matrix.

        Passing the age grid as a **traced argument** blocks the folding and
        the real difference appears: exact **2573 FLOPs** vs fast **1717**,
        a ratio of 0.667 (wall clock agrees: 0.76). FLOP counts come from the
        compiled executable, are identical across recompiles, and cannot be
        moved by scheduler load — so this runs in the ordinary parallel sweep
        and the ``benchmark`` marker (defined in ``pyproject.toml`` as
        "asserts a wall-clock ratio") no longer applies.

        Tracing the grid is not an artificial setup: threading grids as
        runtime arguments instead of baking them in as constants is the
        direction of #1383 and #1650.
        """

        def _flops(fn, *args) -> float:
            """Compiled FLOP count — deterministic, unlike wall clock."""
            return jax.jit(fn).lower(*args).compile().cost_analysis()["flops"]

        # The age grid must be TRACED, not closed over. Closed over, XLA folds
        # the exact path's age-weight work into a constant and the two paths
        # compile to an identical cost, which is what made the old wall-clock
        # form blind.
        flops_exact = _flops(
            lambda tv1, tv2, ages: two_component_dust(
                filter_wavelengths, ages, tau_v1=tv1, tau_v2=tv2, **_CF_KWARGS
            ),
            0.5,
            0.3,
            age_grid,
        )
        flops_fast = _flops(
            lambda tv1, tv2, weights: two_component_dust_fast(
                filter_wavelengths, weights, tau_v1=tv1, tau_v2=tv2, **_CF_KWARGS
            ),
            0.5,
            0.3,
            dust_age_weights,
        )

        assert flops_fast < flops_exact, (
            f"Precomputed path does not do less work: fast {flops_fast:,.0f} FLOPs "
            f"vs exact {flops_exact:,.0f}. Equal counts mean the fast path is "
            f"recomputing the age weights it is meant to receive."
        )
