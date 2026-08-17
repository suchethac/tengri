# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the generic template preintegration module — bounds and finiteness.

Validates:
1. preintegrate_grid() basic functionality with synthetic templates
2. Energy normalization mode
3. Taylor moment computation
4. Output shape preservation and finiteness
5. Correctness against existing SSP precomputation
6. preintegrate_lines() basic functionality
7. interp_nd_triweight() 1D and 2D cases
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tengri.utils.grid_interp import (
    interp_nd_triweight,
    preintegrate_grid,
    preintegrate_lines,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid
from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds

# ── Fixtures: synthetic templates and filters ─────────────────────


@pytest.fixture(scope="module")
def synthetic_template_3d():
    """Simple 3D template: (3 metallicities, 5 ages, 1000 wavelengths)."""
    n_met, n_age, n_wave = 3, 5, 1000
    wave = jnp.linspace(1000.0, 10000.0, n_wave)
    template_0d = wave ** (-1.0)
    template = jnp.tile(template_0d, (n_met, n_age, 1))
    met_factor = jnp.array([0.5, 1.0, 1.5])[:, None, None]
    age_factor = jnp.linspace(0.5, 2.0, n_age)[None, :, None]
    template = template * met_factor * age_factor
    return template, wave


@pytest.fixture(scope="module")
def tophat_filters():
    """3 simple top-hat filters."""
    filter_waves = [
        jnp.linspace(1000.0, 2000.0, 50),
        jnp.linspace(4000.0, 5000.0, 50),
        jnp.linspace(8000.0, 9000.0, 50),
    ]
    filter_trans = [
        jnp.ones(50),
        jnp.ones(50),
        jnp.ones(50),
    ]
    return filter_waves, filter_trans


@pytest.fixture(scope="module")
def line_wavelengths():
    """3 emission lines at specific wavelengths (rest frame)."""
    return jnp.array([1500.0, 4500.0, 8500.0])


# ── Tests: preintegrate_grid() basic functionality ────────────────


class TestPreintegrateGridBasic:
    """Bounds tests: preintegrate_grid() basic functionality."""

    def test_flat_template_returns_its_constant(self, tophat_filters):
        """A flat L_ν template band-averages to exactly its constant value.

        Limit test: Φ = ∫ C·T·w dλ / ∫ T·w dλ = C for any filter and weight
        convention — the bandpass average of a constant is that constant.
        Subsumes the old shape-only / finite-only / positivity smoke tests:
        shapes are asserted and every value is checked exactly.
        """
        const = 2.5e28
        template = jnp.full((3, 5, 1000), const)
        wave = jnp.linspace(1000.0, 10000.0, 1000)
        filter_waves, filter_trans = tophat_filters
        n_filters = len(filter_waves)

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        chex.assert_shape(result.phot, (3, 5, n_filters))
        chex.assert_shape(result.effective_wavelengths, (n_filters,))
        chex.assert_shape(result.effective_wavelengths_rest, (n_filters,))
        npt.assert_allclose(np.asarray(result.phot), const, rtol=1e-10)
        # Stored as log10 (#1859), so the scale itself is ~-57 dex; the
        # assertion is that it denotes a positive factor, not that it is one.
        assert 10.0 ** float(result.log10_flux_scale) > 0.0

    def test_photometry_is_linear_and_separable(self, synthetic_template_3d, tophat_filters):
        """Filter integration is linear in L_ν, so a separable template factorizes.

        The fixture template is base(λ) × met_factor[i] × age_factor[j], so
        phot[i, j, k] must equal phot[ref_i, ref_j, k] rescaled by the factor
        ratios — exactly, since Φ is a linear functional of the template.
        """
        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters
        met_factor = np.array([0.5, 1.0, 1.5])
        age_factor = np.linspace(0.5, 2.0, template.shape[1])

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        phot = np.asarray(result.phot)
        expected = (
            phot[1, 1, :][None, None, :]
            * (met_factor / met_factor[1])[:, None, None]
            * (age_factor / age_factor[1])[None, :, None]
        )
        npt.assert_allclose(phot, expected, rtol=1e-10)
        assert np.all(phot > 0.0)

    def test_effective_wavelengths_in_filter_range(self, synthetic_template_3d, tophat_filters):
        """Effective wavelengths lie within their respective filter ranges.

        Bounds test: definition of effective wavelength.
        """

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        for i, fw in enumerate(filter_waves):
            assert float(result.effective_wavelengths[i]) >= float(jnp.min(fw))
            assert float(result.effective_wavelengths[i]) <= float(jnp.max(fw))


# ── Tests: energy normalization mode ──────────────────────────────


class TestPreintegrateGridEnergyNormalization:
    """Bounds tests: preintegrate_grid() with energy_normalize=True."""

    def test_energy_normalize_divides_by_bolometric_luminosity(
        self, synthetic_template_3d, tophat_filters
    ):
        """energy_normalize divides each grid point by its ∫ L_ν dν.

        The documented contract: templates are normalized to unit bolometric
        luminosity in FREQUENCY space before filter integration (the
        historical bug divided by ∫ L_ν dλ instead, a wavelength-shape-
        dependent error). Recompute ∫ L_ν dν independently with a trapezoid
        over ν and assert the exact per-grid-point division. Subsumes the old
        shape-only and 'values comparable' smoke tests.
        """
        from tengri.utils.physics_constants import C_AA

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        raw = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )
        norm = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            energy_normalize=True,
        )

        chex.assert_equal_shape([norm.phot, raw.phot])
        nu = C_AA / np.asarray(wave)
        order = np.argsort(nu)
        bol = np.trapezoid(np.asarray(template)[..., order], nu[order], axis=-1)
        assert np.all(bol > 0.0)
        npt.assert_allclose(
            np.asarray(norm.phot), np.asarray(raw.phot) / bol[..., None], rtol=1e-10
        )


# ── Tests: Taylor moment computation ──────────────────────────────


class TestPreintegrateGridTaylorMoment:
    """Bounds tests: preintegrate_grid() with taylor=True."""

    def test_taylor_moment_approximately_zero_for_flat_template(self):
        """For a spectrally flat template, the Taylor moment should be ~zero.

        Bounds test: the moment is ∝ ∫ (λ - λ_eff) × T(λ) dλ, which is ~0
        by definition of λ_eff.
        """

        # Flat template (constant SED)
        template = jnp.ones((2, 3, 100))
        wave = jnp.linspace(1000.0, 10000.0, 100)
        filter_waves = [
            jnp.linspace(2000.0, 3000.0, 30),
            jnp.linspace(6000.0, 7000.0, 30),
        ]
        filter_trans = [jnp.ones(30), jnp.ones(30)]

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28, taylor=True
        )

        assert result.moment is not None
        chex.assert_equal_shape([result.moment, result.phot])
        # Moment should be small (not exactly zero due to numerical integration)
        abs_moment = jnp.abs(result.moment)
        max_moment = jnp.max(abs_moment)
        assert float(max_moment) < 1e-6


# ── Tests: preintegrate_lines() basic functionality ───────────────


class TestPreintegrateLines:
    """Bounds tests: preintegrate_lines() basic functionality."""

    def test_output_nonnegative(self, tophat_filters, line_wavelengths):
        """Line weights are non-negative, finite, of shape (n_lines, n_filters).

        Bounds test: weights must be ≥ 0.
        """

        filter_waves, filter_trans = tophat_filters
        lines = line_wavelengths

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        chex.assert_shape(result.line_filter_weights, (len(lines), len(filter_waves)))
        chex.assert_tree_all_finite(result.line_filter_weights)
        assert_non_negative(result.line_filter_weights, name="output")

    def test_lines_outside_filters_have_small_weight(self, tophat_filters):
        """Lines far outside filter ranges have near-zero weight.

        Bounds test: out-of-band suppression.
        """

        filter_waves, filter_trans = tophat_filters
        # Line far below minimum filter wavelength
        lines = jnp.array([100.0])  # < min of all filters

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        # Weights should be very small
        assert jnp.all(result.line_filter_weights < 1e-3)

    def test_line_in_single_filter_has_nonzero_weight(self):
        """A line inside a filter has nonzero weight in that filter only."""

        filter_waves = [jnp.linspace(4000.0, 5000.0, 50)]
        filter_trans = [jnp.ones(50)]
        lines = jnp.array([4500.0])

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        assert float(result.line_filter_weights[0, 0]) > 0.0

    def test_line_transmission_weighted(self):
        """Line weight is proportional to transmission at line wavelength.

        Bounds test: line flux weighted by filter transmission.
        """

        # Filter with variable transmission
        wave = jnp.linspace(4000.0, 5000.0, 100)
        trans = jnp.where(jnp.abs(wave - 4500.0) < 200.0, 1.0, 0.0)  # peaked at 4500
        filter_waves = [wave]
        filter_trans = [trans]

        # Line at the peak of the filter (4500 Angstrom)
        lines_peak = jnp.array([4500.0])
        # Line at the edge where transmission is lower (4300 Angstrom)
        lines_edge = jnp.array([4300.0])

        result_peak = preintegrate_lines(lines_peak, filter_waves, filter_trans, redshift=0.0)
        result_edge = preintegrate_lines(lines_edge, filter_waves, filter_trans, redshift=0.0)

        # Weight at peak should be higher than at edge
        assert float(result_peak.line_filter_weights[0, 0]) > float(
            result_edge.line_filter_weights[0, 0]
        )


# ── Tests: interp_nd_triweight() 1D case ──────────────────────────


class TestInterpNdTriweight1D:
    """Bounds tests: interp_nd_triweight() basic functionality in 1D."""

    def test_constant_grid_interpolates_to_constant_1d(self):
        """Interpolating a constant grid returns exactly the constant.

        Limit test: any convex kernel-weighted average of a constant field is
        that constant — shape (trailing dims) asserted, every value exact.
        Subsumes the old shape-only and finite-only tests.
        """
        axes = (jnp.linspace(0.0, 1.0, 10),)
        edges = (edges_for_grid(axes[0]),)
        grid_values = jnp.full((10, 5), 7.25)

        result = interp_nd_triweight(grid_values, axes, edges, (0.37,))

        chex.assert_shape(result, (5,))
        npt.assert_allclose(np.asarray(result), 7.25, rtol=1e-12)

    def test_query_at_grid_node_returns_value(self):
        """Querying at a grid node returns approximately that node's value.

        Bounds test: interpolation exactness at grid points.
        """

        # 1D grid: 5 nodes, 3 values per node
        axes = (jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),)
        edges = (edges_for_grid(axes[0]),)
        grid_values = jnp.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
            ]
        )

        # Query at node 2 (position 2.0)
        result = interp_nd_triweight(grid_values, axes, edges, (2.0,))

        # Should be close to [3.0, 30.0, 300.0] (with triweight kernel spread)
        import numpy.testing as npt

        npt.assert_allclose(result, grid_values[2], rtol=0.15, atol=0.1)

    def test_query_between_nodes_smooth(self):
        """Querying between nodes gives smooth interpolation.

        Bounds test: smooth interpolation in interior.
        """

        # Simple 1D linear grid
        axes = (jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),)
        edges = (edges_for_grid(axes[0]),)
        grid_values = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )

        # Query between nodes 1 and 2 (at position 1.5)
        result = interp_nd_triweight(grid_values, axes, edges, (1.5,))

        # For linear values, should be close to [1.5, 1.5, 1.5]

        npt.assert_allclose(result, [1.5, 1.5, 1.5], rtol=0.2, atol=0.1)


# ── Tests: interp_nd_triweight() 2D case ──────────────────────────


class TestInterpNdTriweight2D:
    """Bounds tests: interp_nd_triweight() with 2D grid."""

    def test_constant_grid_interpolates_to_constant_2d(self):
        """2D interpolation of a constant grid returns exactly the constant.

        Limit test: subsumes the old 2D shape-only and finite-only tests.
        """
        axes = (
            jnp.linspace(0.0, 1.0, 5),
            jnp.linspace(0.0, 1.0, 4),
        )
        edges = (edges_for_grid(axes[0]), edges_for_grid(axes[1]))
        grid_values = jnp.full((5, 4, 3), -3.5)

        result = interp_nd_triweight(grid_values, axes, edges, (0.41, 0.63))

        chex.assert_shape(result, (3,))
        npt.assert_allclose(np.asarray(result), -3.5, rtol=1e-12)

    def test_query_at_grid_corner_2d(self):
        """Querying at a 2D grid corner returns approximately that corner's value.

        Bounds test: 2D grid point lookup.
        """

        axes = (
            jnp.array([0.0, 1.0, 2.0]),
            jnp.array([0.0, 1.0]),
        )
        edges = (edges_for_grid(axes[0]), edges_for_grid(axes[1]))
        # 3×2 grid
        grid_values = jnp.array(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0]],
            ]
        )

        # Query at corner (axes[0][1], axes[1][1]) = (1.0, 1.0)
        result = interp_nd_triweight(grid_values, axes, edges, (1.0, 1.0))

        # Should be close to [4.0, 40.0]

        npt.assert_allclose(result, [4.0, 40.0], rtol=0.2, atol=0.1)


# ── Tests: Gradient and JIT compatibility ─────────────────────────


class TestGradientAndJIT:
    """Bounds tests: Gradients and JIT compilation work correctly."""

    def test_interp_1d_gradient(self):
        """interp_nd_triweight is differentiable w.r.t. query position (gradient test)."""
        import numpy as np

        def fd_grad_1d(f, x: float, eps: float = 1e-4) -> float:
            """Central finite difference."""
            return float((f(x + eps) - f(x - eps)) / (2.0 * eps))

        axes = (jnp.linspace(0.0, 1.0, 10),)
        edges = (edges_for_grid(axes[0]),)
        grid_values = jnp.array([jnp.sin(axes[0]), jnp.cos(axes[0])]).T

        def loss(x):
            result = interp_nd_triweight(grid_values, axes, edges, (x,))
            return jnp.sum(result)

        grad_jax = float(jax.grad(loss)(0.5))
        grad_fd = fd_grad_1d(loss, 0.5)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="interp_nd_triweight: FD check ∂/∂x",
        )

    def test_interp_1d_jit_matches_eager(self):
        """interp_nd_triweight under jax.jit matches the eager result (rtol=1e-6)."""

        axes = (jnp.linspace(0.0, 1.0, 15),)
        edges = (edges_for_grid(axes[0]),)
        grid_values = jnp.array([jnp.sin(axes[0]), jnp.cos(axes[0]), axes[0] ** 2]).T

        def compute(x):
            return interp_nd_triweight(grid_values, axes, edges, (x,))

        chex.assert_trees_all_close(jax.jit(compute)(0.5), compute(0.5), rtol=1e-6)


# ── Tests: Edge cases and numerical stability ─────────────────────


class TestEdgeCasesAndStability:
    """Bounds tests: Edge cases and numerical robustness."""

    @pytest.mark.parametrize("scale", [1e-30, 1e30], ids=["underflow", "overflow"])
    def test_extreme_template_scales_stay_linear(self, tophat_filters, scale):
        """Photometry scales exactly linearly at extreme template amplitudes.

        Bounds test: under/overflow protection. Φ is linear in L_ν, so
        phot(s·1) must equal s·phot(1) with no loss of precision at
        s = 1e±30 — a stronger statement than the old finite-only checks.
        """

        wave = jnp.linspace(1000.0, 10000.0, 100)
        filter_waves, filter_trans = tophat_filters

        unit = preintegrate_grid(
            jnp.ones((2, 3, 100)), wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )
        scaled = preintegrate_grid(
            jnp.ones((2, 3, 100)) * scale,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
        )

        npt.assert_allclose(np.asarray(scaled.phot), np.asarray(unit.phot) * scale, rtol=1e-10)

    def test_narrow_filter_bandwidth(self):
        """Narrow filter (small bandwidth) is handled correctly.

        Bounds test: narrow spectral features.
        """

        template = jnp.ones((2, 3, 1000))
        wave = jnp.linspace(1000.0, 10000.0, 1000)
        # Very narrow filter (only 10 Angstrom wide)
        filter_waves = [jnp.linspace(4995.0, 5005.0, 20)]
        filter_trans = [jnp.ones(20)]

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        # Should handle narrow bandwidth
        chex.assert_tree_all_finite(result.phot)
        chex.assert_shape(result.phot, (2, 3, 1))

    def test_single_wavelength_per_filter(self):
        """Single wavelength per filter (delta function approx).

        Bounds test: degenerate filter bandwidth.
        """

        template = jnp.ones((2, 3, 100))
        wave = jnp.linspace(1000.0, 10000.0, 100)
        # Filter with single wavelength (delta approx)
        filter_waves = [jnp.array([5000.0])]
        filter_trans = [jnp.array([1.0])]

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        chex.assert_tree_all_finite(result.phot)

    def test_interp_query_outside_grid_range(self):
        """Interpolation query outside grid range is handled gracefully.

        Bounds test: extrapolation behavior.
        """

        axes = (jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),)
        edges = (edges_for_grid(axes[0]),)
        grid_values = jnp.ones((5, 2))

        # Query way outside range
        result = interp_nd_triweight(grid_values, axes, edges, (10.0,))

        chex.assert_tree_all_finite(result)


# ── Tests: slice_fixed_axes() ─────────────────────────────────────


class TestSliceFixedAxes:
    """Bounds tests: slice_fixed_axes() reduces grid dimensionality."""

    def test_slice_removes_axis(self, synthetic_template_3d, tophat_filters):
        """Slicing one axis reduces grid ndim by 1.

        Bounds test: dimensionality reduction.
        """

        from tengri.forward.precompute.grid import preintegrate_grid, slice_fixed_axes

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters
        n_met, n_age, _ = template.shape

        result = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            axes=(np.linspace(-2, 0, n_met), np.linspace(6, 10, n_age)),
        )
        assert result.phot.shape == (n_met, n_age, len(filter_waves))
        assert len(result.axes) == 2

        # Slice axis 0 (metallicity) at value -1.0
        sliced = slice_fixed_axes(result, {0: -1.0})
        assert sliced.phot.shape == (n_age, len(filter_waves))
        assert len(sliced.axes) == 1

    def test_slice_preserves_values(self, synthetic_template_3d, tophat_filters):
        """Slicing at a grid node gives approximately that node's values.

        Bounds test: interpolation consistency.
        """

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters
        n_met, n_age, _ = template.shape
        met_grid = np.linspace(-2, 0, n_met)

        result = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            axes=(met_grid, np.linspace(6, 10, n_age)),
        )

        # Slice at the middle grid node
        mid_idx = n_met // 2
        sliced = slice_fixed_axes(result, {0: float(met_grid[mid_idx])})

        # Should approximate the middle slice of the original
        npt.assert_allclose(sliced.phot, result.phot[mid_idx], rtol=0.15)

    def test_slice_multiple_axes(self, tophat_filters):
        """Slicing multiple axes at once works.

        Bounds test: multi-axis slicing.
        """

        filter_waves, filter_trans = tophat_filters
        n_a, n_b, n_c, n_wave = 4, 3, 5, 200
        template = np.random.default_rng(42).uniform(1e-20, 1e-18, (n_a, n_b, n_c, n_wave))
        wave = np.linspace(3000, 8000, n_wave)
        ax_a = np.linspace(0, 1, n_a)
        ax_b = np.linspace(0, 1, n_b)
        ax_c = np.linspace(0, 1, n_c)

        result = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            axes=(ax_a, ax_b, ax_c),
        )
        assert result.phot.shape == (n_a, n_b, n_c, len(filter_waves))

        # Slice axes 0 and 2 simultaneously
        sliced = slice_fixed_axes(result, {0: 0.5, 2: 0.5})
        assert sliced.phot.shape == (n_b, len(filter_waves))
        assert len(sliced.axes) == 1
        chex.assert_tree_all_finite(sliced.phot)

    def test_empty_fixed_returns_same(self, synthetic_template_3d, tophat_filters):
        """Empty fixed dict returns the same grid.

        Bounds test: identity operation.
        """

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            axes=(np.linspace(-2, 0, template.shape[0]),),
        )
        sliced = slice_fixed_axes(result, {})
        assert sliced is result
