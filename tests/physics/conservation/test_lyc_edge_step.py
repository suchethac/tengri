# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: the Lyman-edge step model splits a grid exactly (#lyc-L1).

``tengri.components.lyc.edge_trapezoid`` and ``edge_interp`` implement one
step model of the grid cell straddling the physical Lyman limit
(911.76 A). The defining conservation law this file pins is additivity:
for ANY wavelength grid and edge position (interior, exactly on a node, or
off the grid on either side), the ionizing-side integral plus the
non-ionizing-side integral equals the whole-grid integral to round-off --
no photon is double-counted or dropped at the boundary cell.

Around that additivity core, this file also pins the closed-form bracket
formulas the module docstring states (a piecewise-constant/linear
reference implemented independently, as an explicit Python loop rather
than ``edge_trapezoid``'s vectorized masks, so a shared implementation bug
cannot pass both), ``edge_interp``'s step values in the bracket cell, and
that the primitive stays JIT/grad/vmap-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.lyc import LYMAN_LIMIT_AA, edge_interp, edge_trapezoid
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.conservation


def _coord(x, variable):
    return C_AA / x if variable == "nu" else x


def _reference_edge_integral(wave, y, edge_aa, variable):
    """Independent, explicit-loop reference for the step-model integral.

    Deliberately not vectorized and not sharing any code with
    :func:`edge_trapezoid`, so a shared off-by-one or comparison-operator bug
    could not pass both. Implements exactly the module docstring's formula:
    ordinary trapezoid away from the bracket cell, the two step-model
    rectangles inside it.
    """
    n = len(wave)
    ionizing_total = 0.0
    nonionizing_total = 0.0
    for i in range(n - 1):
        wa, wb = wave[i], wave[i + 1]
        ya, yb = y[i], y[i + 1]
        xa, xb = _coord(wa, variable), _coord(wb, variable)
        if wb < edge_aa:
            ionizing_total += 0.5 * (ya + yb) * abs(xb - xa)
        elif wa >= edge_aa:
            nonionizing_total += 0.5 * (ya + yb) * abs(xb - xa)
        else:
            x_edge = _coord(edge_aa, variable)
            ionizing_total += ya * abs(x_edge - xa)
            nonionizing_total += yb * abs(xb - x_edge)
    return ionizing_total, nonionizing_total


class TestAdditivity:
    """ionizing + nonionizing == all, for any grid and any edge position."""

    @pytest.mark.parametrize("variable", ["wave", "nu"])
    @pytest.mark.parametrize("seed", range(25))
    def test_ionizing_plus_nonionizing_equals_all(self, variable, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(4, 30)
        wave = jnp.asarray(np.sort(rng.uniform(50.0, 20000.0, size=n)))
        y = jnp.asarray(rng.uniform(-5.0, 5.0, size=n))
        # Random edge, sometimes off-grid on either side.
        edge = rng.choice(
            [
                rng.uniform(float(wave[0]), float(wave[-1])),
                rng.uniform(1.0, float(wave[0])),
                rng.uniform(float(wave[-1]), float(wave[-1]) * 2),
            ]
        )
        ion = edge_trapezoid(y, wave, variable=variable, side="ionizing", edge_aa=edge)
        non = edge_trapezoid(y, wave, variable=variable, side="nonionizing", edge_aa=edge)
        allv = edge_trapezoid(y, wave, variable=variable, side="all", edge_aa=edge)
        assert np.allclose(float(ion) + float(non), float(allv), rtol=1e-10, atol=1e-12)

    def test_default_edge_is_the_physical_lyman_limit(self):
        wave = jnp.linspace(500.0, 1500.0, 11)
        y = jnp.ones_like(wave)
        default = edge_trapezoid(y, wave, variable="wave", side="ionizing")
        explicit = edge_trapezoid(
            y, wave, variable="wave", side="ionizing", edge_aa=LYMAN_LIMIT_AA
        )
        assert float(default) == float(explicit)
        assert pytest.approx(911.76) == LYMAN_LIMIT_AA


class TestAnalyticStepModel:
    """Closed-form bracket formulas, checked against an independent reference."""

    @pytest.mark.parametrize("variable", ["wave", "nu"])
    @pytest.mark.parametrize("seed", range(15))
    def test_matches_independent_reference_random_grids(self, variable, seed):
        rng = np.random.default_rng(1000 + seed)
        n = rng.integers(5, 25)
        wave = np.sort(rng.uniform(50.0, 20000.0, size=n))
        y = rng.uniform(-3.0, 3.0, size=n)
        edge = rng.uniform(float(wave[0]) - 200.0, float(wave[-1]) + 200.0)

        ref_ion, ref_non = _reference_edge_integral(wave, y, edge, variable)
        ion = float(
            edge_trapezoid(
                jnp.asarray(y), jnp.asarray(wave), variable=variable, side="ionizing", edge_aa=edge
            )
        )
        non = float(
            edge_trapezoid(
                jnp.asarray(y),
                jnp.asarray(wave),
                variable=variable,
                side="nonionizing",
                edge_aa=edge,
            )
        )
        assert ion == pytest.approx(ref_ion, rel=1e-10, abs=1e-12)
        assert non == pytest.approx(ref_non, rel=1e-10, abs=1e-12)

    def test_piecewise_constant_step_matches_closed_form_in_wave(self):
        # y is exactly the step model's own shape: L_a below the edge, L_b at
        # or above it, sampled on an irregular grid with several nodes per side.
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        edge = LYMAN_LIMIT_AA
        L_a, L_b = 3.0, 0.4
        y = jnp.where(wave < edge, L_a, L_b)
        ion = float(edge_trapezoid(y, wave, variable="wave", side="ionizing", edge_aa=edge))
        non = float(edge_trapezoid(y, wave, variable="wave", side="nonionizing", edge_aa=edge))
        assert ion == pytest.approx(L_a * (edge - float(wave[0])), rel=1e-12)
        assert non == pytest.approx(L_b * (float(wave[-1]) - edge), rel=1e-12)

    def test_piecewise_constant_step_matches_closed_form_in_nu(self):
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        edge = LYMAN_LIMIT_AA
        L_a, L_b = 3.0, 0.4
        y = jnp.where(wave < edge, L_a, L_b)
        ion = float(edge_trapezoid(y, wave, variable="nu", side="ionizing", edge_aa=edge))
        non = float(edge_trapezoid(y, wave, variable="nu", side="nonionizing", edge_aa=edge))
        nu_0 = C_AA / float(wave[0])
        nu_edge = C_AA / edge
        nu_last = C_AA / float(wave[-1])
        assert ion == pytest.approx(L_a * abs(nu_0 - nu_edge), rel=1e-12)
        assert non == pytest.approx(L_b * abs(nu_edge - nu_last), rel=1e-12)

    def test_node_exactly_at_the_edge(self):
        wave = jnp.asarray([700.0, 905.0, LYMAN_LIMIT_AA, 1200.0, 5000.0])
        y = jnp.asarray([2.0, 2.0, 1.0, 1.0, 1.0])
        ion = float(
            edge_trapezoid(y, wave, variable="wave", side="ionizing", edge_aa=LYMAN_LIMIT_AA)
        )
        non = float(
            edge_trapezoid(y, wave, variable="wave", side="nonionizing", edge_aa=LYMAN_LIMIT_AA)
        )
        # A node exactly at the edge belongs to the non-ionizing side (module
        # docstring: mask is strict wave < edge_aa), so the bracket rectangle's
        # non-ionizing width collapses to zero and the whole cell counts as
        # ionizing at the last-ionizing-node value.
        assert ion == pytest.approx(2.0 * (LYMAN_LIMIT_AA - 700.0), rel=1e-12)
        assert non == pytest.approx(1.0 * (5000.0 - LYMAN_LIMIT_AA), rel=1e-12)

    def test_edge_below_entire_grid_is_all_nonionizing(self):
        wave = jnp.asarray([1000.0, 2000.0, 3000.0])
        y = jnp.asarray([1.0, 2.0, 1.5])
        edge = 500.0  # below wave[0]
        ion = float(edge_trapezoid(y, wave, variable="wave", side="ionizing", edge_aa=edge))
        non = float(edge_trapezoid(y, wave, variable="wave", side="nonionizing", edge_aa=edge))
        allv = float(edge_trapezoid(y, wave, variable="wave", side="all", edge_aa=edge))
        assert ion == 0.0
        assert non == pytest.approx(allv, rel=1e-12)
        assert non == pytest.approx(float(jnp.trapezoid(y, wave)), rel=1e-12)

    def test_edge_above_entire_grid_is_all_ionizing(self):
        wave = jnp.asarray([100.0, 200.0, 300.0])
        y = jnp.asarray([1.0, 2.0, 1.5])
        edge = 5000.0  # above wave[-1]
        ion = float(edge_trapezoid(y, wave, variable="wave", side="ionizing", edge_aa=edge))
        non = float(edge_trapezoid(y, wave, variable="wave", side="nonionizing", edge_aa=edge))
        allv = float(edge_trapezoid(y, wave, variable="wave", side="all", edge_aa=edge))
        assert non == 0.0
        assert ion == pytest.approx(allv, rel=1e-12)
        assert ion == pytest.approx(float(jnp.trapezoid(y, wave)), rel=1e-12)


class TestVariableConsistency:
    """The ``variable`` argument genuinely switches the quadrature coordinate."""

    def test_wave_and_nu_variables_give_different_results_in_general(self):
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        y = jnp.asarray([5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5])
        by_wave = float(edge_trapezoid(y, wave, variable="wave", side="all"))
        by_nu = float(edge_trapezoid(y, wave, variable="nu", side="all"))
        assert not np.isclose(by_wave, by_nu)

    def test_both_variables_recover_the_exact_closed_form_on_the_step_shape(self):
        # Complements test_piecewise_constant_step_matches_closed_form_{in_wave,in_nu}:
        # both quadrature variables are internally consistent with the SAME
        # underlying step model, each exact in its own coordinate.
        wave = jnp.asarray([200.0, 850.0, 950.0, 4000.0])
        edge = LYMAN_LIMIT_AA
        L_a, L_b = 7.0, 1.0
        y = jnp.where(wave < edge, L_a, L_b)
        ion_wave = float(edge_trapezoid(y, wave, variable="wave", side="ionizing", edge_aa=edge))
        ion_nu = float(edge_trapezoid(y, wave, variable="nu", side="ionizing", edge_aa=edge))
        assert ion_wave == pytest.approx(L_a * (edge - 200.0), rel=1e-12)
        assert ion_nu == pytest.approx(L_a * abs(C_AA / 200.0 - C_AA / edge), rel=1e-12)


class TestEdgeInterp:
    """Step values in the bracket cell; ordinary linear interpolation elsewhere."""

    def test_step_in_bracket_cell_linear_elsewhere(self):
        wave = jnp.asarray([700.0, 905.0, 918.0, 1200.0])
        y = jnp.asarray([4.0, 4.0, 1.0, 1.0])
        edge = LYMAN_LIMIT_AA
        # Inside [700, 905]: both grid values equal 4.0, so linear == step.
        assert float(edge_interp(jnp.asarray(800.0), wave, y, edge_aa=edge)) == pytest.approx(4.0)
        # Inside the bracket cell [905, 918]: must be the STEP, not a ramp.
        just_below = float(edge_interp(jnp.asarray(910.0), wave, y, edge_aa=edge))
        just_above = float(edge_interp(jnp.asarray(913.0), wave, y, edge_aa=edge))
        assert just_below == pytest.approx(4.0)
        assert just_above == pytest.approx(1.0)
        # At the grid nodes themselves.
        assert float(edge_interp(jnp.asarray(905.0), wave, y, edge_aa=edge)) == pytest.approx(4.0)
        assert float(edge_interp(jnp.asarray(918.0), wave, y, edge_aa=edge)) == pytest.approx(1.0)
        # At the edge itself: non-ionizing side per convention (x < edge test).
        assert float(edge_interp(jnp.asarray(edge), wave, y, edge_aa=edge)) == pytest.approx(1.0)

    def test_zero_outside_grid_like_jnp_interp(self):
        wave = jnp.asarray([700.0, 905.0, 918.0, 1200.0])
        y = jnp.asarray([4.0, 4.0, 1.0, 1.0])
        assert float(edge_interp(jnp.asarray(1.0), wave, y)) == 0.0
        assert float(edge_interp(jnp.asarray(1e6), wave, y)) == 0.0

    def test_edge_off_grid_falls_back_to_plain_linear(self):
        wave = jnp.asarray([1000.0, 2000.0, 3000.0])
        y = jnp.asarray([1.0, 3.0, 2.0])
        edge = 1.0  # below the entire grid: no bracket cell exists
        plain = jnp.interp(jnp.asarray(1500.0), wave, y, left=0.0, right=0.0)
        stepped = edge_interp(jnp.asarray(1500.0), wave, y, edge_aa=edge)
        assert float(stepped) == pytest.approx(float(plain))


class TestGradients:
    """Gradients through edge_trapezoid / edge_interp are finite and exact."""

    def test_edge_trapezoid_grad_matches_finite_differences(self):
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        y0 = jnp.asarray([5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5])

        def f(y):
            return edge_trapezoid(y, wave, variable="nu", side="ionizing", edge_aa=LYMAN_LIMIT_AA)

        grad = jax.grad(f)(y0)
        assert jnp.all(jnp.isfinite(grad))

        eps = 1e-6
        fd = np.array(
            [
                float((f(y0.at[i].add(eps)) - f(y0.at[i].add(-eps))) / (2 * eps))
                for i in range(y0.shape[0])
            ]
        )
        assert np.allclose(np.asarray(grad), fd, rtol=1e-5, atol=1e-6)

    def test_edge_interp_grad_matches_finite_differences(self):
        wave = jnp.asarray([700.0, 905.0, 918.0, 1200.0])
        y0 = jnp.asarray([4.0, 4.0, 1.0, 1.0])
        x_new = jnp.asarray(913.0)  # inside the bracket cell

        def f(y):
            return edge_interp(x_new, wave, y, edge_aa=LYMAN_LIMIT_AA)

        grad = jax.grad(f)(y0)
        assert jnp.all(jnp.isfinite(grad))

        eps = 1e-6
        fd = np.array(
            [
                float((f(y0.at[i].add(eps)) - f(y0.at[i].add(-eps))) / (2 * eps))
                for i in range(y0.shape[0])
            ]
        )
        assert np.allclose(np.asarray(grad), fd, rtol=1e-5, atol=1e-6)


class TestJitVmap:
    """JIT-compiled and vmapped calls match plain eager evaluation."""

    def test_jit_matches_eager(self):
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        y = jnp.asarray([5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5])

        eager = edge_trapezoid(y, wave, variable="nu", side="ionizing", edge_aa=LYMAN_LIMIT_AA)
        jitted_fn = jax.jit(
            lambda yy: edge_trapezoid(
                yy, wave, variable="nu", side="ionizing", edge_aa=LYMAN_LIMIT_AA
            )
        )
        assert float(jitted_fn(y)) == pytest.approx(float(eager), rel=1e-12)

    def test_vmap_matches_per_row_eager(self):
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        ys = jnp.stack(
            [
                jnp.asarray([5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5]),
                jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            ]
        )
        vmapped = jax.vmap(
            lambda yy: edge_trapezoid(
                yy, wave, variable="wave", side="ionizing", edge_aa=LYMAN_LIMIT_AA
            )
        )(ys)
        per_row = jnp.stack(
            [
                edge_trapezoid(
                    ys[i], wave, variable="wave", side="ionizing", edge_aa=LYMAN_LIMIT_AA
                )
                for i in range(ys.shape[0])
            ]
        )
        assert jnp.allclose(vmapped, per_row, rtol=1e-12)

    def test_edge_trapezoid_axis_argument(self):
        # y has an extra leading batch axis; axis=-1 integrates the last one.
        wave = jnp.asarray([100.0, 300.0, 700.0, 905.0, 918.0, 1200.0, 5000.0])
        ys = jnp.stack(
            [
                jnp.asarray([5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5]),
                jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ]
        )
        batched = edge_trapezoid(
            ys, wave, variable="wave", side="ionizing", edge_aa=LYMAN_LIMIT_AA, axis=-1
        )
        assert batched.shape == (2,)
        for i in range(2):
            single = edge_trapezoid(
                ys[i], wave, variable="wave", side="ionizing", edge_aa=LYMAN_LIMIT_AA
            )
            assert float(batched[i]) == pytest.approx(float(single), rel=1e-12)
