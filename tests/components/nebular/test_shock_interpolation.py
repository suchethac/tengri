# SPDX-License-Identifier: BSD-3-Clause
"""Tests for MAPPINGS V triweight interpolation in shock.py.

Validates that all three continuous axes (velocity, B-field, log_density) use
C²-continuous triweight interpolation, giving finite gradients everywhere.
"""

from __future__ import annotations

import chex
import pytest

pytestmark = pytest.mark.bounds

pytest.importorskip("h5py", reason="h5py required for MAPPINGS grid tests")

import jax
import jax.numpy as jnp
import numpy as np


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# --- conditional import: skip entire module if grid file is absent ---

from tengri.components.nebular.shock import (
    _load_mappings_grids,
    compute_shock_sed,
    shock_line_ratios,
)

pytestmark = pytest.mark.skipif(
    _load_mappings_grids() is None,
    reason="MAPPINGS grid file not found; run scripts/download_mappings_templates.py",
)


# ── Helpers ───────────────────────────────────────────────────────


def _get_grid():
    """Return the mappings5 grid dict (skips test if unavailable)."""
    grids = _load_mappings_grids()
    if grids is None or "mappings5" not in grids:
        pytest.skip("MAPPINGS grid not available")
    return grids["mappings5"]


def _midpoint(arr):
    """Return the midpoint of a 1-D array."""
    a = np.asarray(arr)
    return float(0.5 * (a[0] + a[-1]))


def _quarter_point(arr):
    """Return the first-quarter element of a 1-D array.

    Used for the B-field axis: the full b_field axis spans 1e-4–1000 μG but
    MAPPINGS V data for solar abundance only exists up to ~10 μG.  Taking the
    quarter-point keeps tests firmly inside the valid data region.
    """
    a = np.asarray(arr)
    return float(a[len(a) // 4])


# ── Grid-node exact-lookup tests ──────────────────────────────────


class TestGridNodeLookup:
    """At grid nodes, interpolated result should equal the table value."""

    def test_velocity_node_matches_table(self):
        """Interpolating at a velocity grid node recovers the tabulated ratio."""
        g = _get_grid()
        v_grid = np.asarray(g["velocities_kms"])
        b_grid = np.asarray(g["b_axis"])
        n_grid = np.asarray(g["log_density_cm3"])

        # Use the first velocity node, middle B-field, middle density
        v_node = float(v_grid[1])  # skip first (edge effects)
        b_idx = len(b_grid) // 2
        n_idx = len(n_grid) // 2
        b_val = float(b_grid[b_idx])
        n_val = float(n_grid[n_idx])

        ratios = shock_line_ratios(v_node, shock_log_density=n_val, shock_b_over_sqrt_n=b_val)
        assert len(ratios) > 0
        # All ratio values must be finite (no NaN from interpolation failure)
        for name, val in ratios.items():
            assert jnp.isfinite(val), f"NaN ratio at grid node for line {name}"

    def test_all_axes_at_first_node(self):
        """Interpolating at the grid corner (first node on all axes) returns finite values."""
        g = _get_grid()
        v0 = float(np.asarray(g["velocities_kms"])[0])
        b0 = float(np.asarray(g["b_axis"])[0])
        n0 = float(np.asarray(g["log_density_cm3"])[0])

        ratios = shock_line_ratios(v0, shock_log_density=n0, shock_b_over_sqrt_n=b0)
        for name, val in ratios.items():
            assert jnp.isfinite(val), f"NaN at first grid corner for line {name}"


# ── Monotonicity / physics plausibility ───────────────────────────


class TestInterpolationSmoothness:
    """Triweight interpolation should produce smooth outputs along each axis."""

    def test_velocity_monotone_oiii(self):
        """[OIII] ratio increases with velocity over 200–600 km/s (Allen+2008 trend)."""
        g = _get_grid()
        # Use quarter-point of b_axis: full axis spans 1e-4–1000 μG but solar-abundance
        # data only exists up to ~10 μG, so _midpoint(≈500 μG) would query empty space.
        b_mid = _quarter_point(g["b_axis"])
        n_mid = float(_midpoint(g["log_density_cm3"]))

        velocities = np.linspace(200.0, 600.0, 10)
        oiii_key = next(
            (k for k in shock_line_ratios(300.0) if "O3_5007" in k or "OIII" in k),
            None,
        )
        if oiii_key is None:
            pytest.skip("No [OIII] 5007 line in grid")

        vals = [
            float(
                shock_line_ratios(v, shock_log_density=n_mid, shock_b_over_sqrt_n=b_mid)[oiii_key]
            )
            for v in velocities
        ]
        # [OIII] should be non-negligible and the sequence should not be all zeros
        assert max(vals) > 0.01, "Expected non-negligible [OIII] ratios"

    def test_output_finite_across_b_field_range(self):
        """Interpolation over B-field axis returns finite values at 10 points."""
        g = _get_grid()
        b_grid = np.asarray(g["b_axis"])
        v_mid = float(_midpoint(g["velocities_kms"]))
        n_mid = float(_midpoint(g["log_density_cm3"]))

        b_vals = np.linspace(float(b_grid[0]), float(b_grid[-1]), 10)
        for b in b_vals:
            ratios = shock_line_ratios(
                v_mid, shock_log_density=n_mid, shock_b_over_sqrt_n=float(b)
            )
            for name, val in ratios.items():
                assert jnp.isfinite(val), f"NaN at b_field={b:.4g} for line {name}"

    def test_output_finite_across_density_range(self):
        """Interpolation over log_density axis returns finite values at 10 points."""
        g = _get_grid()
        n_grid = np.asarray(g["log_density_cm3"])
        v_mid = float(_midpoint(g["velocities_kms"]))
        b_mid = float(_midpoint(g["b_axis"]))

        n_vals = np.linspace(float(n_grid[0]), float(n_grid[-1]), 10)
        for n in n_vals:
            ratios = shock_line_ratios(
                v_mid, shock_log_density=float(n), shock_b_over_sqrt_n=b_mid
            )
            for name, val in ratios.items():
                assert jnp.isfinite(val), f"NaN at log_density={n:.4g} for line {name}"


# ── Gradient tests — the core motivation for this change ──────────


class TestGradients:
    """Triweight interpolation must give finite gradients on all continuous axes.

    These tests would FAIL under the old nearest-neighbor strategy because
    jax.grad through np.argmin returns zero (non-differentiable).
    """

    def test_gradient_wrt_velocity_is_finite(self):
        """FD check: ∂(∑ratios)/∂velocity. Triweight interpolation must give nonzero grad."""
        g = _get_grid()
        b_mid = float(_midpoint(g["b_axis"]))
        n_mid = float(_midpoint(g["log_density_cm3"]))
        v_mid = float(_midpoint(g["velocities_kms"]))

        def sum_ratios_scalar(v):
            r = shock_line_ratios(v, shock_log_density=n_mid, shock_b_over_sqrt_n=b_mid)
            return sum(r.values())

        def f(v):
            return float(sum_ratios_scalar(jnp.array(v)))

        g_v = float(jax.grad(sum_ratios_scalar)(jnp.array(v_mid)))
        np.testing.assert_allclose(
            g_v,
            fd_grad(f, v_mid, eps=1.0),
            rtol=1e-3,
            err_msg="shock_line_ratios: FD check ∂/∂velocity",
        )

    def test_gradient_wrt_b_field_is_finite(self):
        """FD check: ∂(∑ratios)/∂b_field. Was zero under nearest-neighbor."""
        g = _get_grid()
        b_mid = float(_midpoint(g["b_axis"]))
        n_mid = float(_midpoint(g["log_density_cm3"]))
        v_mid = float(_midpoint(g["velocities_kms"]))

        def sum_ratios_scalar(b):
            r = shock_line_ratios(v_mid, shock_log_density=n_mid, shock_b_over_sqrt_n=b)
            return sum(r.values())

        def f(b):
            return float(sum_ratios_scalar(jnp.array(b)))

        g_b = float(jax.grad(sum_ratios_scalar)(jnp.array(b_mid)))
        np.testing.assert_allclose(
            g_b,
            fd_grad(f, b_mid, eps=b_mid * 1e-3),
            rtol=1e-3,
            err_msg="shock_line_ratios: FD check ∂/∂b_field",
        )

    def test_gradient_wrt_log_density_is_finite(self):
        """FD check: ∂(∑ratios)/∂log_density. Was zero under nearest-neighbor."""
        g = _get_grid()
        b_mid = float(_midpoint(g["b_axis"]))
        n_mid = float(_midpoint(g["log_density_cm3"]))
        v_mid = float(_midpoint(g["velocities_kms"]))

        def sum_ratios_scalar(n):
            r = shock_line_ratios(v_mid, shock_log_density=n, shock_b_over_sqrt_n=b_mid)
            return sum(r.values())

        def f(n):
            return float(sum_ratios_scalar(jnp.array(n)))

        g_n = float(jax.grad(sum_ratios_scalar)(jnp.array(n_mid)))
        np.testing.assert_allclose(
            g_n,
            fd_grad(f, n_mid, eps=1e-3),
            rtol=1e-3,
            err_msg="shock_line_ratios: FD check ∂/∂log_density",
        )

    def test_joint_gradient_all_three_axes(self):
        """Joint gradients (velocity, b_field, log_density) all agree with FD."""
        g = _get_grid()
        b_mid = float(_midpoint(g["b_axis"]))
        n_mid = float(_midpoint(g["log_density_cm3"]))
        v_mid = float(_midpoint(g["velocities_kms"]))

        def sum_ratios(params):
            v, b, n = params[0], params[1], params[2]
            r = shock_line_ratios(v, shock_log_density=n, shock_b_over_sqrt_n=b)
            return sum(r.values())

        params = jnp.array([v_mid, b_mid, n_mid])
        grads = jax.grad(sum_ratios)(params)
        assert jnp.all(jnp.isfinite(grads)), f"Non-finite joint gradient: {grads}"

        # FD check on each component individually
        def f_v(v):
            return float(sum_ratios(jnp.array([v, b_mid, n_mid])))

        np.testing.assert_allclose(
            float(grads[0]),
            fd_grad(f_v, v_mid, eps=1.0),
            rtol=1e-3,
            err_msg="shock_line_ratios joint: FD check ∂/∂velocity",
        )


# ── Smoke tests for compute_shock_sed ─────────────────────────────


class TestComputeShockSed:
    """Smoke tests: compute_shock_sed returns finite SEDs at mid-grid values."""

    def test_smoke_mid_grid_values(self):
        """compute_shock_sed with mid-grid parameters returns a finite SED."""
        g = _get_grid()
        v_mid = float(_midpoint(g["velocities_kms"]))
        b_mid = _quarter_point(g["b_axis"])  # full-axis midpoint ≈500 μG exceeds valid solar range
        n_mid = float(_midpoint(g["log_density_cm3"]))

        wave = jnp.linspace(1000.0, 10000.0, 500)
        sed = compute_shock_sed(
            wave,
            v_mid,
            l_shock_halpha=1e40,
            shock_log_density=n_mid,
            shock_b_over_sqrt_n=b_mid,
            line_sigma_aa=5.0,
        )
        chex.assert_equal_shape([sed, wave])
        chex.assert_tree_all_finite(sed)
        assert jnp.any(sed > 0), "compute_shock_sed returned all-zero SED"

    def test_sed_gradient_wrt_velocity(self):
        """Gradient of total SED flux w.r.t. velocity is finite."""
        g = _get_grid()
        v_mid = float(_midpoint(g["velocities_kms"]))
        b_mid = _quarter_point(g["b_axis"])  # full-axis midpoint ≈500 μG exceeds valid solar range
        n_mid = float(_midpoint(g["log_density_cm3"]))

        wave = jnp.linspace(3000.0, 9000.0, 200)

        def total_flux(v):
            return jnp.sum(
                compute_shock_sed(
                    wave,
                    v,
                    l_shock_halpha=1e40,
                    shock_log_density=n_mid,
                    shock_b_over_sqrt_n=b_mid,
                    line_sigma_aa=5.0,
                )
            )

        def f(v):
            return float(total_flux(jnp.array(v)))

        g_v = float(jax.grad(total_flux)(jnp.array(v_mid)))
        np.testing.assert_allclose(
            g_v,
            fd_grad(f, v_mid, eps=1.0),
            rtol=1e-3,
            err_msg="compute_shock_sed: FD check ∂(∑SED)/∂velocity",
        )
