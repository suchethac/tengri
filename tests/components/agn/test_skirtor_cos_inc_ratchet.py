# SPDX-License-Identifier: BSD-3-Clause
"""Ratchet test for SKIRTOR cos_inclination non-uniform axis (#1851).

The cos_inclination axis in SKIRTOR is non-uniform (cos of uniform-in-angle
nodes, spacing ratio ~11x). It exhibits the #1851 degeneracy: nearest-neighbor-
like behavior over ~67.5% of the range.

This test ratchets the defect to prevent silent regression: a 40-point sweep
over cos_inc on the real SKIRTOR grid must yield 40 distinct outputs and
0% exactly-zero gradients (the #1851 pattern). Currently the test xfails
because SKIRTOR uses the legacy physical-space path; re-baselining to the
corrected index-space path requires validation of silicate features, goldens,
and parity tests against CIGALE (see #1911).

The test will flip to a passing green mark once that follow-up is resolved
and index_space_interp=True is wired into the SKIRTOR interpolation call.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Locate the SKIRTOR grid, or skip if missing
_SKIRTOR_GRID_DIR = Path(__file__).resolve().parents[3] / "data"
_SKIRTOR_GRID_FILES = list(_SKIRTOR_GRID_DIR.glob("skirtor*_*.h5"))
_has_skirtor = len(_SKIRTOR_GRID_FILES) > 0


@pytest.mark.skipif(not _has_skirtor, reason="SKIRTOR grid not found")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "SKIRTOR cos_inclination exhibits #1851 degeneracy (nearest-neighbor-like "
        "behavior over ~67.5% of range, 0% distinct outputs in sweep). "
        "Re-baselining requires validation of silicate features, goldens, "
        "and CIGALE parity (see #1911)."
    ),
)
def test_skirtor_cos_inc_ratchet_40_point_sweep():
    """SKIRTOR cos_inc: 40-point sweep should yield 40 distinct outputs.

    Currently xfails (the #1851 degeneracy). Flip to passing green once
    SKIRTOR is migrated to index-space interpolation and tests are re-baselined.
    """
    from tengri.components.agn.skirtor import load_skirtor_grid

    # Load one SKIRTOR grid variant (all have the same cos_inc axis)
    grid_path = str(_SKIRTOR_GRID_FILES[0])
    grid = load_skirtor_grid(grid_path)

    # Find the cos_inclination axis (last in the tuple: tau, p, q, oa, radius, inc)
    axes = grid.axes
    cos_inc_axis = axes[-1]  # inclination axis

    # Create a dummy grid for interpolation (vary only cos_inc, fix others)
    grid_data = grid.grid
    edges = grid.edges

    # Create a single query point: sweep only cos_inc, fix other dimensions at midpoints
    fixed_indices = [len(ax) // 2 for ax in axes[:-1]]  # Midpoint for all but cos_inc
    tau_idx, p_idx, q_idx, oa_idx, radius_idx = fixed_indices

    # Slice out a 1-D trace: grid[tau_idx, p_idx, q_idx, oa_idx, radius_idx, :, :]
    # (the last dimension is wavelength, which we interpolate over)
    trace_grid = grid_data[tau_idx, p_idx, q_idx, oa_idx, radius_idx, :, :]  # (n_inc, n_wave)

    # 40-point sweep over cos_inc
    cos_inc_sweep = jnp.linspace(float(cos_inc_axis[0]), float(cos_inc_axis[-1]), 40)

    from tengri.utils.interpolation import compute_grid_weights

    # Interpolate at each cos_inc, summing over wavelength
    results = []
    for cos_inc_val in cos_inc_sweep:
        scatter = 0.5 * (cos_inc_axis[1] - cos_inc_axis[0])
        w = compute_grid_weights(
            cos_inc_val,
            cos_inc_axis,
            scatter=scatter,
            edges=edges[-1],  # cos_inc axis edges
            index_space_interp=None,  # Use default (physical-space) to match current behavior
        )
        interp_val = jnp.dot(w, trace_grid)  # Shape (n_wave,)
        results.append(interp_val)

    results_array = jnp.array(results)  # (40, n_wave)

    # Count distinct outputs (rounding to 10 decimal places to account for float noise)
    distinct_count = len(np.unique(np.round(results_array, 10), axis=0))

    assert distinct_count == 40, (
        f"cos_inc sweep should yield 40 distinct outputs; got {distinct_count}. "
        f"This indicates the #1851 nearest-neighbor degeneracy."
    )


@pytest.mark.skipif(not _has_skirtor, reason="SKIRTOR grid not found")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "SKIRTOR cos_inclination exhibits #1851 degeneracy (0% exactly-zero gradients "
        "over the full range would be corrected by index-space interpolation). "
        "See #1911 for the validation plan."
    ),
)
def test_skirtor_cos_inc_ratchet_zero_gradient_fraction():
    """SKIRTOR cos_inc: gradient should be nonzero throughout.

    Currently xfails (the #1851 degeneracy produces ~67.5% exactly-zero gradients).
    Flip to passing green once SKIRTOR is migrated to index-space interpolation.
    """
    from tengri.components.agn.skirtor import load_skirtor_grid

    grid_path = str(_SKIRTOR_GRID_FILES[0])
    grid = load_skirtor_grid(grid_path)

    axes = grid.axes
    cos_inc_axis = axes[-1]
    grid_data = grid.grid
    edges = grid.edges

    # Single-parameter trace (fix all but cos_inc)
    fixed_indices = [len(ax) // 2 for ax in axes[:-1]]
    tau_idx, p_idx, q_idx, oa_idx, radius_idx = fixed_indices
    trace_grid = grid_data[tau_idx, p_idx, q_idx, oa_idx, radius_idx, :, :]

    cos_inc_sweep = jnp.linspace(float(cos_inc_axis[0]), float(cos_inc_axis[-1]), 40)

    from tengri.utils.interpolation import compute_grid_weights

    @jax.jit
    def interp_sum(cos_inc_val):
        scatter = 0.5 * (cos_inc_axis[1] - cos_inc_axis[0])
        w = compute_grid_weights(
            cos_inc_val, cos_inc_axis, scatter=scatter, edges=edges[-1], index_space_interp=None
        )
        interp_val = jnp.dot(w, trace_grid)
        return jnp.sum(interp_val)

    grad_fn = jax.jit(jax.grad(interp_sum))
    grads = jnp.array([grad_fn(cos_inc) for cos_inc in cos_inc_sweep])

    # Count exactly-zero gradients
    zero_count = jnp.sum(grads == 0.0).item()
    zero_fraction = zero_count / len(cos_inc_sweep)

    assert zero_fraction == 0.0, (
        f"cos_inc sweep should have 0% exactly-zero gradients; got {zero_fraction:.1%}. "
        f"This indicates the #1851 kernel-support issue."
    )
