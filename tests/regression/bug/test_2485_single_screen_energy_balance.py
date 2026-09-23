# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for issue #2485: single-screen dust energy-balance LUT.

Single-screen dust attenuation models with free ``dust_tau_v`` should build the
energy-balance LUT under ``approx=WavePrecomp()``, avoiding the slow full-grid
``L_absorbed`` integral on every gradient call.

The LUT is built by treating single-screen as a degenerate two-component model:
``tau_bc = 0`` (no birth-cloud) and ``tau_diff = tau_v`` (diffuse-only), with
the same attenuation law on both. At runtime, the ``dust_tau_v`` parameter is
mapped to the ``tau_diff`` axis.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest

from tengri import Fixed, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug


def test_bug_2485_defect_single_screen_gets_lut(synthetic_ssp_wide, synthetic_tophat_obs):
    r"""Test that single-screen dust with free ``dust_tau_v`` builds the LUT.

    This is the **defect test**: before the fix, ``_energy_balance_lut_cache``
    remained ``None`` for single-component models because the class gate checked
    ``isinstance(c, DustSEDComponent)`` which only matched two-component, not
    ``DustAttenuationSEDComponent``.

    The test must FAIL on unpatched code (LUT is None), and PASS after the fix
    (LUT is built).
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "delayed", "log_total_mass": Fixed(10.0)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={"type": "dale2014", "T": Fixed(30.0)},
        redshift=Fixed(0.1),
        approx={"wave_precomp": True},
    )

    # Force LUT build
    lut = model._energy_balance_lut_cache_for_jit()

    # Before fix: LUT is None (slow full-grid path)
    # After fix: LUT is an EnergyBalanceLUT with tau_bc_grid=[0.0]
    assert lut is not None, (
        "Single-screen dust with free dust_tau_v should build the LUT. "
        "Before fix: LUT remains None (slow full-grid path). "
        "After fix: LUT is built (fast interpolated path)."
    )

    # Verify structure: single-component maps to degenerate tau_bc axis
    assert lut.tau_bc_grid.shape[0] == 1, (
        "Single-component should have degenerate tau_bc_grid=[0.0]"
    )
    assert float(lut.tau_bc_grid[0]) == 0.0, "Degenerate axis should be exactly 0.0"

    # tau_diff axis should have multiple nodes (free tau_v spans a grid)
    assert lut.tau_diff_grid.shape[0] > 1, "tau_diff grid should be non-trivial"

    # B and G should have compatible shapes
    assert lut.B.ndim == 2, "B should be (n_met, n_age)"
    assert lut.G.ndim == 4, "G should be (n_met, n_age, n_tau_bc, n_tau_diff)"
    assert lut.G.shape[2] == 1, "G's tau_bc dimension should be 1 (degenerate)"
    assert lut.G.shape[3] == lut.tau_diff_grid.shape[0], (
        "G's tau_diff dimension should match grid"
    )


def test_bug_2485_two_component_unchanged(synthetic_ssp_wide, synthetic_tophat_obs):
    r"""Test that two-component models still get the LUT.

    Ensures the fix does not regress two-component dust models.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "delayed", "log_total_mass": Fixed(10.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Uniform(0.0, 3.0),
        },
        dust_emission={"type": "dale2014", "T": Fixed(30.0)},
        redshift=Fixed(0.1),
        approx={"wave_precomp": True},
    )

    # LUT should be built
    lut = model._energy_balance_lut_cache_for_jit()
    assert lut is not None, "Two-component model should build the LUT"

    # Structure should be the normal (non-degenerate) two-component
    assert lut.tau_bc_grid.shape[0] > 1, (
        "Two-component should have non-degenerate tau_bc_grid"
    )
    assert lut.tau_diff_grid.shape[0] > 1, "Two-component should have tau_diff grid"


def test_bug_2485_refuse_lut_for_shape_changing_params(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    r"""Test that free curve-shape parameters still refuse the LUT.

    If ``dust_delta`` (or other curve-shape param) is free, the LUT must be
    skipped because the LUT bakes a specific curve at build time. This is
    fail-safe by design.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "delayed", "log_total_mass": Fixed(10.0)},
        dust_attenuation={
            "type": "single_component",
            "law": "li08",
            "tau_v": Fixed(0.5),
            "delta": Uniform(-0.5, 0.5),  # Free shape parameter
        },
        dust_emission={"type": "dale2014", "T": Fixed(30.0)},
        redshift=Fixed(0.1),
        approx={"wave_precomp": True},
    )

    # LUT should NOT be built (None) because a curve-shape parameter is free
    lut = model._energy_balance_lut_cache_for_jit()
    assert lut is None, (
        "Free dust_delta (curve-shape parameter) should refuse the LUT. "
        "This is fail-safe: any new free parameter defaults to NO LUT, "
        "and must be explicitly whitelisted."
    )
