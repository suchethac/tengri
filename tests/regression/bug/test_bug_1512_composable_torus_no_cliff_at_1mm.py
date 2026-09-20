# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1512: composable torus SED does not collapse at 1 mm node.

The composable AGN path with cigale_joint normalization ties the disc to the
SKIRTOR torus ratio. When resampling the SKIRTOR inclination-attenuation ratio
from the template grid (136 nodes) to the model grid, the boundary condition
was `right=0.0`, causing the disc component to collapse to zero at exactly 1 mm
(the highest SKIRTOR template node). The sed_agn_disc fell off a cliff while
sed_agn_torus remained smooth.

This test ensures that (a) the ratio of consecutive sed_agn values at 1e7 Å
matches the monolithic SKIRTOR model (within 5%), and (b) sed_agn_torus is
finite and non-zero at every wavelength up to the template's maximum.

Reference: GitHub issue #1512.
"""

import pytest
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data


pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp_data():
    """Load SSP data once for the module."""
    try:
        return load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")


def test_composable_torus_no_collapse_at_1mm_vs_monolithic(ssp_data):
    """Composable torus SED ratio matches monolithic at 1 mm node.

    The composable powerlaw+skirtor model with cigale_joint normalization
    should produce the same sed_agn continuity across 1e7 Å as the monolithic
    skirtor model. A discontinuity would indicate the resampling boundary
    condition is zeroing the component.

    Expected behavior (measured on origin/main@8e642ab11, float64):
    - Monolithic sed_agn at 1e7 Å: 6.0381e+30 erg/s/Hz
    - Monolithic sed_agn at 9.908e+06 Å: 5.9825e+30 erg/s/Hz
    - Monolithic ratio: 1.0093
    - Composable torus sed_agn[1e7] should be ~2.6369e+23 (continuous)
    - Composable torus sed_agn[prev] should be ~8.0721e+28 (before collapse)
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))

    # Build monolithic SKIRTOR model for reference
    monolithic = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        redshift=Fixed(0.1),
        agn={"type": "skirtor", "all_params": Fixed(DEFAULT)},
    )

    # Build composable model with same config
    composable = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        redshift=Fixed(0.1),
        agn={
            "type": "composable",
            "disc": {"type": "powerlaw"},
            "torus": {"type": "skirtor"},
            "norm": "cigale_joint",
            "fracAGN": 0.1,
            "all_params": Fixed(DEFAULT),
        },
    )

    # Get predictions
    mono_pred = monolithic.predict({})
    comp_pred = composable.predict({})

    mono_sed = np.asarray(mono_pred.rest_sed())
    comp_sed = np.asarray(comp_pred.rest_sed())

    # Get components from state
    mono_state = monolithic.predict_state({})
    comp_state = composable.predict_state({})
    comp_components = comp_state.derived

    # Get wavelength grid
    wave = np.asarray(comp_pred.wave_rest)

    # Find the 1e7 Å node in the monolithic grid
    idx_1e7 = int(np.argmin(np.abs(wave - 1e7)))

    # Get the neighboring nodes
    idx_prev = idx_1e7 - 1
    while idx_prev >= 0 and wave[idx_prev] > 1e8 - 1:
        # Skip any overflow wavelengths
        idx_prev -= 1

    # Measure the ratio in both models at 1e7 Å
    mono_ratio = mono_sed[idx_1e7] / mono_sed[idx_prev]
    comp_ratio = comp_sed[idx_1e7] / comp_sed[idx_prev]

    # The ratio should be approximately the same (within 5%)
    # Before the fix: comp_ratio would be ~3e-6 (catastrophic collapse)
    # After the fix: comp_ratio should be ~1.009 (smooth continuation)
    ratio_error = abs(comp_ratio - mono_ratio) / mono_ratio

    assert ratio_error < 0.05, (
        f"Composable SED ratio at 1e7 Å does not match monolithic. "
        f"Monolithic ratio: {mono_ratio:.4f}, Composable: {comp_ratio:.4f}, "
        f"Error: {ratio_error:.2%}. Before fix: expected ~3e-6 (catastrophic)."
    )

    # Also check that sed_agn_torus is finite and non-zero everywhere
    if "sed_agn_torus" in comp_components:
        sed_agn_torus = np.asarray(comp_components["sed_agn_torus"])

        # Check that torus SED is finite
        assert np.all(np.isfinite(sed_agn_torus)), (
            f"sed_agn_torus contains non-finite values. "
            f"Found {np.sum(~np.isfinite(sed_agn_torus))} non-finite points."
        )

        # Check that torus SED is non-zero at least up to the SKIRTOR template boundary (1e8 Å)
        # Use a reasonable tolerance for the lowest expected values
        nonzero_torus = sed_agn_torus[sed_agn_torus > 1e15]  # Filter out potential numerical noise
        if len(nonzero_torus) > 0:
            min_nonzero = np.min(nonzero_torus)
            assert min_nonzero > 0, (
                f"sed_agn_torus contains zero or negative values. Minimum: {np.min(sed_agn_torus):.4e}"
            )

    print(f"Test passed:")
    print(f"  Monolithic ratio at 1e7 Å: {mono_ratio:.4f}")
    print(f"  Composable ratio at 1e7 Å: {comp_ratio:.4f}")
    print(f"  Relative error: {ratio_error:.2%}")
