# SPDX-License-Identifier: BSD-3-Clause
"""Regression: CloudyGridBackend Q_H table NaN from zero-age SSP bin.

When an SSP's age axis starts with log10(0) = -inf (typical for BC03),
the -inf entry turns _interp_index_weight weights into NaN/inf products when
bracketing a finite query age between -inf and the next age node. All nebular
SED becomes NaN. The fix floors the zero-age bin to a finite age at
construction time, matching the stellar path's convention (jnp.maximum(..., 5.0)).

The guard refuses any non-finite age or metallicity value in the SSP axis
after construction, naming the axis, index, and value.

Measured mechanism (#2418): BC03 SSP + PRSC CLOUDY grid → 7955/7955 NaN in
sed_nebular; FSPS PRSC SSP → 0 NaN.

References
----------
Issue #2418: CloudyGridBackend on BC03 SSP returns all-NaN sed_nebular.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def _make_synthetic_ssp_with_zero_age(n_met=3, n_age=7, n_wave=100):
    """Create a synthetic SSP with -inf as the first age node.

    Age axis: [-inf, -3.9, -3.85, -3.8, -3.0, -2.0, 0.0] in log10(Gyr).
    When converted to log10(yr) via +9.0, becomes [-inf, 5.1, 5.15, 5.2, 6.0, 7.0, 9.0].
    """
    ssp_lg_age_gyr = jnp.array([-jnp.inf, -3.90, -3.85, -3.80, -3.0, -2.0, 0.0])
    ssp_lgmet = jnp.linspace(-3.0, 0.0, n_met)
    ssp_wave = jnp.logspace(2.0, 5.0, n_wave)  # 100 Å to 100 µm
    ssp_flux = jnp.ones((n_met, n_age, n_wave))  # Simple constant SED

    from tengri.components.stellar.sps import SSPData

    return SSPData(
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        ssp_lgmet=ssp_lgmet,
        ssp_mass_remaining=None,
        ssp_alpha_fe=None,
        imf="chabrier",
        source="test_synthetic_zero_age",
        nebular="bare",
    )


def _make_synthetic_cloudy_grid():
    """Create a minimal synthetic CLOUDY grid for testing."""
    from tengri.components.nebular.cloudy_grid import CloudyGridData

    # Minimal grid: age 6.0-7.3, metallicity -3.0-0.0, logU -3.0 to -1.0
    line_log_age = jnp.array([6.0, 6.5, 7.0, 7.3])
    line_log_met = jnp.array([-3.0, -1.5, 0.0])
    line_log_U = jnp.array([-3.0, -2.0, -1.0])

    # Create dummy line and continuum data
    n_lines = 10
    n_age, n_met, n_U = line_log_age.shape[0], line_log_met.shape[0], line_log_U.shape[0]

    line_luminosity = jnp.ones((n_met, n_age, n_U, n_lines))
    line_wavelengths = jnp.linspace(1000.0, 10000.0, n_lines)

    cont_log_age = line_log_age
    cont_log_met = line_log_met
    cont_log_U = line_log_U
    cont_wavelength = jnp.logspace(2.0, 5.0, 100)
    cont_luminosity = jnp.ones((n_met, n_age, n_U, 100))

    return CloudyGridData(
        line_wavelengths=line_wavelengths,
        line_luminosity=line_luminosity,
        line_log_met=line_log_met,
        line_log_age=line_log_age,
        line_log_U=line_log_U,
        cont_wavelength=cont_wavelength,
        cont_luminosity=cont_luminosity,
        cont_log_met=cont_log_met,
        cont_log_age=cont_log_age,
        cont_log_U=cont_log_U,
    )


def test_bug_2418_cloudy_grid_zero_age_bin_axis_is_finite_after_construction():
    """GREEN: zero-age SSP bin is floored; Q_H axis is finite after construction."""
    from unittest.mock import patch

    from tengri.components.nebular.cloudy_grid import CloudyGridBackend

    ssp = _make_synthetic_ssp_with_zero_age()

    # Mock the grid loader so we don't need real data
    mock_grid = _make_synthetic_cloudy_grid()

    # After the fix, CloudyGridBackend floors the -inf age to a finite value.
    with patch("tengri.components.nebular.cloudy_grid.load_cloudy_grid", return_value=mock_grid):
        backend = CloudyGridBackend(
            grid_path="dummy_path.h5",
            ssp_data=ssp,
            ionizing_source_warning="suppress",
        )

    # Check that the Q_H log age axis is finite after construction (fix applied)
    assert jnp.all(jnp.isfinite(backend._qh_log_age)), (
        f"Q_H age axis should be finite after construction, got: {backend._qh_log_age}"
    )

    # Check that Q_H log metallicity axis is also finite
    assert jnp.all(jnp.isfinite(backend._qh_log_met)), (
        f"Q_H met axis should be finite, got: {backend._qh_log_met}"
    )


def test_bug_2418_guard_refuses_non_finite_metallicity():
    """YELLOW: Guard fires and names the axis/index/value when met axis has NaN."""
    from unittest.mock import patch

    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.components.stellar.sps import SSPData

    # Create SSP with NaN in the metallicity axis (past the age floor)
    n_age = 5
    ssp_lg_age_gyr = jnp.array([-3.90, -3.85, -3.0, -2.0, 0.0])
    # Inject NaN at metallicity index 1
    ssp_lgmet = jnp.array([-3.0, jnp.nan, 0.0])
    ssp_wave = jnp.logspace(2.0, 5.0, 100)
    ssp_flux = jnp.ones((3, n_age, 100))

    ssp = SSPData(
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        ssp_lgmet=ssp_lgmet,
        ssp_mass_remaining=None,
        ssp_alpha_fe=None,
        imf="chabrier",
        source="test_synthetic_nan_met",
        nebular="bare",
    )

    mock_grid = _make_synthetic_cloudy_grid()

    # The guard should fire with a ValueError naming the axis, index, and value
    with (
        patch("tengri.components.nebular.cloudy_grid.load_cloudy_grid", return_value=mock_grid),
        pytest.raises(ValueError) as exc_info,
    ):
        CloudyGridBackend(
            grid_path="dummy_path.h5",
            ssp_data=ssp,
            ionizing_source_warning="suppress",
        )

    # Verify the error message names the axis, index, and value
    error_msg = str(exc_info.value)
    assert "metallicity" in error_msg.lower(), "Error should name the metallicity axis"
    assert "index" in error_msg.lower(), "Error should include the index"
    assert "non-finite" in error_msg.lower(), "Error should indicate non-finite value"


def test_bug_2418_finite_age_ssp_works():
    """GREEN: SSP with finite ages produces finite Q_H table."""
    from unittest.mock import patch

    from tengri.components.nebular.cloudy_grid import CloudyGridBackend

    # Create SSP with finite ages (no -inf)
    n_age = 7
    ssp_lg_age_gyr = jnp.array([-3.90, -3.85, -3.80, -3.0, -2.0, 0.0, 1.0])
    ssp_lgmet = jnp.linspace(-3.0, 0.0, 3)
    ssp_wave = jnp.logspace(2.0, 5.0, 100)
    ssp_flux = jnp.ones((3, n_age, 100))

    from tengri.components.stellar.sps import SSPData

    ssp = SSPData(
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        ssp_lgmet=ssp_lgmet,
        ssp_mass_remaining=None,
        ssp_alpha_fe=None,
        imf="chabrier",
        source="test_synthetic_finite",
        nebular="bare",
    )

    mock_grid = _make_synthetic_cloudy_grid()

    with patch("tengri.components.nebular.cloudy_grid.load_cloudy_grid", return_value=mock_grid):
        backend = CloudyGridBackend(
            grid_path="dummy_path.h5",
            ssp_data=ssp,
            ionizing_source_warning="suppress",
        )

    # Q_H table should be finite
    assert jnp.all(jnp.isfinite(backend._qh_log_age)), (
        f"Expected finite Q_H age axis, got {backend._qh_log_age}"
    )

    # Q_H table values should be mostly finite (some may be 0 for missing UV)
    assert jnp.any(backend._qh_table > 0), "Q_H table should have some positive entries"
