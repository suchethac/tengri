# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for linearity probe evaluation at the fit's own fixed values.

Tests that:
1. Model-evaluation errors propagate from the probe instead of being logged
2. The probe evaluates at the Fitter's resolved fixed values (including params_override)
3. The numeric-invalid path still works when other thetas are valid
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Fixed, Observation, Photometry, SEDModel, recipes
from tengri.inference import mass_profile
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


def _minimal_photometry_model(ssp_data):
    """Minimal model for testing (photometry only)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def test_linearity_probe_model_error_propagates(ssp_data_bc03, monkeypatch, caplog):
    """Model-evaluation errors propagate out instead of being logged as invalid thetas."""
    model = _minimal_photometry_model(ssp_data_bc03)
    # Build with profile_mass=False so probe doesn't run during construction
    fitter = Fitter(model, method="map", profile_mass=False)

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    # Monkeypatch _predict_full_vector to raise KeyError on first call
    original_predict = mass_profile._predict_full_vector
    call_count = [0]

    def mock_predict_full_vector(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyError("boom")
        return original_predict(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", mock_predict_full_vector)

        # The probe should propagate the KeyError, not log it as a warning
        with pytest.raises(KeyError, match="boom"):
            mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

    # No "linearity probe: ... valid bands" warning should be present
    assert "linearity probe:" not in caplog.text


def test_linearity_probe_evaluates_at_fitter_fixed_values(ssp_data_bc03, monkeypatch):
    """Probe evaluates at Fitter's resolved fixed values, including params_override redshift."""
    # Build with Fixed redshift at z1
    z1 = 0.5
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    model = SEDModel.build(
        ssp_data=ssp_data_bc03,
        observation=obs,
        redshift=Fixed(z1),
        **recipes.mock_recovery_minimal(),
    )

    # Create fitter with params_override for runtime z2
    z2 = 1.5
    fitter = Fitter(
        model,
        method="map",
        profile_mass=False,
        params_override={"redshift": z2},
    )

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    # Capture the redshift values used in _predict_full_vector calls
    recorded_redshifts = []
    original_predict = mass_profile._predict_full_vector

    def track_redshift_predict(model, data_type, params, **kwargs):
        z_val = params.get("redshift", None)
        if z_val is not None:
            recorded_redshifts.append(float(z_val))
        return original_predict(model, data_type, params, **kwargs)

    # Monkeypatch to track what redshift is used
    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", track_redshift_predict)
        # Call the probe directly
        mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

    # Verify that all recorded redshifts are z2, not z1
    # (with some tolerance for floating point)
    assert len(recorded_redshifts) >= 2, "Probe should call predict_full_vector multiple times"
    for z_recorded in recorded_redshifts:
        assert abs(z_recorded - z2) < 1e-12, f"Expected redshift {z2} but probe used {z_recorded}"


def test_linearity_probe_numeric_invalid_path_works(ssp_data_bc03, monkeypatch):
    """Numeric-invalid (NaN) predictions skip without raising when other thetas valid."""
    model = _minimal_photometry_model(ssp_data_bc03)
    # Build with profile_mass=False so probe doesn't run during construction
    fitter = Fitter(model, method="map", profile_mass=False)

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    # Monkeypatch _predict_full_vector to return NaN for theta index 0
    original_predict_fn = mass_profile._predict_full_vector
    call_count = [0]

    def mock_predict_full_vector(*args, **kwargs):
        call_count[0] += 1
        result = original_predict_fn(*args, **kwargs)
        # Return NaN for the first two calls (theta 0's two masses)
        if call_count[0] <= 2:
            return jnp.full_like(result, jnp.nan)
        return result

    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", mock_predict_full_vector)
        # This should not raise - the probe should skip the NaN theta
        # and continue with the other 8 thetas
        max_dev, tol, kind = mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

        # Verify the probe still returned valid results (from other 8 thetas)
        assert jnp.isfinite(max_dev), "Max deviation should be finite"
        assert jnp.isfinite(tol), "Tolerance should be finite"
        assert kind in ("proportional", "affine", "nonlinear"), "Kind should be one of the three"
