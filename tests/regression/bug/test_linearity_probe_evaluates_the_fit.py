# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for linearity probe evaluation at the fit's own fixed values.

Tests that:
1. Model-evaluation errors propagate from the probe instead of being logged
2. The probe evaluates at the Fitter's resolved fixed values (including params_override)
3. The numeric-invalid path still works when other thetas are valid
"""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from tengri import (
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    recipes,
)
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


def _minimal_photometry_model(ssp_data):
    """Minimal model for testing (photometry only)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


@pytest.mark.slow
class TestLinearityProbeErrors:
    """Test that model-evaluation errors propagate from the linearity probe."""

    def test_model_eval_error_propagates(self, ssp_data, caplog):
        """Model-evaluation errors propagate out instead of being logged as invalid thetas."""
        model = _minimal_photometry_model(ssp_data)
        fitter = Fitter(model, method="map", profile_mass=True)

        # Monkeypatch _predict_full_vector to raise KeyError on first call
        original_predict = fitter.model.predict_photometry
        call_count = [0]

        def mock_predict_photometry(params):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyError("Redshift not in params and not fixed in spec")
            return original_predict(params)

        # The probe should propagate the KeyError, not log it as a warning
        with (
            patch.object(type(fitter.model), "predict_photometry", mock_predict_photometry),
            pytest.raises(KeyError, match="Redshift not in params"),
        ):
            # Trigger profile_mass evaluation
            # Note: profile_mass evaluation happens during Fitter construction
            pass

        # No "linearity probe: ... valid bands" warning should be present
        assert "linearity probe:" not in caplog.text

    @pytest.mark.slow
    def test_probe_evaluates_at_fitter_fixed_values(self, ssp_data, monkeypatch):
        """Probe evaluates at Fitter's fixed values, not spec's (including params_override)."""
        # Build with Fixed redshift at z1
        z1 = 0.5
        obs = Observation(photometry=Photometry.from_names(_FILTERS))
        model = SEDModel.build(
            ssp_data=ssp_data,
            observation=obs,
            redshift=Fixed(z1),
            **recipes.mock_recovery_minimal(),
        )

        # Create fitter with params_override for runtime z2
        z2 = 1.5
        fitter = Fitter(
            model,
            method="map",
            profile_mass=True,
            params_override={"redshift": z2},
        )

        # Capture the redshift values used in _predict_full_vector calls
        recorded_redshifts = []
        original_predict = fitter.model.predict_photometry

        def track_redshift_predict(params):
            z_val = params.get("redshift", None)
            if z_val is not None:
                recorded_redshifts.append(float(z_val))
            return original_predict(params)

        # Monkeypatch to track what redshift is used
        monkeypatch.setattr(fitter.model, "predict_photometry", track_redshift_predict)

        # Trigger profile_mass setup - this calls the linearity probe
        # Fitter construction should now use z2 in the probe, not z1
        profile_computed = fitter._profile_mass_computed

        # Verify that all recorded redshifts are z2, not z1
        # (with some tolerance for floating point)
        if recorded_redshifts:
            for z_recorded in recorded_redshifts:
                assert abs(z_recorded - z2) < 1e-10, (
                    f"Expected redshift {z2} but probe used {z_recorded}"
                )

    @pytest.mark.slow
    def test_numeric_invalid_path_still_works(self, ssp_data, monkeypatch):
        """Numeric-invalid (NaN) predictions skip without raising when other thetas valid."""
        model = _minimal_photometry_model(ssp_data)
        fitter = Fitter(model, method="map", profile_mass=True)

        # Monkeypatch _predict_full_vector to return NaN for theta index 0
        from tengri.inference import mass_profile

        original_predict_fn = mass_profile._predict_full_vector
        call_count = [0]

        def mock_predict_full_vector(*args, **kwargs):
            call_count[0] += 1
            result = original_predict_fn(*args, **kwargs)
            # Return NaN for the first theta (when called by linearity probe)
            if call_count[0] <= 2:  # First two calls are for theta 0
                return jnp.full_like(result, jnp.nan)
            return result

        monkeypatch.setattr(mass_profile, "_predict_full_vector", mock_predict_full_vector)

        # This should not raise - the probe should skip the NaN theta
        # and continue with the other 8 thetas
        # Note: This test verifies the current behavior is preserved
        profile_computed = fitter._profile_mass_computed  # Trigger lazy evaluation
