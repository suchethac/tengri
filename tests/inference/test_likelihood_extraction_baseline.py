# SPDX-License-Identifier: BSD-3-Clause
"""Baseline log-likelihood capture for Step D extraction validation.

Captures the numerical baseline before Likelihood module extraction,
then verifies bit-equality after extraction. This ensures the refactoring
is a pure structural rearrangement with no physics changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tests._doubles import FakeSpec

pytestmark = pytest.mark.contract


class _MockModel:
    """Minimal model exposing predict_photometry."""

    def __init__(self, fnu_pred):
        self._fnu_pred = jnp.asarray(fnu_pred)
        self.spec = None

    def predict_photometry(self, params, mode=None):
        return self._fnu_pred


def _make_mock_fitter(fnu_pred, fnu_obs, fnu_err):
    """Build a mock Fitter for likelihood testing."""
    spec = FakeSpec(free_names=[])
    model = _MockModel(fnu_pred)
    model.spec = spec

    return SimpleNamespace(
        model=model,
        data=jnp.asarray(fnu_obs),
        noise=jnp.asarray(fnu_err),
        data_type="photometry",
        data_mask=None,
        spec=spec,
        _free_names=spec.free_params,
        _fixed_values={},
        _bounds={},
        _calibration_marginalize=False,
        _has_spectroscopy=False,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _eline_marginalize=False,
        _eline_fitted=False,
        _eline_prior_type=None,
        _eline_wavelengths=None,
        _eline_constraint_matrix=None,
        _eline_independent_wavelengths=None,
        _eline_prior_width_dex=0.1,
        _eline_prior_sigma=1e10,
        _eline_amplitude_names=[],
        _data_args={
            "data": jnp.asarray(fnu_obs),
            "noise": jnp.asarray(fnu_err),
        },
    )


# Baseline values captured before Likelihood extraction
# These are pinned to verify bit-equality after refactoring
_BASELINE_LOG_P_SIMPLE_PHOTOMETRY = -1.1249999999999987


@pytest.mark.integration
class TestLikelihoodExtractionBaseline:
    """Verify likelihood values are preserved across extraction."""

    def test_simple_photometry_baseline(self):
        """Capture baseline for simple photometry likelihood.

        This test runs BEFORE and AFTER the Likelihood extraction refactoring.
        Both must produce the same log_p value to verify the refactoring is
        a pure structural rearrangement with no physics changes.
        """
        from tengri.inference.context import InferenceContext
        from tengri.inference.likelihood import build_base_likelihood

        # Simple photometry case
        fnu_pred = jnp.array([1e-29, 2e-29, 3e-29])
        fnu_obs = jnp.array([1.1e-29, 1.9e-29, 3.05e-29])
        fnu_err = jnp.array([0.1e-29, 0.1e-29, 0.1e-29])

        fitter = _make_mock_fitter(fnu_pred, fnu_obs, fnu_err)

        # Get the likelihood object via the extracted likelihood module.
        # Step-D-prime (ADR-0009) changed the signature to accept an
        # InferenceContext rather than a raw Fitter; wrap the mock here.
        lk = build_base_likelihood(InferenceContext.from_target(fitter))
        assert lk is not None, "Likelihood builder returned None"

        # Call log_prob with prediction dict and params
        # For photometry-only case, prediction is just the model prediction
        prediction = {"phot_fnu": fnu_pred}
        params = fitter.spec.get_fixed_values()
        log_p_computed = float(lk.log_prob(prediction, params))

        # Verify against baseline (zero tolerance — structural change only)
        assert log_p_computed == _BASELINE_LOG_P_SIMPLE_PHOTOMETRY, (
            f"Baseline mismatch: got {log_p_computed}, "
            f"expected {_BASELINE_LOG_P_SIMPLE_PHOTOMETRY}"
        )
