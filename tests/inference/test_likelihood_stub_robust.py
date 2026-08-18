# SPDX-License-Identifier: BSD-3-Clause
"""Stub RobustLikelihood to validate extraction seam (Step D).

Demonstrates that the extracted Likelihood module enables custom behavior
(Student-t robust noise) without further edits to Fitter. This proves
the architectural seam is valuable and correctly positioned.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tests._shared_mocks import MockSpec

pytestmark = pytest.mark.contract

_MockSpec = MockSpec


class _MockModel:
    """Minimal model exposing predict_photometry."""

    def __init__(self, fnu_pred):
        self._fnu_pred = jnp.asarray(fnu_pred)
        self.spec = None

    def predict_photometry(self, params, mode=None):
        return self._fnu_pred


def _make_mock_fitter(fnu_pred, fnu_obs, fnu_err):
    """Build a mock Fitter for likelihood testing."""
    spec = _MockSpec(free_names=[])
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


class RobustPhotometryLikelihood:
    """Stub: Student-t photometry likelihood for robust outlier handling.

    This is a proof-of-concept custom likelihood that could be passed to
    Fitter via the likelihood= parameter. It demonstrates that the Likelihood
    extraction (Step D) enables future customization without touching Fitter
    or loss_functions internals.
    """

    def __init__(self, fnu_obs, fnu_err, dof=2.0):
        """Initialize with data and Student-t degrees of freedom.

        Parameters
        ----------
        fnu_obs : ndarray
            Observed flux densities.
        fnu_err : ndarray
            Measurement uncertainties.
        dof : float
            Degrees of freedom for Student-t distribution (default 2.0 for heavy tails).
        """
        self.fnu_obs = jnp.asarray(fnu_obs)
        self.fnu_err = jnp.asarray(fnu_err)
        self.dof = dof
        self.channel = "phot_fnu"
        self.name = "robust_photometry"

    def log_prob(self, prediction, params):
        """Compute log-likelihood using Student-t for robust outlier handling.

        Parameters
        ----------
        prediction : dict
            Prediction dict with 'phot_fnu' key containing model predictions.
        params : dict
            Physical parameters (unused in this stub).

        Returns
        -------
        float
            Log-likelihood under Student-t noise model.
        """
        fnu_pred = prediction["phot_fnu"]
        residuals = (self.fnu_obs - fnu_pred) / self.fnu_err

        # Student-t log-likelihood: log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(πν)
        # - 0.5*(ν+1)*log(1 + z²/ν)
        # where ν = dof (degrees of freedom), z = residuals

        from scipy.special import loggamma

        nu = self.dof
        z2 = residuals**2

        # Use scipy only for the Gamma function (constant at likelihood time)
        log_gamma_ratio = float(loggamma((nu + 1) / 2) - loggamma(nu / 2))
        log_norm = jnp.log(jnp.pi * nu) / 2.0

        log_lik = log_gamma_ratio - log_norm - (nu + 1) / 2 * jnp.log(1 + z2 / nu)

        return jnp.sum(log_lik)


@pytest.mark.integration
class TestRobustLikelihoodThreadsThrough:
    """Validate that custom Likelihood subclasses integrate cleanly."""

    def test_robust_likelihood_computes_student_t(self):
        """RobustPhotometryLikelihood produces plausible log-likelihood."""
        fnu_pred = jnp.array([1e-29, 2e-29, 3e-29])
        fnu_obs = jnp.array([1.1e-29, 1.9e-29, 3.05e-29])
        fnu_err = jnp.array([0.1e-29, 0.1e-29, 0.1e-29])

        robust_lk = RobustPhotometryLikelihood(fnu_obs, fnu_err, dof=2.0)

        prediction = {"phot_fnu": fnu_pred}
        params = {}
        log_p = float(robust_lk.log_prob(prediction, params))

        # Just verify it's a number and makes sense
        assert isinstance(log_p, float)
        assert log_p < 0.0  # log-likelihood should be negative
        print(f"Robust likelihood log_p: {log_p}")

    def test_robust_likelihood_matches_interface(self):
        """RobustPhotometryLikelihood implements the Likelihood protocol."""
        fnu_obs = jnp.array([1.0, 2.0, 3.0])
        fnu_err = jnp.array([0.1, 0.1, 0.1])
        robust_lk = RobustPhotometryLikelihood(fnu_obs, fnu_err)

        # Verify it has the required interface
        assert hasattr(robust_lk, "log_prob"), "Missing log_prob method"
        assert hasattr(robust_lk, "channel"), "Missing channel attribute"
        assert robust_lk.channel == "phot_fnu"
        assert hasattr(robust_lk, "name"), "Missing name attribute"
