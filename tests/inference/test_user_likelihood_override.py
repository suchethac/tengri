from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the user-supplied ``likelihood=`` override path on Fitter.

Validates that ``Fitter(model, data, noise, likelihood=Custom)`` routes
the χ² term through the user's :class:`tengri.protocols.Likelihood` object
instead of the built-in dispatch in :mod:`tengri.inference.loss_functions`.

These tests mock the Fitter interface so we don't need to spin up an
SSP grid — :func:`build_loss_fn` and :func:`build_loglikelihood_fn`
read a small set of attributes from ``fitter`` which we provide
directly via :class:`SimpleNamespace`.
"""


from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tengri.inference.likelihoods.gaussian import diag_gaussian_log_prob
from tengri.inference.loss_functions import build_loglikelihood_fn, build_loss_fn
from tengri.inference.photometry_likelihood import PhotometryLikelihood
from tests._shared_mocks import MockSpec

_MockSpec = MockSpec


class _MockModel:
    """Minimal model exposing predict_photometry."""

    def __init__(self, fnu_pred):
        self._fnu_pred = jnp.asarray(fnu_pred)
        self.spec = None  # set after construction

    def predict_photometry(self, params):
        # Scale prediction by the first free parameter so we can verify
        # that gradient-relevant params actually flow through.
        scale = params.get("flux_scale", 1.0)
        return self._fnu_pred * scale


def _make_fitter(user_likelihood=None):
    """Mock the Fitter attributes that build_loss_fn / build_loglikelihood_fn read."""
    fnu_pred = jnp.array([1e-29, 2e-29, 3e-29])
    fnu_obs = jnp.array([1.1e-29, 1.9e-29, 3.05e-29])
    fnu_err = jnp.array([0.1e-29, 0.1e-29, 0.1e-29])

    spec = _MockSpec(free_names=["flux_scale"])
    model = _MockModel(fnu_pred)
    model.spec = spec

    fitter = SimpleNamespace(
        model=model,
        data=fnu_obs,
        noise=fnu_err,
        data_type="photometry",
        data_mask=None,
        spec=spec,
        _free_names=spec.free_params,
        _fixed_values={},
        _bounds={"flux_scale": (-jnp.inf, jnp.inf)},
        _calibration_marginalize=False,
        _has_spectroscopy=False,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _eline_marginalize=False,
        _eline_fitted=False,
        _eline_wavelengths=None,
        _eline_independent_wavelengths=None,
        _eline_constraint_matrix=None,
        _eline_prior_type=None,
        _eline_prior_sigma=None,
        _eline_prior_width_dex=None,
        _eline_amplitude_names=[],
        _data_args={
            "data": fnu_obs,
            "noise": fnu_err,
        },
        _user_likelihood=user_likelihood,
    )
    return fitter, fnu_obs, fnu_err, fnu_pred


def test_legacy_default_path_now_raises():
    """Phase II-2.3 hardening: the unreachable defensive default-Gaussian
    fall-through now raises AssertionError. Production never hits it
    because Fitter._maybe_build_default_likelihood always builds an
    adapter for the configurations it supports. The raise surfaces
    missing auto-build coverage loudly rather than silently degrading."""
    fitter, *_ = _make_fitter(user_likelihood=None)
    loss_fn = build_loss_fn(fitter)
    with pytest.raises(AssertionError, match="auto-build"):
        loss_fn({"flux_scale": 1.0}, fitter._data_args)


def test_user_likelihood_replaces_chi2():
    """With PhotometryLikelihood, loss = -log_prob + prior penalty."""
    _, fnu_obs, fnu_err, fnu_pred = _make_fitter(user_likelihood=None)
    user_lik = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)
    fitter, *_ = _make_fitter(user_likelihood=user_lik)

    loss_fn = build_loss_fn(fitter)
    result = loss_fn({"flux_scale": 1.0}, fitter._data_args)

    # User likelihood is exactly diag_gaussian_log_prob, so loss must match
    # the legacy path (modulo the 0.5 vs neg-log-prob convention).
    log_prob = diag_gaussian_log_prob(fnu_pred, fnu_obs, fnu_err)
    expected = -log_prob + 0.5 * 1.0  # prior penalty for ξ=1.0
    assert float(result) == pytest.approx(float(expected), rel=1e-6)


def test_user_likelihood_loglikelihood_path():
    """build_loglikelihood_fn also routes through the user likelihood."""
    _, fnu_obs, fnu_err, fnu_pred = _make_fitter(user_likelihood=None)
    user_lik = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)
    fitter, *_ = _make_fitter(user_likelihood=user_lik)

    loglik_fn = build_loglikelihood_fn(fitter)
    result = loglik_fn({"flux_scale": 1.0}, fitter._data_args)

    expected = diag_gaussian_log_prob(fnu_pred, fnu_obs, fnu_err)
    assert float(result) == pytest.approx(float(expected), rel=1e-6)


def test_custom_user_likelihood_changes_result():
    """A non-Gaussian custom likelihood produces a different scalar."""

    class _AbsoluteResidualLikelihood:
        """L1 likelihood — definitely not equal to Gaussian χ²."""

        name = "l1"

        def __init__(self, fnu_obs, err):
            self.obs = fnu_obs
            self.err = err

        def log_prob(self, prediction, params):
            return -jnp.sum(jnp.abs(prediction["phot_fnu"] - self.obs) / self.err)

        def declared_parameters(self):
            return []

    fnu_obs_arr = jnp.array([1.1e-29, 1.9e-29, 3.05e-29])
    err_arr = jnp.array([0.1e-29, 0.1e-29, 0.1e-29])
    fitter, fnu_obs, fnu_err, fnu_pred = _make_fitter(
        user_likelihood=_AbsoluteResidualLikelihood(fnu_obs=fnu_obs_arr, err=err_arr)
    )
    loglik_fn = build_loglikelihood_fn(fitter)
    result = loglik_fn({"flux_scale": 1.0}, fitter._data_args)

    # L1 sum of residuals.
    l1 = jnp.sum(jnp.abs(fnu_pred - fnu_obs) / fnu_err)
    expected = -l1
    assert float(result) == pytest.approx(float(expected), rel=1e-6)
    # Sanity: differs from the Gaussian answer.
    gaussian = float(diag_gaussian_log_prob(fnu_pred, fnu_obs, fnu_err))
    assert abs(float(result) - gaussian) > 1e-3


def test_prior_term_added_for_free_params_and_psd_xi():
    """The user-likelihood short-circuit must add ½ ξᵀξ over both
    free_names AND psd_xi (stochastic SFH case)."""

    # Use MockSpec with stochastic=True for this test
    class _StochasticSpec(MockSpec):
        def __init__(self, free_names):
            super().__init__(free_names, stochastic=True)

    fnu_pred = jnp.array([1.0e-29, 2.0e-29])
    fnu_obs = jnp.array([1.0e-29, 2.0e-29])
    fnu_err = jnp.array([0.1e-29, 0.1e-29])

    spec = _StochasticSpec(free_names=["a", "b", "c"])
    model = _MockModel(fnu_pred)
    model.spec = spec

    user_lik = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)

    fitter = SimpleNamespace(
        model=model,
        data=fnu_obs,
        noise=fnu_err,
        data_type="photometry",
        data_mask=None,
        spec=spec,
        _free_names=spec.free_params,
        _fixed_values={},
        _bounds={n: (-jnp.inf, jnp.inf) for n in spec.free_params},
        _calibration_marginalize=False,
        _has_spectroscopy=False,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _eline_marginalize=False,
        _eline_fitted=False,
        _eline_wavelengths=None,
        _eline_independent_wavelengths=None,
        _eline_constraint_matrix=None,
        _eline_prior_type=None,
        _eline_prior_sigma=None,
        _eline_prior_width_dex=None,
        _eline_amplitude_names=[],
        _data_args={"data": fnu_obs, "noise": fnu_err},
        _user_likelihood=user_lik,
    )

    # Drive loss_fn with non-trivial ξ and psd_xi values.
    xi = {"a": 0.5, "b": -1.0, "c": 2.0, "psd_xi": jnp.array([0.1, 0.2, 0.3])}
    loss_fn = build_loss_fn(fitter)
    actual = float(loss_fn(xi, fitter._data_args))

    # Expected: −log_prob_data + ½ (ξᵀξ + psd_xiᵀpsd_xi)
    log_prob_data = float(diag_gaussian_log_prob(fnu_pred, fnu_obs, fnu_err))
    xi_sq = 0.5**2 + 1.0**2 + 2.0**2  # for free names
    psd_sq = 0.1**2 + 0.2**2 + 0.3**2  # for psd_xi
    expected = -log_prob_data + 0.5 * (xi_sq + psd_sq)
    assert actual == pytest.approx(expected, rel=1e-6)


def test_prior_includes_only_free_params_when_not_stochastic():
    """Non-stochastic spec → no psd_xi term even if ξ-dict has the key."""
    fnu_pred = jnp.array([1.0e-29])
    fnu_obs = jnp.array([1.0e-29])
    fnu_err = jnp.array([0.1e-29])

    spec = _MockSpec(free_names=["a"])  # non-stochastic by default
    model = _MockModel(fnu_pred)
    model.spec = spec

    user_lik = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)
    fitter = SimpleNamespace(
        model=model,
        data=fnu_obs,
        noise=fnu_err,
        data_type="photometry",
        data_mask=None,
        spec=spec,
        _free_names=spec.free_params,
        _fixed_values={},
        _bounds={"a": (-jnp.inf, jnp.inf)},
        _calibration_marginalize=False,
        _has_spectroscopy=False,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _eline_marginalize=False,
        _eline_fitted=False,
        _eline_wavelengths=None,
        _eline_independent_wavelengths=None,
        _eline_constraint_matrix=None,
        _eline_prior_type=None,
        _eline_prior_sigma=None,
        _eline_prior_width_dex=None,
        _eline_amplitude_names=[],
        _data_args={"data": fnu_obs, "noise": fnu_err},
        _user_likelihood=user_lik,
    )

    # Even if a "psd_xi" key sneaks into params_unbounded, the
    # non-stochastic branch must NOT include it in the prior.
    xi = {"a": 1.5, "psd_xi": jnp.array([99.0, 99.0])}
    loss_fn = build_loss_fn(fitter)
    actual = float(loss_fn(xi, fitter._data_args))

    log_prob_data = float(diag_gaussian_log_prob(fnu_pred, fnu_obs, fnu_err))
    expected = -log_prob_data + 0.5 * 1.5**2  # only free name's ξ²
    assert actual == pytest.approx(expected, rel=1e-6)


def test_composite_likelihood_through_fitter():
    """CompositeLikelihood with a single PhotometryLikelihood matches the
    standalone wrapper."""
    from tengri.inference.composite_likelihood import CompositeLikelihood

    _, fnu_obs, fnu_err, fnu_pred = _make_fitter(user_likelihood=None)
    composite = CompositeLikelihood(PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err))
    fitter, *_ = _make_fitter(user_likelihood=composite)

    loglik_fn = build_loglikelihood_fn(fitter)
    result = loglik_fn({"flux_scale": 1.0}, fitter._data_args)

    expected = diag_gaussian_log_prob(fnu_pred, fnu_obs, fnu_err)
    assert float(result) == pytest.approx(float(expected), rel=1e-6)
