# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the unified loss-function core.

After the Phase II-2 refactor, ``build_loss_fn``,
``build_loglikelihood_fn`` and ``build_loglikelihood_unbounded_fn``
are thin wrappers over a single
``_build_data_neg_log_likelihood_fn`` core. These tests pin down
the relationship between the three so future drift would be caught
immediately.

Mocks the Fitter interface via ``SimpleNamespace`` (same pattern as
``test_user_likelihood_override.py``) — no SSP grid required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract
from types import SimpleNamespace

import jax.numpy as jnp

from tengri.inference.likelihoods.gaussian import diag_gaussian_log_prob
from tengri.inference.loss_functions import (
    build_loglikelihood_fn,
    build_loglikelihood_unbounded_fn,
    build_loss_fn,
)
from tests._doubles import FakeSpec


class _MockModel:
    def __init__(self, fnu_pred):
        self._fnu_pred = jnp.asarray(fnu_pred)

    def predict_photometry(self, params):
        scale = params.get("flux_scale", 1.0)
        return self._fnu_pred * scale


def _make_fitter(user_likelihood=None, data_mask=None):
    fnu_pred = jnp.array([1e-29, 2e-29, 3e-29])
    fnu_obs = jnp.array([1.1e-29, 1.9e-29, 3.05e-29])
    fnu_err = jnp.array([0.1e-29, 0.1e-29, 0.1e-29])
    spec = FakeSpec(free_names=["flux_scale"])
    model = _MockModel(fnu_pred)
    data_args = {"data": fnu_obs, "noise": fnu_err}
    if data_mask is not None:
        data_args["data_mask"] = data_mask
    fitter = SimpleNamespace(
        model=model,
        data=fnu_obs,
        noise=fnu_err,
        data_type="photometry",
        data_mask=data_mask,
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
        _data_args=data_args,
        _user_likelihood=user_likelihood,
    )
    return fitter, fnu_obs, fnu_err, fnu_pred


def test_three_builders_share_one_data_term():
    """Pin down the wrapper relationship around the shared core.
    For physical params θ = unstandardize(ξ):
        build_loss_fn(ξ)                  = -log_lik(θ) + ½ ξᵀξ
        build_loglikelihood_fn(θ)         = +log_lik(θ)
        build_loglikelihood_unbounded_fn(ξ) = +log_lik(θ)
    Any future drift between the three (different sign, missed
    branch, divergent prediction dict) trips this test.
    """
    # Post Phase II-2.3 the legacy default-Gaussian path raises; pass
    # an auto-built-style PhotometryLikelihood as user_likelihood so the
    # data term goes through the standard adapter path (matches what
    # production Fitter does via _maybe_build_default_likelihood).
    from tengri.inference.photometry_likelihood import PhotometryLikelihood

    _, fnu_obs, fnu_err, fnu_pred = _make_fitter()
    user_lik = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)
    fitter, *_ = _make_fitter(user_likelihood=user_lik)
    loss_fn = build_loss_fn(fitter)
    loglik_fn = build_loglikelihood_fn(fitter)
    loglik_unbounded_fn = build_loglikelihood_unbounded_fn(fitter)
    xi = {"flux_scale": 0.7}  # IdentityDist makes physical = unbounded
    data_args = fitter._data_args
    loss = float(loss_fn(xi, data_args))
    ll = float(loglik_fn(xi, data_args))
    ll_unbounded = float(loglik_unbounded_fn(xi, data_args))
    expected_ll = float(diag_gaussian_log_prob(fnu_pred * 0.7, fnu_obs, fnu_err))
    prior = 0.5 * 0.7**2
    assert ll == pytest.approx(expected_ll, rel=1e-6)
    assert ll_unbounded == pytest.approx(expected_ll, rel=1e-6)
    assert loss == pytest.approx(-expected_ll + prior, rel=1e-6)


def test_legacy_default_path_now_raises():
    """Phase II-2.3 hardening: the unreachable defensive default-Gaussian
    fall-through now raises AssertionError. Production never reaches it
    because Fitter._maybe_build_default_likelihood always builds an
    adapter for the configurations it covers (everything except
    data_mask + non-photometry, which goes through the censored
    branch). The raise surfaces missing auto-build coverage loudly.
    """
    fitter, *_ = _make_fitter(user_likelihood=None)
    loss_fn = build_loss_fn(fitter)
    with pytest.raises(AssertionError, match="auto-build"):
        loss_fn({"flux_scale": 1.0}, fitter._data_args)


def test_loglik_handles_censored_likelihood_via_user_path():
    """Regression for the pre-refactor drift bug.
    Before unification, ``build_loglikelihood_fn`` had no censored
    branch — NSS/ESS on data with non-detections silently treated
    masked bands as zero-flux detections. After the unification, the
    censored case lives in the auto-built ``CensoredLikelihood``
    behind the user-likelihood short-circuit, which is shared by
    all three wrappers.
    """
    from tengri.inference.likelihoods.protocol import CensoredLikelihood

    fnu_obs = jnp.array([1.1e-29, 1.9e-29, 3.05e-29])
    fnu_err = jnp.array([0.1e-29, 0.1e-29, 0.1e-29])
    mask = jnp.array([True, False, True])  # band 1 is an upper limit
    censored_lik = CensoredLikelihood(obs=fnu_obs, err=fnu_err, mask=mask, channel="phot_fnu")
    fitter, *_ = _make_fitter(user_likelihood=censored_lik, data_mask=mask)
    ll = float(build_loglikelihood_fn(fitter)({"flux_scale": 1.0}, fitter._data_args))
    loss = float(build_loss_fn(fitter)({"flux_scale": 1.0}, fitter._data_args))
    assert jnp.isfinite(ll)
    assert jnp.isfinite(loss)
    # If the masked band were silently treated as detected, the
    # likelihood would differ — confirm the censored adapter actually
    # ran by comparing against the same Fitter without the mask
    # (using a plain PhotometryLikelihood as user_likelihood since
    # the legacy default Gaussian now raises post-II-2.3).
    from tengri.inference.photometry_likelihood import PhotometryLikelihood

    plain_lik = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)
    fitter_no_mask, *_ = _make_fitter(user_likelihood=plain_lik, data_mask=None)
    ll_no_mask = float(
        build_loglikelihood_fn(fitter_no_mask)({"flux_scale": 1.0}, fitter_no_mask._data_args)
    )
    assert ll != pytest.approx(ll_no_mask)
