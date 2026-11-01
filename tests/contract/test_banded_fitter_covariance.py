# SPDX-License-Identifier: BSD-3-Clause
"""Banded pixel covariance through Spectroscopy, the likelihood builder and the marginals."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.likelihood import build_base_likelihood
from tengri.inference.likelihoods import (
    CalibrationELineMarginalizedLikelihood,
    CalibrationMarginalizedLikelihood,
    ELineMarginalizedLikelihood,
    MultivariateGaussianLikelihood,
)
from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood
from tengri.observation.banded import BandedMatrix
from tengri.observation.calibration import calibration_basis
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract

N_PIX = 60
HALF_BAND = 3
LOG_2PI = float(np.log(2.0 * np.pi))


def _banded_spd(seed: int, n: int, p: int) -> np.ndarray:
    """Dense symmetric positive-definite matrix with half-bandwidth p."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    i, j = np.indices((n, n))
    a = np.where(np.abs(i - j) <= p, 0.5 * (a + a.T), 0.0)
    return a + (np.abs(a).sum(axis=1).max() + 1.0) * np.eye(n)


def _as_banded(a: np.ndarray, p: int) -> BandedMatrix:
    """Symmetric diagonal-offset storage, data[k, i] = a[i, i + offsets[k]]."""
    n = a.shape[0]
    offsets = np.arange(-p, p + 1)
    data = np.zeros((offsets.size, n))
    for k, off in enumerate(offsets):
        rows = np.arange(max(0, -off), min(n, n - off))
        data[k, rows] = a[rows, rows + off]
    return BandedMatrix(offsets=offsets, data=jnp.asarray(data))


def _proxy_fitter(spectroscopy, data, noise, **flags):
    """Thin Fitter stand-in carrying the attributes the likelihood builder reads."""
    model = SimpleNamespace(
        observation=SimpleNamespace(spectroscopy=spectroscopy, n_data_phot=None),
        wave_obs=spectroscopy.wave_obs,
    )
    return SimpleNamespace(
        data=jnp.asarray(data),
        noise=jnp.asarray(noise),
        data_type="spectroscopy",
        data_mask=None,
        spec=SimpleNamespace(all_params=[], free_params=[]),
        model=model,
        _calibration_marginalize=flags.get("cal", False),
        _has_spectroscopy=True,
        _eline_marginalize=False,
        _eline_fitted=False,
        _eline_prior_type=None,
        _eline_wavelengths=None,
        _eline_constraint_matrix=None,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _fixed_values={},
        _data_args=flags["data_args"],
    )


def _fixture(seed=0, p=HALF_BAND):
    rng = np.random.default_rng(seed)
    dense = _banded_spd(seed, N_PIX, p)
    wave = np.linspace(4000.0, 7000.0, N_PIX)
    obs = rng.normal(size=N_PIX) + 5.0
    model_flux = 5.0 + 0.3 * rng.normal(size=N_PIX)
    return dense, wave, obs, model_flux


def test_spectroscopy_factors_dense_banded_and_banded_inputs_alike():
    dense, wave, _, _ = _fixture()
    from_dense = Spectroscopy(wave_obs=wave, covariance=dense)
    from_banded = Spectroscopy(wave_obs=wave, covariance=_as_banded(dense, HALF_BAND))
    assert from_dense.has_banded_covariance
    assert from_dense.chol_offsets == tuple(-np.arange(HALF_BAND + 1))
    np.testing.assert_allclose(from_dense.chol_data, from_banded.chol_data, rtol=1e-12)
    lower = np.linalg.cholesky(dense)
    np.testing.assert_allclose(from_banded.chol_data[0], np.diag(lower), rtol=1e-12)
    np.testing.assert_allclose(from_banded.cov_inv, np.linalg.inv(dense), rtol=1e-10)


def test_spectroscopy_keeps_a_dense_inverse_for_a_non_banded_covariance():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(N_PIX, N_PIX))
    dense = a @ a.T + N_PIX * np.eye(N_PIX)
    spec = Spectroscopy(wave_obs=np.linspace(4000.0, 7000.0, N_PIX), covariance=dense)
    assert spec.has_covariance
    assert not spec.has_banded_covariance
    assert spec.chol_data is None
    np.testing.assert_allclose(spec.cov_inv, np.linalg.inv(dense), rtol=1e-9)


def _fitter_data_args(spec, obs, noise):
    """The data_args the Fitter threads into its loss for this spectroscopy."""
    holder = SimpleNamespace(
        data=jnp.asarray(obs),
        noise=jnp.asarray(noise),
        data_mask=None,
        presence=None,
        _runtime_redshift=None,
        _params_override=None,
        _resolved_line_fluxes=lambda: None,
        _resolved_line_model_sigma=lambda: None,
    )
    model = SimpleNamespace(observation=SimpleNamespace(spectroscopy=spec))
    return Fitter._build_data_args(holder, model)


def test_fitter_built_banded_likelihood_equals_dense_quadratic_form():
    dense, wave, obs, model_flux = _fixture()
    spec = Spectroscopy(wave_obs=wave, covariance=_as_banded(dense, HALF_BAND))
    data_args = _fitter_data_args(spec, obs, np.sqrt(np.diag(dense)))
    assert "spec_cov_inv" not in data_args
    proxy = _proxy_fitter(spec, obs, np.sqrt(np.diag(dense)), data_args=data_args)
    likelihood = build_base_likelihood(InferenceContext.from_target(proxy))
    assert isinstance(likelihood, MultivariateGaussianLikelihood)
    value = likelihood.log_prob({"spec_fnu": jnp.asarray(model_flux)}, None, data_args)
    diff = obs - model_flux
    expected = -0.5 * diff @ np.linalg.inv(dense) @ diff
    np.testing.assert_allclose(float(value), expected, rtol=1e-10)


def test_fitter_built_diagonal_covariance_equals_the_sigma_path():
    _, wave, obs, model_flux = _fixture()
    sigma = 0.3 + 0.1 * np.random.default_rng(7).random(N_PIX)
    spec = Spectroscopy(wave_obs=wave, covariance=np.diag(sigma**2))
    data_args = _fitter_data_args(spec, obs, sigma)
    proxy = _proxy_fitter(spec, obs, sigma, data_args=data_args)
    banded = build_base_likelihood(InferenceContext.from_target(proxy))
    plain = SpectroscopyLikelihood(fnu_obs=jnp.asarray(obs), fnu_err=jnp.asarray(sigma))
    prediction = {"spec_fnu": jnp.asarray(model_flux)}
    np.testing.assert_allclose(
        float(banded.log_prob(prediction, None, data_args)),
        float(
            plain.log_prob(
                prediction, None, {"data": jnp.asarray(obs), "noise": jnp.asarray(sigma)}
            )
        ),
        rtol=1e-12,
    )


def _whitened(dense, vectors):
    lower = np.linalg.cholesky(dense)
    return lower, [np.linalg.solve(lower, v) for v in vectors]


def test_calibration_marginal_equals_hand_whitened_reference():
    dense, wave, obs, model_flux = _fixture(1)
    spec = Spectroscopy(wave_obs=wave, covariance=dense)
    likelihood = CalibrationMarginalizedLikelihood(
        jnp.asarray(obs),
        jnp.asarray(np.sqrt(np.diag(dense))),
        jnp.asarray(wave),
        n_poly=3,
        prior_sigma=0.5,
        chol_data=spec.chol_data,
        chol_offsets=spec.chol_offsets,
    )
    value = likelihood.log_prob({"spec_fnu": jnp.asarray(model_flux)})
    basis = np.asarray(calibration_basis(jnp.asarray(wave), wave.min(), wave.max(), 3))
    lower = np.linalg.cholesky(dense)
    design = np.linalg.solve(lower, model_flux[:, None] * basis)
    obs_w, model_w = np.linalg.solve(lower, obs), np.linalg.solve(lower, model_flux)
    reference = marginalize_linear_nuisance(
        jnp.asarray(obs_w),
        jnp.asarray(model_w),
        jnp.asarray(design),
        sigma=jnp.ones(N_PIX),
        prior_precision=jnp.eye(3) / 0.5**2,
    ).log_like - np.sum(np.log(np.diag(lower)))
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-10)


def _line_design(wave):
    centers = np.array([4800.0, 5600.0, 6300.0])
    return jnp.asarray(np.exp(-0.5 * ((wave[:, None] - centers) / 30.0) ** 2))


def test_eline_marginal_equals_hand_whitened_reference():
    dense, wave, obs, model_flux = _fixture(2)
    spec = Spectroscopy(wave_obs=wave, covariance=dense)
    design = _line_design(wave)
    variance = 4.0
    likelihood = ELineMarginalizedLikelihood(
        jnp.asarray(obs),
        jnp.asarray(np.sqrt(np.diag(dense))),
        design_matrix=design,
        prior_variance=jnp.full(3, variance),
        chol_data=spec.chol_data,
        chol_offsets=spec.chol_offsets,
    )
    value = likelihood.log_prob({"spec_fnu": jnp.asarray(model_flux)})
    lower, (resid_w,) = _whitened(dense, [obs - model_flux])
    design_w = np.linalg.solve(lower, np.asarray(design))
    reference = (
        marginalize_linear_nuisance(
            jnp.asarray(resid_w),
            jnp.zeros(N_PIX),
            jnp.asarray(design_w),
            sigma=jnp.ones(N_PIX),
            prior_precision=jnp.eye(3) / variance,
        ).log_like
        + 0.5 * N_PIX * LOG_2PI
    )
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-10)


def test_calibration_eline_marginal_equals_hand_whitened_reference():
    dense, wave, obs, model_flux = _fixture(4)
    spec = Spectroscopy(wave_obs=wave, covariance=dense)
    design = _line_design(wave)
    likelihood = CalibrationELineMarginalizedLikelihood(
        jnp.asarray(obs),
        jnp.asarray(np.sqrt(np.diag(dense))),
        jnp.asarray(wave),
        design_matrix_builder=lambda params: design,
        n_poly=2,
        prior_sigma=0.5,
        eline_prior_sigma=2.0,
        chol_data=spec.chol_data,
        chol_offsets=spec.chol_offsets,
    )
    value = likelihood.log_prob({"spec_fnu": jnp.asarray(model_flux)})
    basis = np.asarray(calibration_basis(jnp.asarray(wave), wave.min(), wave.max(), 2))
    lower = np.linalg.cholesky(dense)
    full = np.concatenate([model_flux[:, None] * basis, np.asarray(design)], axis=1)
    precision = np.diag(np.concatenate([np.full(2, 1 / 0.5**2), np.full(3, 1 / 2.0**2)]))
    reference = (
        marginalize_linear_nuisance(
            jnp.asarray(np.linalg.solve(lower, obs)),
            jnp.asarray(np.linalg.solve(lower, model_flux)),
            jnp.asarray(np.linalg.solve(lower, full)),
            sigma=jnp.ones(N_PIX),
            prior_precision=jnp.asarray(precision),
        ).log_like
        + 0.5 * N_PIX * LOG_2PI
    )
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-10)
