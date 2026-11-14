# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the published log-evidence carries the Gaussian normalization constant (#3427).

The data term drops ``-sum_i log sigma_i - (n/2) log 2 pi`` for a Gaussian channel. The
Fitter documents ``log_evidence`` as the Bayesian evidence, so for a fixed-noise channel it
was off by that constant (315.95 nats on the one-parameter five-band toy of the issue).

Each likelihood variant now exposes its exact log normalization for the part the
log-likelihood omits. The part that depends on a free noise parameter (the calibration
floor, the Student-t dof is fixed) is already inside the log-likelihood, so the posterior
is unaffected; the constant is added to the published evidence only.

The toy checks compare against a dense grid of the exact density; the adapter checks
compare the log-likelihood plus its normalization against ``scipy.stats``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

import tengri
from tengri import (
    DEFAULT,
    Fitter,
    Fixed,
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
    parse_groups,
)
from tengri.inference.likelihoods import (
    CensoredLikelihood,
    GaussianLikelihood,
    MultivariateGaussianLikelihood,
    StudentTLikelihood,
)

pytestmark = pytest.mark.regression_bug

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_TRUE_LOGM = 10.0
_GRID = np.linspace(9.0, 11.0, 20001)


# ── Adapter level: log p(d|theta) + normalization == exact scipy density ─────


def _mu_d(n=6, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(1.0, 3.0, n)
    d = mu + rng.normal(0.0, 0.2, n)
    sig = rng.uniform(0.1, 0.3, n)
    return d, mu, sig


def test_gaussian_normalization_matches_scipy_density():
    d, mu, sig = _mu_d()
    lik = GaussianLikelihood(obs=jnp.asarray(d), err=jnp.asarray(sig), channel="phot_fnu")
    ll = float(lik.log_prob({"phot_fnu": jnp.asarray(mu)}))
    exact = float(np.sum(stats.norm.logpdf(d, mu, sig)))
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-9)


def test_gaussian_normalization_with_fractional_floor_matches_scipy_density():
    d, mu, sig = _mu_d(seed=3)
    lik = GaussianLikelihood(
        obs=jnp.asarray(d), err=jnp.asarray(sig), channel="phot_fnu", sigma_floor=0.1
    )
    ll = float(lik.log_prob({"phot_fnu": jnp.asarray(mu)}))
    sig_eff = np.hypot(sig, 0.1 * d)
    exact = float(np.sum(stats.norm.logpdf(d, mu, sig_eff)))
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-9)


def test_student_t_normalization_matches_scipy_density():
    d, mu, sig = _mu_d(seed=1)
    dof = 4.0
    lik = StudentTLikelihood(obs=jnp.asarray(d), err=jnp.asarray(sig), dof=dof)
    ll = float(lik.log_prob({"phot_fnu": jnp.asarray(mu)}))
    exact = float(np.sum(stats.t.logpdf(d, df=dof, loc=mu, scale=sig)))
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-9)


def test_censored_normalization_matches_scipy_density_with_limits():
    d, mu, sig = _mu_d(seed=2)
    mask = np.array([0, 0, 1, -1, 0, 0], dtype=np.int32)
    lik = CensoredLikelihood(
        obs=jnp.asarray(d), err=jnp.asarray(sig), mask=jnp.asarray(mask), channel="phot_fnu"
    )
    ll = float(lik.log_prob({"phot_fnu": jnp.asarray(mu)}))
    det = mask == 0
    up = mask == 1
    low = mask == -1
    exact = (
        float(np.sum(stats.norm.logpdf(d[det], mu[det], sig[det])))
        + float(np.sum(stats.norm.logcdf((d[up] - mu[up]) / sig[up])))
        + float(np.sum(stats.norm.logcdf((mu[low] - d[low]) / sig[low])))
    )
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-9)


def test_censored_student_t_normalization_matches_scipy_density():
    d, mu, sig = _mu_d(seed=4)
    dof = 5.0
    mask = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
    lik = CensoredLikelihood(
        obs=jnp.asarray(d),
        err=jnp.asarray(sig),
        mask=jnp.asarray(mask),
        dof=dof,
        channel="phot_fnu",
    )
    ll = float(lik.log_prob({"phot_fnu": jnp.asarray(mu)}))
    det = mask == 0
    up = mask == 1
    exact = float(np.sum(stats.t.logpdf(d[det], df=dof, loc=mu[det], scale=sig[det]))) + float(
        np.sum(stats.norm.logcdf((d[up] - mu[up]) / sig[up]))
    )
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-9)


def test_multivariate_normalization_matches_scipy_density_dense():
    d, mu, sig = _mu_d(n=5, seed=5)
    cov = np.diag(sig**2) + 0.3 * np.diag(sig[:-1] * sig[1:], k=1)
    cov = cov + cov.T - np.diag(np.diag(cov))
    lik = MultivariateGaussianLikelihood(
        obs=jnp.asarray(d), cov_inv=jnp.asarray(np.linalg.inv(cov)), channel="spec_fnu"
    )
    ll = float(lik.log_prob({"spec_fnu": jnp.asarray(mu)}))
    exact = float(stats.multivariate_normal.logpdf(d, mean=mu, cov=cov))
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-8)


def test_multivariate_normalization_matches_scipy_density_banded():
    d, mu, sig = _mu_d(n=5, seed=6)
    cov = np.diag(sig**2) + np.diag(0.3 * sig[:-1] * sig[1:], k=1)
    cov = cov + cov.T - np.diag(np.diag(cov))
    chol = np.linalg.cholesky(cov)
    chol_data = np.zeros((2, len(d)))
    chol_data[0] = np.diag(chol)
    chol_data[1, 1:] = np.diag(chol, k=-1)
    lik = MultivariateGaussianLikelihood(
        obs=jnp.asarray(d),
        channel="spec_fnu",
        chol_data=jnp.asarray(chol_data),
        chol_offsets=(0, -1),
    )
    ll = float(lik.log_prob({"spec_fnu": jnp.asarray(mu)}))
    exact = float(stats.multivariate_normal.logpdf(d, mean=mu, cov=cov))
    assert ll + float(lik.log_normalization()) == pytest.approx(exact, abs=1e-8)


# ── Toy level: the one-parameter five-band problem of the issue ─────────────


def _toy_mock(lo, hi, floor_prior=None):
    """Five-band SDSS mock at log M = 10; returns the fitter-ready model and data."""
    ssp = load_ssp_data(tengri.download_ssp())
    obs_kwargs = {}
    if floor_prior is not None:
        obs_kwargs["noise"] = NoiseModel(calibration_floor=floor_prior)
    obs = Observation(photometry=Photometry.from_names(_BANDS), **obs_kwargs)
    groups = dict(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(lo, hi)},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        igm={"type": "none"},
        redshift=Fixed(0.05),
    )
    if floor_prior is None:
        sed = SEDModel.build(ssp_data=ssp, observation=obs, **groups)
        fwd = ForwardModel.build(sed=sed, observation=obs)
    else:
        spec = parse_groups(**groups).merge_observation_params(noise_frac_cal=floor_prior)
        sed = SEDModel(spec, ssp, observation=obs)
        fwd = sed
    truth = {"sfh_dpl_log_total_mass": _TRUE_LOGM}
    if floor_prior is not None:
        truth = {**sed.spec.sample(jax.random.PRNGKey(0)), **truth, "noise_frac_cal": 0.0}
    mock = sed.mock(truth, snr=20.0, key=jax.random.PRNGKey(7))
    return fwd, np.asarray(mock.flux_obs), np.asarray(mock.noise), np.asarray(mock.flux_true)


def _grid_log_evidence(fobs, sig, ftrue, lo, hi):
    """Dense-grid ln Z of the exact Gaussian density, prior Uniform(lo, hi) in log M."""
    grid = np.linspace(lo, hi, 20001)
    scale = 10.0 ** (grid[:, None] - _TRUE_LOGM)
    chi2 = np.sum(((fobs[None, :] - ftrue[None, :] * scale) / sig) ** 2, axis=1)
    log_like = -0.5 * chi2 - np.sum(np.log(sig)) - 0.5 * len(fobs) * np.log(2 * np.pi)
    weights = np.ones_like(grid)
    weights[[0, -1]] = 0.5
    return float(logsumexp(log_like - np.log(hi - lo), b=weights) + np.log(grid[1] - grid[0]))


@pytest.fixture(scope="module")
def toy():
    fwd, fobs, sig, ftrue = _toy_mock(9.0, 11.0)
    grid = _grid_log_evidence(fobs, sig, ftrue, 9.0, 11.0)
    return {"fwd": fwd, "fobs": fobs, "sig": sig, "grid": grid}


def test_laplace_log_evidence_matches_dense_grid_with_constant(toy):
    res = Fitter(toy["fwd"], toy["fobs"], toy["sig"]).run("laplace")
    assert res.log_evidence == pytest.approx(toy["grid"], abs=0.05)


def test_nss_log_evidence_within_three_sigma_of_dense_grid_with_constant(toy):
    res = Fitter(toy["fwd"], toy["fobs"], toy["sig"]).run("nss", preset="fast")
    sigma = res.diagnostics["log_evidence_err"]
    assert abs(res.log_evidence - toy["grid"]) < 3.0 * sigma


def test_bayes_factor_between_two_priors_is_unchanged_by_the_constant():
    wide_fwd, fobs, sig, ftrue = _toy_mock(9.0, 11.0)
    narrow_fwd, _, _, _ = _toy_mock(9.5, 10.5)
    z_wide = Fitter(wide_fwd, fobs, sig).run("laplace").log_evidence
    z_narrow = Fitter(narrow_fwd, fobs, sig).run("laplace").log_evidence
    grid_bf = _grid_log_evidence(fobs, sig, ftrue, 9.5, 10.5) - _grid_log_evidence(
        fobs, sig, ftrue, 9.0, 11.0
    )
    assert (z_narrow - z_wide) == pytest.approx(grid_bf, abs=0.1)


def test_free_calibration_floor_map_is_not_pinned_at_the_upper_bound():
    fwd, fobs, sig, _ = _toy_mock(9.0, 11.0, floor_prior=Uniform(0.0, 0.5))
    res = Fitter(fwd, fobs, sig).run("laplace")
    floor = float(np.ravel(res.params["noise_frac_cal"])[0])
    assert floor < 0.4, f"calibration floor pinned toward its upper bound: {floor}"
