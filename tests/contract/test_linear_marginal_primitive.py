# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the linear-Gaussian marginal primitive and its calibration wrapper.

``_reference_v1`` is a frozen copy of the closed-form ``marginalize_calibration`` body
that preceded the primitive (``git show HEAD:src/tengri/observation/calibration.py``
before the change that introduced ``marginalize_linear_nuisance``). It exists only to
pin the wrapper's three outputs to the original formulas.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from tengri.observation.calibration import (
    calibration_basis,
    calibration_polynomial,
    marginalize_calibration,
)
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.utils.scale import representable_floor, whiten

pytestmark = pytest.mark.contract

LOG_2PI = np.log(2.0 * np.pi)
_ERR_FLOOR = 1e-300


def _reference_v1(model_flux, obs_flux, obs_err, wavelength, n_poly=3, prior_sigma=1.0):
    """Frozen closed-form calibration marginal; reference for the wrapper only."""
    wave_min = wavelength[0]
    wave_max = wavelength[-1]
    x = 2.0 * (wavelength - wave_min) / (wave_max - wave_min) - 1.0

    def _cheb_step(carry, _k):
        t_prev, t_curr = carry
        t_next = 2.0 * x * t_curr - t_prev
        return (t_curr, t_next), t_next

    t0 = jnp.ones_like(x)
    t1 = x
    _, higher = jax.lax.scan(_cheb_step, (t0, t1), jnp.arange(2, n_poly + 1))
    basis = jnp.concatenate([t1[jnp.newaxis], higher], axis=0)
    err_safe = jnp.maximum(obs_err, representable_floor(_ERR_FLOOR))
    weighted_model = whiten(model_flux, err_safe)
    b_matrix = basis * weighted_model[jnp.newaxis, :]
    a_matrix = b_matrix @ b_matrix.T
    prior_precision = 1.0 / (prior_sigma**2)
    a_matrix = a_matrix + prior_precision * jnp.eye(a_matrix.shape[0])
    residual = obs_flux - model_flux
    rhs = jnp.sum(
        basis * weighted_model[jnp.newaxis, :] * whiten(residual, err_safe)[jnp.newaxis, :],
        axis=1,
    )
    c_hat = jnp.linalg.solve(a_matrix, rhs)
    sigma_post = jnp.linalg.solve(a_matrix, jnp.eye(a_matrix.shape[0]))
    c_hat_err = jnp.sqrt(jnp.maximum(jnp.diag(sigma_post), 0.0))
    model_cal = (1.0 + jnp.dot(c_hat, basis)) * model_flux
    chi2 = jnp.sum(whiten(obs_flux - model_cal, err_safe) ** 2)
    prior_penalty = prior_precision * jnp.dot(c_hat, c_hat)
    _sign_a, logdet_a = jnp.linalg.slogdet(a_matrix)
    log_det_lambda = n_poly * jnp.log(prior_sigma**2)
    n_wave = wavelength.shape[0]
    log_norm = -0.5 * n_wave * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(err_safe))
    log_like = log_norm - 0.5 * (chi2 + prior_penalty + logdet_a + log_det_lambda)
    return log_like, c_hat, c_hat_err


def _monotonic_grid(seed, n=300):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.sort(rng.uniform(3800.0, 9000.0, n)))


def _spectrum(wave):
    return 1e-17 * (wave / 5000.0) ** (-1.5) * (1.0 + 0.1 * jnp.sin(wave / 300.0))


def _noisy_obs(wave, model, seed):
    rng = np.random.default_rng(seed)
    cal = 1.0 + 0.05 * jnp.asarray(rng.normal(size=wave.shape[0]))
    return model * cal


def _gaussian_ll_no_nuisance(data, model0, sigma):
    r = (data - model0) / sigma
    n = data.shape[0]
    return -0.5 * n * LOG_2PI - np.sum(np.log(sigma)) - 0.5 * np.sum(r**2)


def _quadrature_log_evidence(data, model0, design, sigma, prior_mean, prior_precision, n_nodes=60):
    """Gauss-Hermite log of the integral of N(d; m0 + X b, sigma^2) N(b; mu, Lambda) over b."""
    k = design.shape[1]
    chol = np.linalg.cholesky(np.linalg.inv(prior_precision))
    nodes, weights = hermgauss(n_nodes)
    grid = np.stack(np.meshgrid(*([nodes] * k), indexing="ij"), axis=-1).reshape(-1, k)
    wgrid = np.stack(np.meshgrid(*([weights] * k), indexing="ij"), axis=-1).reshape(-1, k)
    log_w = np.sum(np.log(wgrid), axis=1)
    betas = prior_mean[None, :] + np.sqrt(2.0) * grid @ chol.T
    pred = model0[None, :] + betas @ design.T
    resid = (data[None, :] - pred) / sigma[None, :]
    n = data.shape[0]
    loglik = -0.5 * np.sum(resid**2, axis=1) - 0.5 * n * LOG_2PI - np.sum(np.log(sigma))
    terms = log_w + loglik
    top = np.max(terms)
    return top + np.log(np.sum(np.exp(terms - top))) - 0.5 * k * np.log(np.pi)


class TestCalibrationWrapperMatchesReference:
    """marginalize_calibration reproduces the frozen closed form."""

    @pytest.mark.parametrize("n_poly", [1, 3, 5])
    @pytest.mark.parametrize("prior_sigma", [0.5, 10.0])
    def test_outputs_match_frozen_reference(self, n_poly, prior_sigma):
        """Log-likelihood, c_hat and c_hat_err agree with the frozen v1 body at rtol 1e-12."""
        wave = _monotonic_grid(seed=7)
        model = _spectrum(wave)
        obs = _noisy_obs(wave, model, seed=11)
        obs_err = 0.02 * model
        new = marginalize_calibration(
            model, obs, obs_err, wave, n_poly=n_poly, prior_sigma=prior_sigma
        )
        old = _reference_v1(model, obs, obs_err, wave, n_poly=n_poly, prior_sigma=prior_sigma)
        for got, want in zip(new, old, strict=True):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=0)


class TestPixelMask:
    """A masked pixel is absent from every term of the marginal."""

    def test_masked_pixels_equal_deleted_pixels(self):
        """Masking pixels in the wrapper equals deleting them, including N_eff and sum ln sigma."""
        wave = _monotonic_grid(seed=3)
        model = _spectrum(wave)
        obs = _noisy_obs(wave, model, seed=5)
        obs_err = 0.03 * model
        keep = np.ones(wave.shape[0], dtype=bool)
        keep[::7] = False
        keep[:3] = False
        wave_range = (float(wave.min()), float(wave.max()))
        masked = marginalize_calibration(
            model,
            obs,
            obs_err,
            wave,
            n_poly=3,
            prior_sigma=1.0,
            pix_mask=jnp.asarray(keep),
            wave_range=wave_range,
        )
        deleted = marginalize_calibration(
            model[keep],
            obs[keep],
            obs_err[keep],
            wave[keep],
            n_poly=3,
            prior_sigma=1.0,
            wave_range=wave_range,
        )
        for got, want in zip(masked, deleted, strict=True):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=0)


class TestColumnMask:
    """A masked column contributes exactly nothing, and all-masked reduces to no nuisance."""

    def _problem(self, seed):
        rng = np.random.default_rng(seed)
        n, k = 40, 3
        design = jnp.asarray(rng.normal(size=(n, k)))
        model0 = jnp.asarray(rng.normal(size=n))
        data = model0 + jnp.asarray(rng.normal(size=n))
        sigma = jnp.asarray(0.7 + 0.3 * rng.random(n))
        a = rng.normal(size=(k, k))
        prec = jnp.asarray(a @ a.T + k * np.eye(k))
        return data, model0, design, sigma, prec

    @pytest.mark.parametrize("gaussian", [True, False])
    def test_masked_column_equals_deleted_column(self, gaussian):
        """Masking column 1 equals deleting it, with a Gaussian or a flat prior."""
        data, model0, design, sigma, prec = self._problem(seed=21)
        keep = np.array([0, 2])
        prec_arg = prec if gaussian else None
        masked = marginalize_linear_nuisance(
            data,
            model0,
            design,
            sigma=sigma,
            prior_precision=prec_arg,
            col_mask=jnp.asarray([1.0, 0.0, 1.0]),
        )
        deleted = marginalize_linear_nuisance(
            data,
            model0,
            design[:, keep],
            sigma=sigma,
            prior_precision=None if prec_arg is None else prec[np.ix_(keep, keep)],
        )
        np.testing.assert_allclose(masked.log_like, deleted.log_like, rtol=1e-12, atol=0)
        np.testing.assert_allclose(masked.coef_hat[keep], deleted.coef_hat, rtol=1e-12, atol=0)
        assert float(masked.coef_hat[1]) == 0.0
        np.testing.assert_allclose(
            np.asarray(masked.coef_cov)[np.ix_(keep, keep)],
            np.asarray(deleted.coef_cov),
            rtol=1e-12,
            atol=0,
        )

    @pytest.mark.parametrize("gaussian", [True, False])
    def test_all_columns_masked_is_gaussian_likelihood(self, gaussian):
        """With every column masked the value is finite and equals the no-nuisance likelihood."""
        data, model0, design, sigma, prec = self._problem(seed=33)
        result = marginalize_linear_nuisance(
            data,
            model0,
            design,
            sigma=sigma,
            prior_precision=prec if gaussian else None,
            col_mask=jnp.zeros(3),
        )
        want = _gaussian_ll_no_nuisance(np.asarray(data), np.asarray(model0), np.asarray(sigma))
        assert np.isfinite(float(result.log_like))
        np.testing.assert_allclose(float(result.log_like), want, rtol=1e-10, atol=0)


class TestQuadratureCheck:
    """The closed form matches Gauss-Hermite integration over the coefficients."""

    @pytest.mark.parametrize("k", [1, 2])
    def test_matches_gauss_hermite(self, k):
        """Every normalization term is checked with a nonzero prior mean and full precision."""
        rng = np.random.default_rng(100 + k)
        n = 20
        design = rng.normal(size=(n, k))
        model0 = rng.normal(size=n) * 0.2
        data = model0 + design @ np.linspace(0.3, -0.5, k) + rng.normal(size=n) * 0.5
        sigma = np.full(n, 2.0)
        prior_mean = np.linspace(0.4, -0.2, k)
        a = rng.normal(size=(k, k))
        prior_precision = a @ a.T + np.eye(k)
        want = _quadrature_log_evidence(data, model0, design, sigma, prior_mean, prior_precision)
        got = marginalize_linear_nuisance(
            jnp.asarray(data),
            jnp.asarray(model0),
            jnp.asarray(design),
            sigma=jnp.asarray(sigma),
            prior_mean=jnp.asarray(prior_mean),
            prior_precision=jnp.asarray(prior_precision),
        )
        np.testing.assert_allclose(float(got.log_like), want, rtol=1e-8, atol=0)


class TestScale:
    """The scale knob leaves the value unchanged and keeps float32 finite."""

    def _problem(self):
        n, k = 60, 2
        phase = np.linspace(0.0, 4.0, n)
        base = 1e-27 * (1.0 + 0.2 * np.sin(phase))
        design = np.stack([np.ones(n), np.cos(phase)], axis=1)
        data = base * (1.0 + 0.05 * np.sin(3.0 * phase))
        return n, k, base, design, data

    def test_scale_equals_unscaled_float64(self):
        """scale=1e-28 with data of order 1e-27 equals scale=None at rtol 1e-10."""
        _n, k, base, design, data = self._problem()
        sigma = jnp.asarray(0.03 * base)
        args = dict(
            sigma=sigma,
            prior_mean=jnp.zeros(k),
            prior_precision=jnp.eye(k) * 1e-3,
        )
        scaled = marginalize_linear_nuisance(
            jnp.asarray(data), jnp.asarray(base), jnp.asarray(design), scale=1e-28, **args
        )
        plain = marginalize_linear_nuisance(
            jnp.asarray(data), jnp.asarray(base), jnp.asarray(design), **args
        )
        np.testing.assert_allclose(float(scaled.log_like), float(plain.log_like), rtol=1e-10)
        np.testing.assert_allclose(
            np.asarray(scaled.coef_hat), np.asarray(plain.coef_hat), rtol=1e-10, atol=0
        )

    @pytest.mark.parametrize("gaussian", [True, False])
    def test_float32_scaled_value_and_gradient_finite_and_nonzero(self, gaussian):
        """In float32 the scaled call is finite with a finite nonzero model0 gradient.

        The design is O(10) and sigma is O(1e-27), so X/sigma is O(1e28): its square
        overflows float32, and the scale brings the whitened design back to O(1). The
        Gaussian case also exercises the prior block, which underflows at this scale.
        """
        n = 60
        phase = np.linspace(0.0, 4.0, n)
        model0_np = 1e-27 * (1.0 + 0.2 * np.sin(phase))
        data_np = model0_np * (1.0 + 0.05 * np.sin(3.0 * phase))
        design_np = 10.0 * np.stack([np.ones(n), np.cos(phase)], axis=1)
        sigma_np = 0.1 * model0_np

        with jax.enable_x64(False):
            data = jnp.asarray(data_np, dtype=jnp.float32)
            design = jnp.asarray(design_np, dtype=jnp.float32)
            sigma = jnp.asarray(sigma_np, dtype=jnp.float32)
            m0 = jnp.asarray(model0_np, dtype=jnp.float32)
            prec = jnp.eye(2, dtype=jnp.float32) if gaussian else None

            def log_like(model0):
                return marginalize_linear_nuisance(
                    data, model0, design, sigma=sigma, prior_precision=prec, scale=1e-28
                ).log_like

            value = log_like(m0)
            grad = jax.grad(log_like)(m0)
            assert jnp.isfinite(value)
            assert bool(jnp.all(jnp.isfinite(grad)))
            assert bool(jnp.any(grad != 0.0))


class TestCalibrationBasis:
    """calibration_basis is the Chebyshev matrix behind calibration_polynomial."""

    def test_agrees_with_calibration_polynomial(self):
        """1 + basis @ c equals calibration_polynomial(wave, c) on 200 random draws."""
        wave = jnp.linspace(3800.0, 9000.0, 257)
        wmin, wmax = float(wave[0]), float(wave[-1])
        rng = np.random.default_rng(5)
        for _ in range(200):
            n_poly = int(rng.integers(1, 6))
            coeffs = jnp.asarray(rng.normal(size=n_poly))
            basis = calibration_basis(wave, wmin, wmax, n_poly)
            got = 1.0 + basis @ coeffs
            want = calibration_polynomial(wave, coeffs, wmin, wmax)
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-13, rtol=0)


class TestTransforms:
    """jit and vmap leave the primitive's values unchanged."""

    def _problem(self, seed):
        rng = np.random.default_rng(seed)
        n, k = 30, 2
        design = jnp.asarray(rng.normal(size=(n, k)))
        model0 = jnp.asarray(rng.normal(size=n))
        data = model0 + jnp.asarray(rng.normal(size=n))
        sigma = jnp.asarray(0.5 + 0.5 * rng.random(n))
        prec = jnp.eye(k) * 2.0
        return data, model0, design, sigma, prec

    def test_jit_equals_eager(self):
        """Jitted and eager results agree at rtol 1e-12."""
        data, model0, design, sigma, prec = self._problem(seed=1)

        def run(d):
            return marginalize_linear_nuisance(
                d, model0, design, sigma=sigma, prior_precision=prec
            ).log_like

        eager = run(data)
        jitted = jax.jit(run)(data)
        np.testing.assert_allclose(float(jitted), float(eager), rtol=1e-12, atol=0)

    def test_vmap_equals_loop(self):
        """vmap over four problems equals a Python loop over the same problems."""
        problems = [self._problem(seed=s) for s in range(4)]
        data = jnp.stack([p[0] for p in problems])
        model0 = jnp.stack([p[1] for p in problems])
        design = jnp.stack([p[2] for p in problems])
        sigma = jnp.stack([p[3] for p in problems])
        prec = problems[0][4]

        def run(d, m, x, s):
            return marginalize_linear_nuisance(d, m, x, sigma=s, prior_precision=prec).log_like

        batched = jax.vmap(run)(data, model0, design, sigma)
        looped = jnp.stack([run(data[i], model0[i], design[i], sigma[i]) for i in range(4)])
        np.testing.assert_allclose(np.asarray(batched), np.asarray(looped), rtol=1e-12, atol=0)


def _mixed_problem(seed):
    """Two-column problem: column 0 Gaussian N(0, 1), column 1 flat."""
    rng = np.random.default_rng(seed)
    n = 40
    design = np.stack([rng.uniform(0.5, 1.5, n), rng.uniform(0.2, 1.0, n)], axis=1)
    sigma = np.full(n, 0.1)
    model0 = np.zeros(n)
    data = 0.7 * design[:, 0] + 0.3 * design[:, 1] + rng.normal(0.0, 0.1, n)
    return data, model0, design, sigma


def _mixed_grid_log_evidence(data, model0, design, sigma, a2_center):
    """Log of the integral over a1 (Gaussian N(0, 1)) and a2 (flat) on a dense grid."""
    a1 = np.linspace(-8.0, 8.0, 1601)
    a2 = np.linspace(a2_center - 0.6, a2_center + 0.6, 2401)
    log_prior_a1 = -0.5 * a1**2 - 0.5 * LOG_2PI
    rows = []
    for a1_val, lp in zip(a1, log_prior_a1, strict=True):
        pred = model0[None, :] + a1_val * design[None, :, 0] + a2[:, None] * design[None, :, 1]
        resid = (data[None, :] - pred) / sigma[None, :]
        rows.append(lp - 0.5 * np.sum(resid**2, axis=1))
    terms = np.concatenate(rows)
    top = np.max(terms)
    return top + np.log(np.sum(np.exp(terms - top)))


class TestMixedFlatGaussianColumns:
    """A zero prior precision on a column makes that column flat, alongside Gaussian columns."""

    def test_all_zero_precision_equals_flat_branch(self):
        """A zero precision matrix is the flat prior: the result equals prior_precision=None."""
        data, model0, design, sigma = _mixed_problem(seed=3)
        flat = marginalize_linear_nuisance(
            jnp.asarray(data),
            jnp.asarray(model0),
            jnp.asarray(design),
            sigma=jnp.asarray(sigma),
            scale=0.05,
        )
        zero = marginalize_linear_nuisance(
            jnp.asarray(data),
            jnp.asarray(model0),
            jnp.asarray(design),
            sigma=jnp.asarray(sigma),
            prior_precision=jnp.zeros((2, 2)),
            scale=0.05,
        )
        np.testing.assert_allclose(float(zero.log_like), float(flat.log_like), rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            np.asarray(zero.coef_hat), np.asarray(flat.coef_hat), rtol=1e-12
        )

    def test_mixed_difference_matches_grid_integral(self):
        """Differences of the mixed log-likelihood across data sets match a dense-grid integral."""
        precision = jnp.diag(jnp.array([1.0, 0.0]))
        values = []
        grid = []
        for seed in (3, 5):
            data, model0, design, sigma = _mixed_problem(seed=seed)
            got = marginalize_linear_nuisance(
                jnp.asarray(data),
                jnp.asarray(model0),
                jnp.asarray(design),
                sigma=jnp.asarray(sigma),
                prior_precision=precision,
                scale=0.05,
            )
            values.append(float(got.log_like))
            a2_center = float(np.linalg.lstsq(design, data, rcond=None)[0][1])
            grid.append(_mixed_grid_log_evidence(data, model0, design, sigma, a2_center))
        np.testing.assert_allclose(values[0] - values[1], grid[0] - grid[1], rtol=1e-6, atol=0)

    def test_mixed_result_is_finite_and_scale_invariant(self):
        """The mixed prior stays finite and independent of the coefficient scale."""
        data, model0, design, sigma = _mixed_problem(seed=7)
        precision = jnp.diag(jnp.array([2.0, 0.0]))
        results = [
            marginalize_linear_nuisance(
                jnp.asarray(data),
                jnp.asarray(model0),
                jnp.asarray(design),
                sigma=jnp.asarray(sigma),
                prior_precision=precision,
                scale=s,
            ).log_like
            for s in (1.0, 1e-3)
        ]
        assert np.isfinite(float(results[0]))
        np.testing.assert_allclose(float(results[0]), float(results[1]), rtol=1e-10, atol=0)
