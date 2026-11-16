# SPDX-License-Identifier: BSD-3-Clause
"""Student-t noise with a marginalized linear nuisance: limits, robustness and gradients.

The value is a Laplace approximation at the iteratively reweighted mode (see
:func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`); its error against
quadrature is in ``test_student_t_marginal_quadrature.py``. These cells check what holds
exactly: the limit of large ``dof``, robustness to an outlier, the gradients (first and second
order, float32 and float64), the convergence diagnostic, masks, and the Fitter wiring.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference import channels
from tengri.inference.channels import CalibrationSpec, NoiseSpec, Predictions, score
from tengri.observation import linear_marginal
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tests.contract import _channel_fixtures as fixtures
from tests.contract.test_channel_engine_sweep import (
    DOF,
    N_BAND,
    N_PIX,
    _synthetic,
    check_gradients,
)

pytestmark = pytest.mark.contract

LARGE_DOF = 1.0e6
CELLS = (
    ("calibration", "cal_marg"),
    ("calibration", "cal_marg_segments"),
    ("calibration", "cal_marg_banded"),
    ("calibration", "cal_marg_dense"),
    ("calibration", "floor_cal_marg"),
    ("lines", "lines_marginalized"),
    ("lines", "lines_marginalized_model"),
    ("lines", "lines_marginalized_cloudy"),
    ("lines", "lines_marginalized_banded"),
    ("lines", "lines_marginalized_dense"),
    ("lines", "floor_lines_marginalized"),
    ("calibration_lines", "cal_marg_lines"),
    ("calibration_lines", "cal_marg_lines_segments"),
    ("calibration_lines", "cal_marg_lines_segments_model"),
    ("calibration_lines", "cal_marg_lines_cloudy"),
    ("calibration_lines", "cal_marg_lines_banded"),
    ("calibration_lines", "floor_cal_marg_lines"),
)
GRADIENT_CELLS = (
    ("calibration", "cal_marg"),
    ("calibration", "floor_cal_marg"),
    ("calibration", "cal_marg_banded"),
    ("lines", "lines_marginalized"),
    ("lines", "floor_lines_marginalized"),
    ("calibration_lines", "cal_marg_lines"),
    ("calibration_lines", "cal_marg_lines_segments_model"),
)


def _heavy(noise, dof):
    """The noise spec with Student-t noise of ``dof`` degrees of freedom."""
    return replace(noise, student_t=True, dof=dof)


def _gaussian_and_heavy(channel, label, dof):
    preds, data, noise = _synthetic(channel, label)
    return preds, data, noise, _heavy(noise, dof)


@pytest.mark.parametrize(("channel", "label"), CELLS)
def test_the_heavy_tailed_marginal_tends_to_the_gaussian_marginal_as_one_over_dof(channel, label):
    """A Student-t differs from a Gaussian by O(1/dof) per datum, and so do the marginals."""
    preds, data, noise = _synthetic(channel, label)
    gaussian = float(score(preds, data, noise))
    gap = {
        dof: abs(float(score(preds, data, _heavy(noise, dof))) - gaussian) for dof in (1e4, 1e6)
    }
    assert gap[1e6] < 2e-3
    assert 30.0 < gap[1e4] / gap[1e6] < 300.0, gap


@pytest.mark.parametrize(("channel", "label"), CELLS)
def test_the_heavy_tailed_marginal_selects_the_marginal_channel_and_is_finite(channel, label):
    preds, data, _noise, heavy = _gaussian_and_heavy(channel, label, DOF)
    channels.validate(heavy, data.keys())
    assert [c.name for c in channels.selected_channels(data, heavy)] == [channel]
    assert np.isfinite(float(score(preds, data, heavy)))


def _with_outlier(data, index, sigmas):
    moved = np.array(data["data"], dtype=float)
    moved[index] += sigmas * float(np.asarray(data["noise"])[index])
    return {**data, "data": moved}


@pytest.mark.parametrize(("channel", "label"), CELLS[:1] + CELLS[5:6] + CELLS[11:12])
def test_one_pixel_moved_by_fifty_sigma_costs_the_heavy_tailed_marginal_a_bounded_amount(
    channel, label
):
    preds, data, noise, heavy = _gaussian_and_heavy(channel, label, DOF)
    moved = _with_outlier(data, 7, 50.0)
    gaussian = float(score(preds, data, noise)) - float(score(preds, moved, noise))
    student = float(score(preds, data, heavy)) - float(score(preds, moved, heavy))
    assert gaussian > 500.0
    assert 0.0 < student < 40.0
    assert student < gaussian / 20.0


@pytest.mark.parametrize(("channel", "label"), GRADIENT_CELLS)
def test_gradients_match_finite_differences_in_the_model_the_floor_and_the_design(channel, label):
    preds, data, _noise, heavy = _gaussian_and_heavy(channel, label, DOF)
    grads = check_gradients(preds, data, heavy)
    assert np.all(np.isfinite(np.asarray(grads["spec_fnu"])))


def _float32(tree):
    return jax.tree_util.tree_map(
        lambda x: (
            np.asarray(x, np.float32)
            if isinstance(x, np.ndarray | jnp.ndarray) and np.asarray(x).dtype.kind == "f"
            else x
        ),
        tree,
    )


@pytest.mark.parametrize(("channel", "label"), GRADIENT_CELLS[:1] + GRADIENT_CELLS[3:4])
def test_float32_value_and_gradient_are_finite_and_agree_with_float64(channel, label):
    preds, data, _noise, heavy = _gaussian_and_heavy(channel, label, DOF)
    reference = float(score(preds, data, heavy))
    grad64 = jax.grad(lambda p: score(p, data, heavy))(
        replace(preds, spec_fnu=jnp.asarray(preds.spec_fnu))
    )
    with jax.enable_x64(False):
        p32, d32 = _float32(preds), _float32(data)
        value = score(p32, d32, heavy)
        grad32 = jax.grad(lambda p: score(p, d32, heavy))(p32)
    assert value.dtype == jnp.float32
    assert np.isfinite(float(value))
    assert abs(float(value) - reference) < 1e-3 * abs(reference)
    g32, g64 = np.asarray(grad32.spec_fnu), np.asarray(grad64.spec_fnu)
    assert np.all(np.isfinite(g32))
    np.testing.assert_allclose(g32, g64, rtol=2e-2, atol=2e-2 * np.max(np.abs(g64)))


def _primitive_inputs(n_data=30, n_col=3, seed=0):
    rng = np.random.default_rng(seed)
    sigma = 0.1 * (1.0 + 0.3 * rng.random(n_data))
    model0 = 1.0 + 0.2 * rng.random(n_data)
    design = model0[:, None] * rng.standard_normal((n_data, n_col)) * 0.3
    data = (
        model0
        + design @ np.array([0.1, -0.05, 0.02][:n_col])
        + sigma * rng.standard_normal(n_data)
    )
    return data, model0, design, sigma


@pytest.mark.parametrize("dof", (3.0, 30.0, 1.0e3, 1.0e6))
def test_weights_and_their_derivatives_stay_finite_in_float32(dof):
    """Weights for an exact-fit pixel (z = 0), ordinary pixels and a 1000-sigma pixel."""
    data, model0, design, sigma = _primitive_inputs()
    data[0] = model0[0]
    data[1] = model0[1] + 1000.0 * sigma[1]
    with jax.enable_x64(False):
        args = [jnp.asarray(a, jnp.float32) for a in (data, model0, design, sigma)]

        def value(d, m, x, s):
            return marginalize_linear_nuisance(
                d, m, x, sigma=s, prior_precision=jnp.eye(3, dtype=jnp.float32), dof=dof
            ).log_like

        result = value(*args)
        grads = jax.grad(value, argnums=(0, 1, 2, 3))(*args)
    assert np.isfinite(float(result))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)


def test_the_convergence_diagnostic_is_small_at_the_chosen_step_count_and_large_for_too_few():
    for covariance in ("diagonal", "banded"):
        data, model0, design, sigma = _primitive_inputs()
        data[5] += 50.0 * sigma[5]
        kw = _noise_kwargs(sigma, covariance)
        result = marginalize_linear_nuisance(data, model0, design, dof=3.0, **kw)
        assert float(result.weight_change) < 1e-8
        assert marginalize_linear_nuisance(data, model0, design, **kw).weight_change is None


def test_too_few_iterations_are_visible_in_the_diagnostic(monkeypatch):
    data, model0, design, sigma = _primitive_inputs()
    data[5] += 50.0 * sigma[5]
    monkeypatch.setattr(linear_marginal, "_IRLS_STEPS", 1)
    short = marginalize_linear_nuisance(data, model0, design, sigma=sigma, dof=3.0)
    assert float(short.weight_change) > 1e-3
    monkeypatch.setattr(linear_marginal, "_IRLS_STEPS", 60)
    long = marginalize_linear_nuisance(data, model0, design, sigma=sigma, dof=3.0)
    assert float(long.weight_change) < 1e-12
    assert abs(float(short.log_like) - float(long.log_like)) > 1e-5


def _noise_kwargs(sigma, covariance):
    if covariance == "diagonal":
        return {"sigma": sigma}
    from tengri.observation.banded import banded_cholesky, dense_to_banded

    n = sigma.size
    cov = np.diag(sigma**2)
    for lag, rho in ((1, 0.3), (2, 0.1)):
        idx = np.arange(n - lag)
        cov[idx, idx + lag] = rho * sigma[idx] * sigma[idx + lag]
        cov[idx + lag, idx] = cov[idx, idx + lag]
    return {"chol_lower": banded_cholesky(dense_to_banded(cov))}


@pytest.mark.parametrize("covariance", ("diagonal", "banded"))
def test_the_mode_is_differentiated_implicitly_to_second_order(covariance):
    data, model0, design, sigma = _primitive_inputs()
    data[5] += 20.0 * sigma[5]
    kw = _noise_kwargs(sigma, covariance)

    def value(m):
        return marginalize_linear_nuisance(
            data, m, design, dof=4.0, prior_precision=jnp.eye(3), **kw
        ).log_like

    x = jnp.asarray(model0)
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(x.shape))
    eps = 1e-6
    first = jax.grad(value)(x)
    central = (value(x + eps * direction) - value(x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(float(first @ direction), float(central), rtol=1e-7)
    forward = jax.jvp(value, (x,), (direction,))[1]
    np.testing.assert_allclose(float(forward), float(first @ direction), rtol=1e-10)
    hessian = jax.hessian(value)(x)
    gradient_difference = (
        jax.grad(value)(x + eps * direction) - jax.grad(value)(x - eps * direction)
    ) / (2 * eps)
    np.testing.assert_allclose(
        np.asarray(hessian @ direction),
        np.asarray(gradient_difference),
        rtol=1e-5,
        atol=1e-5 * float(jnp.max(jnp.abs(gradient_difference))),
    )
    assert float(jnp.max(jnp.abs(hessian - hessian.T))) < 1e-8 * float(jnp.max(jnp.abs(hessian)))


def test_a_start_that_diverges_does_not_poison_the_value_or_gradient_of_the_other():
    """A flat prior ignores its mean, but the weights at that mean overflow: that start is NaN."""
    n_data = 30
    grid = np.linspace(-1.0, 1.0, n_data)
    design = jnp.asarray(np.stack([np.ones(n_data), grid], axis=1))
    data = jnp.asarray(1.0 + 0.3 * grid + 0.05 * np.random.default_rng(0).standard_normal(n_data))
    sigma = jnp.full(n_data, 0.05)

    def marginal(d, mean):
        return marginalize_linear_nuisance(
            d, jnp.zeros(n_data), design, sigma=sigma, prior_mean=jnp.array([mean, 0.0]), dof=3.0
        )

    value = marginal(data, 1.0e200).log_like
    assert float(value) == pytest.approx(float(marginal(data, 0.0).log_like), abs=1e-9)
    grad = jax.grad(lambda d: marginal(d, 1.0e200).log_like)(data)
    reference = jax.grad(lambda d: marginal(d, 0.0).log_like)(data)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(reference), rtol=1e-8, atol=1e-10)


def test_a_masked_pixel_is_the_pixel_removed():
    data, model0, design, sigma = _primitive_inputs()
    data[4] += 30.0 * sigma[4]
    keep = np.ones(data.size, dtype=bool)
    keep[[4, 9, 17]] = False
    kw = {"dof": 3.0, "prior_precision": jnp.eye(3)}
    masked = marginalize_linear_nuisance(
        data, model0, design, sigma=sigma, pix_mask=keep.astype(float), **kw
    )
    removed = marginalize_linear_nuisance(
        data[keep], model0[keep], design[keep], sigma=sigma[keep], **kw
    )
    np.testing.assert_allclose(float(masked.log_like), float(removed.log_like), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(masked.coef_hat), np.asarray(removed.coef_hat), rtol=1e-10
    )


def test_an_absent_band_is_the_band_removed_with_a_heavy_tailed_marginal_spectrum():
    rng = np.random.default_rng(5)
    wave = np.linspace(4000.0, 8000.0, N_PIX)
    spec = rng.uniform(1.0, 2.0, N_PIX)
    spec_data = spec * (1.0 + 0.1 * rng.standard_normal(N_PIX))
    spec_err = rng.uniform(0.05, 0.15, N_PIX)
    phot = rng.uniform(1.0, 2.0, N_BAND)
    phot_data = phot * (1.0 + 0.1 * rng.standard_normal(N_BAND))
    phot_err = rng.uniform(0.05, 0.15, N_BAND)
    calibration = CalibrationSpec(wavelength=wave, n_poly=2, prior_sigma=0.3)
    absent = 2

    def noise(n_phot):
        return NoiseSpec(
            data_type="joint",
            n_phot=n_phot,
            student_t=True,
            dof=DOF,
            f_cal_param="noise_frac_cal",
            calibration=calibration,
        )

    params = {"noise_frac_cal": 0.05}
    zeroed = {
        "data": np.r_[np.where(np.arange(N_BAND) == absent, 0.0, phot_data), spec_data],
        "noise": np.r_[np.where(np.arange(N_BAND) == absent, 0.0, phot_err), spec_err],
        "presence": np.where(np.arange(N_BAND) == absent, 0.0, 1.0),
    }
    keep = np.arange(N_BAND) != absent
    removed = {
        "data": np.r_[phot_data[keep], spec_data],
        "noise": np.r_[phot_err[keep], spec_err],
    }
    full = score(Predictions(phot_fnu=phot, spec_fnu=spec, params=params), zeroed, noise(N_BAND))
    short = score(
        Predictions(phot_fnu=phot[keep], spec_fnu=spec, params=params), removed, noise(N_BAND - 1)
    )
    np.testing.assert_allclose(float(full), float(short), rtol=1e-12)


@pytest.mark.parametrize(
    "kind", ("spec_student_t_cal", "spec_student_t_lines", "spec_student_t_cal_lines")
)
def test_the_fitter_scores_heavy_tailed_marginals_through_the_scorer_and_the_adapters(kind):
    from tengri.inference.context import InferenceContext
    from tengri.inference.likelihood import build_base_likelihood, build_likelihood_extras

    fitter = fixtures.fitter_of(kind)
    term, physical = fixtures.build_term(fitter)
    data_args = fitter._data_args
    preds = term.predictions(physical, data_args)
    expected = float(term.log_prob(physical, data_args))
    assert np.isfinite(expected)
    np.testing.assert_allclose(expected, float(score(preds, data_args, term.noise)), rtol=1e-12)
    context = InferenceContext.from_target(fitter)
    adapter = build_base_likelihood(context)
    assert getattr(adapter, "dof", None) is not None or hasattr(adapter, "likelihoods")
    got = float(adapter.log_prob(preds.as_mapping(), physical, data_args))
    extras = build_likelihood_extras(context)
    assert extras is None or isinstance(extras, list | tuple)
    np.testing.assert_allclose(got, expected, rtol=1e-10)
