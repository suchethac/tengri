# SPDX-License-Identifier: BSD-3-Clause
"""#3333: pointwise log-likelihood, WAIC, PSIS-LOO and elpd comparison.

The criteria are checked against a conjugate normal model, where the exact
leave-one-out predictive density is known in closed form, and the pointwise
evaluator against the Gaussian formula written out directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import tengri
from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.inference.predictive import (
    ElpdResult,
    compare_models,
    loo,
    pointwise_log_likelihood,
    waic,
)
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.contract

# ── Conjugate normal model: y_i ~ N(mu, 1), mu ~ N(0, tau^2) ──────

_TAU = 10.0
_N = 40
_S = 4000


def _posterior(y):
    precision = y.size + 1.0 / _TAU**2
    return np.sum(y) / precision, 1.0 / precision


def _normal_case(seed=0, n_draws=_S):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.7, 1.0, _N)
    mean, var = _posterior(y)
    mu = rng.normal(mean, np.sqrt(var), n_draws)
    log_lik = stats.norm.logpdf(y[None, :], mu[:, None], 1.0)
    exact = np.empty(_N)
    for i in range(_N):
        m, v = _posterior(np.delete(y, i))
        exact[i] = stats.norm.logpdf(y[i], m, np.sqrt(1.0 + v))
    return log_lik, exact, float(np.var(y))


def test_loo_matches_the_exact_leave_one_out_density():
    log_lik, exact, scatter = _normal_case()
    result = loo(log_lik)
    np.testing.assert_allclose(result.pointwise, exact, atol=5.0e-3)
    assert result.elpd == pytest.approx(np.sum(exact), abs=0.05)
    # One parameter, unit noise: the effective number is the sample variance of y.
    assert result.p_eff == pytest.approx(scatter, abs=0.1)
    assert result.n_bad_k == 0
    assert result.criterion == "loo" and result.n_draws == _S and result.n_data == _N


def test_waic_agrees_with_exact_leave_one_out_for_a_regular_model():
    log_lik, exact, scatter = _normal_case()
    result = waic(log_lik)
    assert result.elpd == pytest.approx(np.sum(exact), abs=0.05)
    assert result.p_eff == pytest.approx(scatter, abs=0.1)
    assert result.pareto_k is None and result.n_bad_k == 0
    assert result.deviance == -2.0 * result.elpd


def test_waic_formula_on_a_small_matrix():
    matrix = np.log(np.array([[0.2, 0.5], [0.4, 0.5], [0.6, 0.5]]))
    result = waic(matrix)
    lppd = np.log(np.array([0.4, 0.5]))
    penalty = np.array([np.var(matrix[:, 0], ddof=1), 0.0])
    np.testing.assert_allclose(result.pointwise, lppd - penalty, rtol=1e-12)
    assert result.se == pytest.approx(np.sqrt(2.0 * np.var(lppd - penalty)))
    assert result.p_eff == pytest.approx(np.sum(penalty))


def test_an_influential_point_has_the_largest_pareto_k():
    """A point far from the rest dominates its own importance ratios."""
    rng = np.random.default_rng(1)
    y = np.append(rng.normal(0.0, 1.0, 12), 9.0)
    mean, var = np.sum(y) / (y.size + 0.01), 1.0 / (y.size + 0.01)
    mu = rng.normal(mean, np.sqrt(var), 2000)
    result = loo(stats.norm.logpdf(y[None, :], mu[:, None], 1.0))
    assert int(np.argmax(result.pareto_k)) == 12
    assert result.pareto_k[12] > np.median(result.pareto_k) + 0.2


def test_loo_uses_the_shared_psis_routine():
    """The weights are those of ``backends._psis.psis``, not a second smoother."""
    from scipy.special import logsumexp

    from tengri.inference.backends._psis import psis

    log_lik, _, _ = _normal_case(n_draws=500)
    result = loo(log_lik)
    for i in (0, 7, 39):
        ref = psis(-log_lik[:, i])
        assert result.pareto_k[i] == ref.pareto_k
        assert result.pointwise[i] == logsumexp(ref.log_weights + log_lik[:, i])


def test_unfitted_tails_count_as_unreliable():
    """Identical ratios leave no tail to fit: k is nan and the point is flagged."""
    flat = np.tile(np.log([0.3, 0.4]), (50, 1))
    result = loo(flat)
    assert np.all(np.isnan(result.pareto_k))
    assert result.n_bad_k == 2
    np.testing.assert_allclose(result.pointwise, np.log([0.3, 0.4]))


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.zeros(5), "shape"),
        (np.zeros((1, 5)), "two posterior draws"),
        (np.vstack([[0.0, np.nan]] + [[0.0, 0.0]] * 11), "non-finite"),
    ],
)
def test_bad_matrices_are_refused(bad, match):
    for estimator in (waic, loo):
        with pytest.raises(ValueError, match=match):
            estimator(bad)


# ── Comparison ────────────────────────────────────────────────────


def _two_models(seed=0):
    """Data from N(0.7, 1); model A fits the mean, model B pins it at zero."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.7, 1.0, 200)
    mean, var = np.sum(y) / (y.size + 0.01), 1.0 / (y.size + 0.01)
    mu = rng.normal(mean, np.sqrt(var), 2000)
    fitted = stats.norm.logpdf(y[None, :], mu[:, None], 1.0)
    pinned = np.tile(stats.norm.logpdf(y, 0.0, 1.0), (2000, 1))
    return fitted, pinned


@pytest.mark.parametrize("criterion", ["loo", "waic"])
def test_compare_prefers_the_generating_model(criterion):
    fitted, pinned = _two_models()
    rows = compare_models({"pinned": pinned, "fitted": fitted}, criterion=criterion)
    assert [row["name"] for row in rows] == ["fitted", "pinned"]
    assert rows[0]["delta_elpd"] == 0.0 and rows[0]["delta_se"] == 0.0
    assert rows[1]["delta_elpd"] < -5.0 * rows[1]["delta_se"]
    diff = rows[1]["elpd"] - rows[0]["elpd"]
    assert rows[1]["delta_elpd"] == pytest.approx(diff)
    # Paired: tighter than the quadrature sum of the two standard errors.
    assert rows[1]["delta_se"] < np.hypot(rows[0]["se"], rows[1]["se"])
    assert rows[0]["log_evidence"] is None


def test_redundant_parameter_costs_about_one_nat():
    """A parameter the data do not need lowers the elpd by order one, not by many."""
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.0, 200)
    prec = y.size + 0.01
    mu = rng.normal(np.sum(y) / prec, np.sqrt(1.0 / prec), 2000)
    one = stats.norm.logpdf(y[None, :], mu[:, None], 1.0)
    # The same mean plus a slope on an uninformative covariate, drawn from its posterior.
    x = rng.normal(0.0, 1.0, y.size)
    slope_var = 1.0 / (np.sum(x**2) + 0.01)
    slope = rng.normal(np.sum(x * y) * slope_var, np.sqrt(slope_var), 2000)
    two = stats.norm.logpdf(y[None, :], mu[:, None] + slope[:, None] * x[None, :], 1.0)
    rows = compare_models({"one": one, "two": two})
    assert [row["name"] for row in rows] == ["one", "two"]
    assert -2.5 < rows[1]["delta_elpd"] < 0.0


def test_compare_refusals():
    fitted, pinned = _two_models()
    with pytest.raises(ValueError, match="criterion"):
        compare_models({"a": fitted, "b": pinned}, criterion="aic")
    with pytest.raises(ValueError, match="at least two"):
        compare_models({"a": fitted})
    with pytest.raises(ValueError, match="different numbers of data points"):
        compare_models({"a": fitted, "b": pinned[:, :-1]})
    with pytest.raises(ValueError, match="carries a 'waic' result"):
        compare_models({"a": waic(fitted), "b": pinned}, criterion="loo")


def test_compare_accepts_precomputed_results():
    fitted, pinned = _two_models()
    rows = compare_models({"a": loo(fitted), "b": loo(pinned)})
    assert isinstance(loo(fitted), ElpdResult)
    assert rows[0]["name"] == "a" and "n_bad_k" in rows[0]


# ── Pointwise log-likelihood from a fitter ────────────────────────

_WAVE = np.linspace(4000.0, 7500.0, 300)
_Z = 0.1


def _ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    flux = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (3, 20, 100))) * 1e-3 + 1e-5
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=flux,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _spectral_fit(eline_mode="off", n_draws=64):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=500.0,
        eline_mode=eline_mode,
        eline_catalog=LineList.from_names(["Hbeta", "Halpha"]),
        eline_prior_sigma=1.0e-14,
    )
    model = SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Fixed(_Z),
    )
    rng = np.random.default_rng(11)
    truth = np.asarray(model.predict_spectrum({"sfh_dpl_log_total_mass": 10.0}))
    noise = 0.02 * np.abs(truth) + 1e-3 * np.abs(truth).max()
    data = truth + rng.normal(0.0, 1.0, truth.size) * noise
    fitter = Fitter(
        model, jnp.asarray(data), jnp.asarray(noise), data_type="spectroscopy", approx=None
    )
    samples = {"sfh_dpl_log_total_mass": rng.normal(10.0, 2.0e-3, n_draws)}
    return model, fitter, samples, data, noise


def test_pointwise_terms_are_the_gaussian_formula_and_sum_to_the_likelihood():
    model, fitter, samples, data, noise = _spectral_fit()
    matrix = pointwise_log_likelihood(fitter, samples)
    assert matrix.shape == (64, _WAVE.size)
    log_lik = build_loglikelihood_fn(fitter)
    for s in (0, 17, 63):
        params = {"sfh_dpl_log_total_mass": float(samples["sfh_dpl_log_total_mass"][s])}
        predicted = np.asarray(model.predict_spectrum(params))
        direct = stats.norm.logpdf(data, predicted, noise)
        np.testing.assert_allclose(matrix[s], direct, rtol=1e-8, atol=1e-8)
        # The fitter's Gaussian is -chi2/2; the pointwise terms carry the normalization.
        total = float(log_lik(params, fitter._data_args))
        log_norm = -np.sum(np.log(noise)) - 0.5 * noise.size * np.log(2.0 * np.pi)
        assert np.sum(matrix[s]) == pytest.approx(total + log_norm, rel=1e-9)


def test_draw_subset_and_missing_parameters():
    _, fitter, samples, _, _ = _spectral_fit()
    subset = pointwise_log_likelihood(fitter, samples, n_draws=10, key=jax.random.PRNGKey(2))
    assert subset.shape == (10, _WAVE.size)
    full = pointwise_log_likelihood(fitter, samples)
    rows = {tuple(np.round(r, 9)) for r in full}
    assert all(tuple(np.round(r, 9)) in rows for r in subset)
    with pytest.raises(ValueError, match="lack draws of the free parameters"):
        pointwise_log_likelihood(fitter, {})


def _line_mock(lines, *, eline_prior_type="flat", sample=True):
    """A marginalized-line fit on a mock with H-beta and H-alpha in emission."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=500.0,
        eline_mode="marginalized",
        eline_catalog=LineList.from_names(lines),
        eline_prior_sigma=1.0e-14,
        eline_prior_type=eline_prior_type,
    )
    model = SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Fixed(_Z),
    )
    continuum = np.asarray(model.predict_spectrum({"sfh_dpl_log_total_mass": 10.0}))
    lam = np.array([4862.68, 6564.61]) * (1 + _Z)
    width = lam / (2.355 * 500.0)
    gauss = np.exp(-0.5 * ((_WAVE[:, None] - lam) / width) ** 2) / (np.sqrt(2 * np.pi) * width)
    amps = np.array([0.05, 0.15]) * np.abs(continuum).max() * 20.0
    noise = 0.02 * np.abs(continuum) + 1e-3 * np.abs(continuum).max()
    rng = np.random.default_rng(12)
    data = continuum + gauss @ amps + rng.normal(0.0, 1.0, _WAVE.size) * noise
    fitter = Fitter(
        model, jnp.asarray(data), jnp.asarray(noise), data_type="spectroscopy", approx=None
    )
    draws = (
        _posterior_draws(fitter, rng) if sample else {"sfh_dpl_log_total_mass": np.full(20, 10.0)}
    )
    return model, fitter, draws, data, noise


def _posterior_draws(fitter, rng, n_draws=400):
    """Draws of the one free parameter from its exact posterior, by inverse CDF on a grid.

    The prior is flat, so the posterior is the marginal likelihood the fitter evaluates.
    """
    log_lik = build_loglikelihood_fn(fitter)
    coarse = np.linspace(8.0, 12.0, 401)
    values = np.array(
        [float(log_lik({"sfh_dpl_log_total_mass": m}, fitter._data_args)) for m in coarse]
    )
    center = coarse[int(np.argmax(values))]
    grid = np.linspace(center - 0.02, center + 0.02, 801)
    values = np.array(
        [float(log_lik({"sfh_dpl_log_total_mass": m}, fitter._data_args)) for m in grid]
    )
    weights = np.exp(values - values.max())
    assert weights[0] < 1e-6 and weights[-1] < 1e-6, "posterior is wider than the grid"
    cdf = np.cumsum(weights) / np.sum(weights)
    return {"sfh_dpl_log_total_mass": np.interp(rng.uniform(size=n_draws), cdf, grid)}


def test_conditional_line_moments_match_the_marginalization_primitive():
    """The Hessian route reproduces marginalize_emission_lines' amplitude posterior."""
    from tengri.inference.predictive import _conditional_line_moments, _fitted_line_sibling
    from tengri.observation.eline_marginalization import (
        apply_doublet_constraints,
        build_eline_design_matrix,
        marginalize_emission_lines,
    )

    model, fitter, _, data, noise = _line_mock(["Hbeta", "Halpha"])
    sibling = _fitted_line_sibling(fitter)
    names = tuple(sibling._eline_amplitude_names)
    log_lik = build_loglikelihood_fn(sibling)
    theta = {"sfh_dpl_log_total_mass": 10.0}
    mean, cov = _conditional_line_moments(
        lambda p: log_lik(p, sibling._data_args), theta, names, 1.0e-14
    )
    catalog = model.observation.spectroscopy.effective_catalog
    design = apply_doublet_constraints(
        build_eline_design_matrix(jnp.asarray(_WAVE), catalog.wavelengths, 500.0, _Z),
        catalog.build_constraint_matrix(),
    )
    residual = jnp.asarray(data) - model.predict_spectrum(theta)
    _, a_hat, a_cov = marginalize_emission_lines(
        residual, jnp.asarray(noise), design, prior_sigma=1.0e-14
    )
    np.testing.assert_allclose(np.asarray(mean), np.asarray(a_hat), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(cov), np.asarray(a_cov), rtol=1e-6)


def test_marginalized_lines_give_pointwise_terms_of_the_joint_posterior():
    _, fitter, samples, _, _ = _line_mock(["Hbeta", "Halpha"])
    matrix = pointwise_log_likelihood(fitter, samples)
    assert matrix.shape == (400, _WAVE.size)
    result = loo(matrix)
    # One stellar mass plus two amplitudes.
    assert result.p_eff == pytest.approx(3.0, abs=0.6)
    again = pointwise_log_likelihood(fitter, samples)
    np.testing.assert_array_equal(matrix, again)


def test_compare_prefers_the_catalog_with_the_emitting_line():
    """Data with H-alpha in emission: the catalog that omits it loses.

    The loss sits in the few pixels under H-alpha, so the paired standard error of
    the difference is of the same order as the difference: a localized feature
    is decided by the size of the loss, not by its ratio to that error.
    """
    _, with_line, samples_with, _, _ = _line_mock(["Hbeta", "Halpha"])
    _, without, samples_without, _, _ = _line_mock(["Hbeta"])
    rows = compare_models(
        {
            "hbeta_only": pointwise_log_likelihood(without, samples_without),
            "both": pointwise_log_likelihood(with_line, samples_with),
        }
    )
    assert rows[0]["name"] == "both"
    assert rows[1]["delta_elpd"] < -100.0


def test_marginalized_lines_with_a_centered_prior_are_refused():
    _, fitter, samples, _, _ = _line_mock(
        ["Hbeta", "Halpha"], eline_prior_type="cloudy", sample=False
    )
    with pytest.raises(ValueError, match="eline_prior_type='flat' only"):
        pointwise_log_likelihood(fitter, samples)


def test_subpackage_exports_and_criteria_on_a_fit():
    model, fitter, _, data, noise = _spectral_fit()
    # The flux is linear in the mass, so the posterior of log10 M is Gaussian to a
    # good approximation: draw from it, centered on the weighted least-squares scale.
    unit = np.asarray(model.predict_spectrum({"sfh_dpl_log_total_mass": 10.0}))
    weight = np.sum((unit / noise) ** 2)
    scale = np.sum(data * unit / noise**2) / weight
    sigma_log = 1.0 / (np.sqrt(weight) * scale * np.log(10.0))
    draws = np.random.default_rng(4).normal(10.0 + np.log10(scale), sigma_log, 400)
    samples = {"sfh_dpl_log_total_mass": draws}
    matrix = tengri.inference.pointwise_log_likelihood(fitter, samples)
    by_waic, by_loo = tengri.inference.waic(matrix), tengri.inference.loo(matrix)
    assert by_waic.elpd == pytest.approx(by_loo.elpd, abs=0.1)
    assert by_loo.p_eff == pytest.approx(1.0, abs=0.3)
    assert tengri.inference.compare_models is compare_models


def test_posterior_without_a_fitter_is_refused():
    from tengri.inference.posterior import Posterior

    bare = Posterior(
        samples={"a": np.zeros(3)}, params={}, method="x", wall_time_s=0.0, diagnostics={}
    )
    with pytest.raises(ValueError, match="needs the posterior's draws and its fitter"):
        bare.pointwise_log_likelihood()


def test_arviz_parity():
    az = pytest.importorskip("arviz")
    log_lik, _, _ = _normal_case(n_draws=2000)
    idata = az.from_dict(log_likelihood={"y": log_lik[None, :, :]})
    ref_waic = az.waic(idata)
    ref_loo = az.loo(idata, pointwise=True)
    ours_waic, ours_loo = waic(log_lik), loo(log_lik)
    assert ours_waic.elpd == pytest.approx(float(ref_waic.elpd_waic), rel=1e-8)
    assert ours_loo.elpd == pytest.approx(float(ref_loo.elpd_loo), abs=1e-6)
    np.testing.assert_allclose(ours_loo.pareto_k, np.asarray(ref_loo.pareto_k), atol=1e-6)
