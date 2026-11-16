# SPDX-License-Identifier: BSD-3-Clause
"""The Student-t mode: two starts, and a visible report when the mode may not be the best.

The posterior of the nuisance coefficients under Student-t noise can have several modes. The
marginal runs the reweighting from two starting points and keeps the better marginal; what it
can detect it reports (``weight_change``, ``start_gap``), and a finished fit warns once. Each
cell below compares the reached mode with the best of optimizer restarts of the same penalized
density, at 60 data and 20-sigma outliers of random sign, 10 seeds per cell unless stated. With at
most 10 columns (ratio 0.17) no seed missed the best mode. With 12 columns (ratio 0.2), 3 degrees
of freedom and 30% outliers the mode missed the best in 2 of 10 seeds, both flagged by the weight
change and the start gap; with 12 columns at 10 degrees of freedom or at 10% outliers, 0 of 10
missed. At 20 columns (ratio 0.33), 0 of 5 missed and all 5 were flagged. At 0.33 with at least
10% outliers and 30 restarts, 7 to 9 of 10 seeds reached the best mode (one of the misses carries
no flag). The warning therefore fires at a ratio of 0.2 and above.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

from tengri.inference.sampler_health import check_student_t_mode
from tengri.observation import linear_marginal
from tengri.observation.linear_marginal import (
    START_GAP_WARN,
    WEIGHT_CHANGE_WARN,
    marginalize_linear_nuisance,
)
from tests.contract import _channel_fixtures as fixtures
from tests.contract.test_student_t_marginal_quadrature import _chebyshev, _log_student_t

pytestmark = pytest.mark.contract

PRIOR_SIGMA = 0.5


def _cell(n_data, n_col, frac, seed, outlier_sigmas=20.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_data)
    model0 = 1.0 + 0.3 * np.sin(3.0 * x)
    sigma = 0.05 * (1.0 + 0.5 * np.cos(5.0 * x))
    design = model0[:, None] * _chebyshev(x, n_col)
    data = model0 + design @ (0.03 * np.ones(n_col)) + sigma * rng.standard_normal(n_data)
    hit = rng.permutation(n_data)[: round(frac * n_data)]
    data[hit] += rng.choice([-1.0, 1.0], size=hit.size) * outlier_sigmas * sigma[hit]
    return model0, sigma, design, data


def _objective(beta, cell, dof):
    model0, sigma, design, data = cell
    log_density = _log_student_t(data - model0 - design @ beta, sigma, dof).sum()
    return log_density - 0.5 * np.sum(beta**2) / PRIOR_SIGMA**2


def _best_of_restarts(cell, dof, n_start=30):
    n_col = cell[2].shape[1]
    rng = np.random.default_rng(1)
    starts = [np.zeros(n_col)] + [rng.normal(0.0, 0.2, n_col) for _ in range(n_start - 1)]
    return max(
        -scipy.optimize.minimize(
            lambda b: -_objective(b, cell, dof), s, method="BFGS", tol=1e-12
        ).fun
        for s in starts
    )


def _run(cell, dof):
    model0, sigma, design, data = cell
    n_col = design.shape[1]
    return marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(model0),
        jnp.asarray(design),
        sigma=jnp.asarray(sigma),
        prior_mean=jnp.zeros(n_col),
        prior_precision=jnp.eye(n_col) / PRIOR_SIGMA**2,
        dof=dof,
    )


def _flagged(result):
    moved = float(result.weight_change) > WEIGHT_CHANGE_WARN
    return moved or float(result.start_gap) > START_GAP_WARN


@pytest.mark.parametrize("seed", range(4))
def test_a_clean_case_reaches_the_best_mode_and_is_not_flagged(seed):
    cell = _cell(60, 3, 0.1, seed)
    result = _run(cell, 3.0)
    assert not _flagged(result)
    gap = _best_of_restarts(cell, 3.0) - _objective(np.asarray(result.coef_hat), cell, 3.0)
    assert gap < 0.1


@pytest.mark.parametrize(
    ("seed", "reaches_best", "flagged"),
    (
        (0, True, False),
        (17, True, True),
        (12, False, True),
        (14, False, True),
        (18, False, False),
    ),
)
def test_a_multimodal_case_reaches_the_best_mode_and_is_flagged_as_measured(
    seed, reaches_best, flagged
):
    """30 data, 10 columns, 30% outliers at 10 sigma: what the mode reached and what warns.

    Seed 18 is a miss that carries no flag: the report does not detect every miss.
    """
    cell = _cell(30, 10, 0.3, seed, outlier_sigmas=10.0)
    result = _run(cell, 3.0)
    gap = _best_of_restarts(cell, 3.0) - _objective(np.asarray(result.coef_hat), cell, 3.0)
    assert (gap < 0.1) == reaches_best
    assert _flagged(result) == flagged


def _weighted_solve(cell, weight):
    model0, sigma, design, data = cell
    precision = design.T @ (design * (weight / sigma**2)[:, None])
    precision = precision + np.eye(design.shape[1]) / PRIOR_SIGMA**2
    return np.linalg.solve(precision, design.T @ (weight * (data - model0) / sigma**2))


def _weights(cell, beta, dof):
    model0, sigma, design, data = cell
    z = (data - model0 - design @ beta) / sigma
    return (dof + 1.0) / (dof + z * z)


def _laplace_at(cell, beta, dof):
    """The documented Laplace marginal at ``beta``, from the Fisher information."""
    model0, sigma, design, data = cell
    n_col = design.shape[1]
    fisher = (dof + 1.0) / (dof + 3.0) * design.T @ (design / sigma[:, None] ** 2)
    fisher = fisher + np.eye(n_col) / PRIOR_SIGMA**2
    return (
        _log_student_t(data - model0 - design @ beta, sigma, dof).sum()
        - 0.5 * np.sum(beta**2) / PRIOR_SIGMA**2
        - 0.5 * n_col * np.log(PRIOR_SIGMA**2)
        - 0.5 * np.linalg.slogdet(fisher)[1]
    )


def _reweighted_mode(cell, dof, start_weight):
    beta = _weighted_solve(cell, start_weight)
    for _ in range(2000):
        beta = _weighted_solve(cell, _weights(cell, beta, dof))
    return beta


def test_a_second_start_is_taken_when_it_has_the_larger_marginal():
    """30 data, 10 columns, 30% outliers at 20 sigma, seed 4: the starts reach different modes."""
    cell, dof = _cell(30, 10, 0.3, 4, outlier_sigmas=20.0), 3.0
    result = _run(cell, dof)
    from_gaussian = _reweighted_mode(cell, dof, np.ones(30))
    from_prior = _reweighted_mode(cell, dof, _weights(cell, np.zeros(10), dof))
    marginals = [_laplace_at(cell, b, dof) for b in (from_gaussian, from_prior)]
    assert float(result.start_gap) > START_GAP_WARN
    assert abs(marginals[0] - marginals[1]) > START_GAP_WARN
    assert float(result.log_like) == pytest.approx(max(marginals), abs=1e-6)
    assert float(result.log_like) > min(marginals) + START_GAP_WARN


def test_the_gaussian_marginal_has_no_report():
    model0, sigma, design, data = _cell(60, 3, 0.0, 0)
    result = marginalize_linear_nuisance(data, model0, design, sigma=sigma)
    assert result.weight_change is None and result.start_gap is None


def _stub_posterior(fitter):
    physical = fixtures.physical_params(fitter)
    return type(
        "Result",
        (),
        {
            "params": {n: physical[n] for n in fitter._free_names if n in physical},
            "diagnostics": {},
        },
    )()


@pytest.mark.parametrize("kind", ("spec_student_t_cal", "spec_student_t_lines"))
def test_a_finished_fit_records_the_report_and_stays_quiet_when_clean(kind, monkeypatch):
    monkeypatch.setattr(linear_marginal, "RELIABLE_COLUMN_RATIO", 1.0)
    fitter = fixtures.fitter_of(kind)
    posterior = _stub_posterior(fitter)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_student_t_mode(fitter, posterior)
    assert posterior.diagnostics["student_t_weight_change"] <= WEIGHT_CHANGE_WARN
    assert posterior.diagnostics["student_t_start_gap"] <= START_GAP_WARN


def test_a_finished_fit_without_a_heavy_tailed_marginal_records_nothing():
    fitter = fixtures.fitter_of("phot_student_t")
    posterior = _stub_posterior(fitter)
    check_student_t_mode(fitter, posterior)
    assert posterior.diagnostics == {}


def test_a_weight_change_above_the_threshold_warns_once_with_the_measured_values(monkeypatch):
    fitter = fixtures.fitter_of("spec_student_t_lines")
    posterior = _stub_posterior(fitter)
    monkeypatch.setattr(linear_marginal, "RELIABLE_COLUMN_RATIO", 1.0)
    monkeypatch.setattr(linear_marginal, "WEIGHT_CHANGE_WARN", -1.0)
    with pytest.warns(RuntimeWarning) as caught:
        check_student_t_mode(fitter, posterior)
    assert len(caught) == 1
    assert caught[0].message.measurements["weight_change"] >= 0.0
    assert "weights still change" in str(caught[0].message)


def test_a_ratio_exactly_at_the_threshold_warns(monkeypatch):
    """A cell at exactly the threshold missed the best mode in 2 of 10 seeds, so it warns."""
    fitter = fixtures.fitter_of("spec_student_t_lines")
    posterior = _stub_posterior(fitter)
    report = fixtures.build_term(fitter)[0].mode_report(
        fixtures.physical_params(fitter), fitter._data_args
    )
    monkeypatch.setattr(
        linear_marginal, "RELIABLE_COLUMN_RATIO", report["n_columns"] / report["n_data"]
    )
    with pytest.warns(RuntimeWarning, match="nuisance columns"):
        check_student_t_mode(fitter, posterior)


def test_too_many_columns_for_the_data_warns():
    """The fixture has 11 columns for 40 data, above the reliable ratio."""
    fitter = fixtures.fitter_of("spec_student_t_lines")
    posterior = _stub_posterior(fitter)
    with pytest.warns(RuntimeWarning) as caught:
        check_student_t_mode(fitter, posterior)
    assert caught[0].message.measurements["column_ratio"] > linear_marginal.RELIABLE_COLUMN_RATIO


@pytest.mark.parametrize(
    "kind", ("spec_student_t_cal", "spec_student_t_lines", "spec_student_t_cal_lines")
)
def test_the_scorer_returns_the_diagnostics_beside_the_unchanged_value(kind):
    from tengri.inference.channels import mode_summary, score

    fitter = fixtures.fitter_of(kind)
    term, physical = fixtures.build_term(fitter)
    preds = term.predictions(physical, fitter._data_args)
    value, results = score(preds, fitter._data_args, term.noise, return_diagnostics=True)
    assert float(value) == float(score(preds, fitter._data_args, term.noise))
    summary = mode_summary(results)
    assert summary["weight_change"] >= 0.0 and summary["n_columns"] >= 1
    jitted = term.mode_report(physical, fitter._data_args)
    assert (jitted["n_columns"], jitted["n_data"]) == (summary["n_columns"], summary["n_data"])
