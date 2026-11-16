# SPDX-License-Identifier: BSD-3-Clause
"""The Student-t mode: two starts, and a visible report when the mode may not be the best.

The posterior of the nuisance coefficients under Student-t noise can have several modes. The
marginal runs the reweighting from two starting points and keeps the better marginal; what it
can detect it reports (``weight_change``, ``start_gap``), and a finished fit warns once. Each
cell below compares the reached mode with the best of 30 optimizer restarts of the same penalized
density. Measured on 10 seeds per cell at 60 data, 20-sigma outliers of random sign, the reached
mode is within 0.1 nat of the best in every seed up to a ratio of nuisance columns to data of
0.2, and in 7 to 9 of 10 seeds at 0.33 with at least 10% outliers (one of those misses carries
no flag).
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


@pytest.mark.parametrize("seed", (12, 14, 17))
def test_a_multimodal_case_reaches_the_better_mode_or_is_flagged(seed):
    """30 data, 10 columns, 30% outliers at 10 sigma: the multimodal regime."""
    cell = _cell(30, 10, 0.3, seed, outlier_sigmas=10.0)
    result = _run(cell, 3.0)
    gap = _best_of_restarts(cell, 3.0) - _objective(np.asarray(result.coef_hat), cell, 3.0)
    assert gap < 0.1 or _flagged(result)


def test_a_second_start_is_taken_when_it_has_the_larger_marginal():
    """30 data, 10 columns, 30% outliers at 20 sigma, seed 4: the starts reach different modes."""
    result = _run(_cell(30, 10, 0.3, 4, outlier_sigmas=20.0), 3.0)
    assert float(result.start_gap) > START_GAP_WARN


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
