# SPDX-License-Identifier: BSD-3-Clause
"""The Student-t marginal over a linear nuisance, against direct numerical quadrature.

With Student-t noise the integral over the nuisance coefficients has no closed form. The
scored value is a Laplace approximation at the iteratively reweighted mode; these cells
integrate the same model on a grid and report the error in nats for one and two nuisance
coefficients, ``dof`` in {3, 5, 10, 30}, 20 and 200 data, with and without one outlier. The
reference below uses numpy and scipy only and shares no code with the scored value.

Validity measured here: the scored value differs from the integral by at most 0.46 nat, in
the hardest cell (``dof = 3``, two coefficients, 20 data, one 50-sigma outlier), and by at most
0.04 nat with 200 data. The error is mostly negative, grows with the number of coefficients
and as ``dof`` and the number of data fall, and an outlier changes it by less than 0.06 nat.
More than two coefficients are not measured.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate
import scipy.optimize
import scipy.special

from tengri.inference.likelihoods.terms import calibration_marginal_term, eline_flat_marginal_term

pytestmark = pytest.mark.contract

DOFS = (3.0, 5.0, 10.0, 30.0)
SIZES = (20, 200)
PRIOR_SIGMA = 0.5
OUTLIER_SIGMAS = 50.0

#: Largest tolerated ``|scored - quadrature|`` [nats] per number of coefficients and ``dof``,
#: from the measurement in ``test_the_error_table_stays_inside_its_stated_validity``: 1.5 times
#: the largest error measured over both sizes and both outlier settings.
TOLERANCE = {
    (1, 3.0): 0.26,
    (1, 5.0): 0.17,
    (1, 10.0): 0.10,
    (1, 30.0): 0.04,
    (2, 3.0): 0.70,
    (2, 5.0): 0.42,
    (2, 10.0): 0.22,
    (2, 30.0): 0.09,
}


def _log_student_t(resid, scale, dof):
    z = resid / scale
    return (
        scipy.special.gammaln(0.5 * (dof + 1.0))
        - scipy.special.gammaln(0.5 * dof)
        - 0.5 * np.log(dof * np.pi)
        - np.log(scale)
        - 0.5 * (dof + 1.0) * np.log1p(z * z / dof)
    )


def _chebyshev(x, n_coef):
    return np.stack(
        [np.polynomial.chebyshev.chebval(x, np.eye(n_coef + 1)[k]) for k in range(1, n_coef + 1)],
        axis=1,
    )


def _cell(n_data, n_coef, dof, outlier, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_data)
    model0 = 1.0 + 0.3 * np.sin(3.0 * x)
    sigma = 0.05 * (1.0 + 0.5 * np.cos(5.0 * x))
    basis = _chebyshev(x, n_coef)
    truth = 0.03 * np.ones(n_coef)
    data = model0 * (1.0 + basis @ truth) + sigma * rng.standard_t(dof, size=n_data)
    if outlier:
        data[n_data // 3] += OUTLIER_SIGMAS * sigma[n_data // 3]
    return x, model0, sigma, basis, data


def _log_integrand(beta, model0, sigma, basis, data, dof, prior_sigma):
    resid = data[:, None] - model0[:, None] * (1.0 + basis @ beta.T)
    log_like = _log_student_t(resid, sigma[:, None], dof).sum(axis=0)
    n_coef = beta.shape[1]
    log_prior = (
        -0.5 * np.sum(beta**2, axis=1) / prior_sigma**2
        - n_coef * np.log(prior_sigma)
        - 0.5 * n_coef * np.log(2.0 * np.pi)
    )
    return log_like + log_prior


def quadrature(model0, sigma, basis, data, dof, prior_sigma=PRIOR_SIGMA):
    """``ln`` of the integral of the likelihood times the prior over the coefficients."""
    n_coef = basis.shape[1]

    def neg(beta):
        return -_log_integrand(beta[None, :], model0, sigma, basis, data, dof, prior_sigma)[0]

    mode = scipy.optimize.minimize(neg, np.zeros(n_coef), method="BFGS", tol=1e-12).x
    design = model0[:, None] * basis
    cov = np.linalg.inv(
        design.T @ (design / sigma[:, None] ** 2) + np.eye(n_coef) / prior_sigma**2
    )
    width = 16.0 * np.sqrt(np.diag(cov))
    n_grid = 401 if n_coef == 1 else 241
    axes = [np.linspace(m - w, m + w, n_grid) for m, w in zip(mode, width, strict=True)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n_coef)
    values = np.concatenate(
        [
            _log_integrand(chunk, model0, sigma, basis, data, dof, prior_sigma)
            for chunk in np.array_split(mesh, max(1, mesh.shape[0] // 4000))
        ]
    ).reshape((n_grid,) * n_coef)
    peak = values.max()
    weights = np.exp(values - peak)
    for axis in reversed(range(n_coef)):
        weights = scipy.integrate.trapezoid(weights, axes[axis], axis=axis)
    return float(peak + np.log(weights))


def scored(x, model0, sigma, data, dof, n_coef, prior_sigma=PRIOR_SIGMA):
    """The scored calibration marginal."""
    wave = 4000.0 + 2000.0 * (x + 1.0)
    return float(
        calibration_marginal_term(
            model0, data, sigma, wave, n_poly=n_coef, prior_sigma=prior_sigma, dof=dof
        )
    )


def error_cell(n_data, n_coef, dof, outlier):
    """``scored - quadrature`` [nats] of one cell."""
    x, model0, sigma, basis, data = _cell(n_data, n_coef, dof, outlier)
    return scored(x, model0, sigma, data, dof, n_coef) - quadrature(
        model0, sigma, basis, data, dof
    )


@pytest.mark.parametrize("outlier", (False, True))
@pytest.mark.parametrize("n_data", SIZES)
@pytest.mark.parametrize("dof", DOFS)
@pytest.mark.parametrize("n_coef", (1, 2))
def test_the_error_table_stays_inside_its_stated_validity(n_coef, dof, n_data, outlier):
    error = error_cell(n_data, n_coef, dof, outlier)
    print(f"QUAD n_coef={n_coef} dof={dof} N={n_data} outlier={outlier} error={error:+.5f}")
    assert abs(error) < TOLERANCE[(n_coef, dof)]


def test_a_flat_ish_line_prior_is_scored_by_the_same_quadrature():
    """Marginalized lines: the design is added, not multiplied, and the prior is per line."""
    n_data, dof = 200, 5.0
    rng = np.random.default_rng(3)
    x = np.linspace(-1.0, 1.0, n_data)
    model0 = 1.0 + 0.3 * np.sin(3.0 * x)
    sigma = 0.05 * np.ones(n_data)
    design = np.stack([np.exp(-0.5 * ((x - c) / 0.05) ** 2) for c in (-0.4, 0.3)], axis=1)
    data = model0 + design @ np.array([0.4, 0.2]) + sigma * rng.standard_t(dof, size=n_data)
    data[n_data // 3] += OUTLIER_SIGMAS * sigma[0]
    prior_variance = 1.0

    def log_integrand(beta):
        resid = data[:, None] - model0[:, None] - design @ beta.T
        log_like = _log_student_t(resid, sigma[:, None], dof).sum(axis=0)
        return (
            log_like
            - 0.5 * np.sum(beta**2, axis=1) / prior_variance
            - np.log(2.0 * np.pi * prior_variance)
        )

    mode = scipy.optimize.minimize(
        lambda b: -log_integrand(b[None, :])[0], np.zeros(2), method="BFGS", tol=1e-12
    ).x
    axes = [np.linspace(m - 0.6, m + 0.6, 241) for m in mode]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    values = log_integrand(mesh).reshape(241, 241)
    peak = values.max()
    weights = scipy.integrate.trapezoid(
        scipy.integrate.trapezoid(np.exp(values - peak), axes[1], axis=1), axes[0]
    )
    reference = float(peak + np.log(weights))
    got = float(
        eline_flat_marginal_term(
            model0, data, sigma, design, prior_variance=prior_variance, dof=dof
        )
    )
    print(f"QUAD lines dof={dof} N={n_data} error={got - reference:+.5f}")
    assert abs(got - reference) < 0.04
