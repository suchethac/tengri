# SPDX-License-Identifier: BSD-3-Clause
"""The multivariate Student-t marginal over a linear nuisance, against numerical quadrature.

A pixel covariance makes the noise one multivariate Student-t of scale matrix ``C``. The scored
value is a Laplace approximation at the iteratively reweighted mode whose curvature is the
mixing-weight one, ``H = w G^T C^-1 G + Lambda^-1``. These cells integrate the same model over
one calibration coefficient on a fine grid, for a banded and for a full covariance, 30 and 200
data, ``dof`` in {3, 4.5, 10}, with no outlier, one 10-sigma and one 200-sigma outlier. The
reference uses numpy only and shares no code with the scored value.

Measured: the scored value differs from the integral by at most 0.023 nat in every cell. The
expected-information curvature ``(nu+n)/(nu+n+2)`` of the independent-noise marginal is off by
up to 3.2 nat on these cells (0.06 to 1.3 nat with 10-sigma outlier or none, 3.0 to 3.2 nat
with 200 sigma) and does not improve with the number of data, because one weight scales the
whole covariance and the curvature collapses with it; that is why this path takes the
mixing-weight curvature. More than one coefficient is not measured on this path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from tengri.observation.banded import banded_cholesky
from tengri.observation.linear_marginal import marginalize_linear_nuisance

pytestmark = pytest.mark.contract

DOFS = (3.0, 4.5, 10.0)
SIZES = (30, 200)
OUTLIERS = (0.0, 10.0, 200.0)
PRIOR_SIGMA = 0.5

#: Largest tolerated ``|scored - quadrature|`` [nats]: about twice the largest error measured
#: over the cells below (0.023 nat).
TOLERANCE = 0.05


def _covariance(n_data, kind):
    x = np.linspace(-1.0, 1.0, n_data)
    sigma = 0.05 * (1.0 + 0.5 * np.cos(5.0 * x))
    if kind == "banded":
        corr = (
            np.eye(n_data)
            + 0.3 * (np.eye(n_data, k=1) + np.eye(n_data, k=-1))
            + 0.15 * (np.eye(n_data, k=2) + np.eye(n_data, k=-2))
        )
    else:
        lag = np.arange(n_data)
        corr = np.exp(-np.abs(lag[:, None] - lag[None, :]) / 4.0)
    return x, sigma, corr * sigma[:, None] * sigma[None, :]


def _cell(n_data, kind, dof, outlier, seed=0):
    rng = np.random.default_rng(seed)
    x, sigma, cov = _covariance(n_data, kind)
    model0 = 1.0 + 0.3 * np.sin(3.0 * x)
    design = (model0 * x)[:, None]
    noise = np.linalg.cholesky(cov) @ rng.standard_normal(n_data)
    noise = noise / np.sqrt(rng.chisquare(dof) / dof)
    data = model0 + design[:, 0] * 0.03 + noise
    data[n_data // 3] += outlier * sigma[n_data // 3]
    return cov, model0, design, data


def quadrature(cov, model0, design, data, dof):
    """``ln`` of the integral of the multivariate-t likelihood times the prior, one coefficient."""
    n_data = data.shape[0]
    inverse = np.linalg.inv(cov)
    log_det = np.linalg.slogdet(cov)[1]
    resid0, column = data - model0, design[:, 0]
    q_00 = resid0 @ inverse @ resid0
    q_01 = column @ inverse @ resid0
    q_11 = column @ inverse @ column

    def log_integrand(beta):
        chi2 = q_00 - 2.0 * beta * q_01 + beta**2 * q_11
        return (
            scipy.special.gammaln(0.5 * (dof + n_data))
            - scipy.special.gammaln(0.5 * dof)
            - 0.5 * n_data * np.log(dof * np.pi)
            - 0.5 * log_det
            - 0.5 * (dof + n_data) * np.log1p(chi2 / dof)
            - 0.5 * beta**2 / PRIOR_SIGMA**2
            - np.log(PRIOR_SIGMA)
            - 0.5 * np.log(2.0 * np.pi)
        )

    grid = np.linspace(-3.0, 3.0, 40001)
    values = log_integrand(grid)
    inside = np.where(values > values.max() - 40.0)[0]
    low, high = grid[inside[0]], grid[inside[-1]]
    pad = 0.2 * (high - low) + 1e-3
    grid = np.linspace(low - pad, high + pad, 80001)
    values = log_integrand(grid)
    peak = values.max()
    return float(peak + np.log(np.trapezoid(np.exp(values - peak), grid)))


def scored(cov, model0, design, data, dof):
    result = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(model0),
        jnp.asarray(design),
        chol_lower=banded_cholesky(cov),
        prior_mean=jnp.zeros(1),
        prior_precision=jnp.eye(1) / PRIOR_SIGMA**2,
        dof=dof,
    )
    return float(result.log_like)


@pytest.mark.parametrize("outlier", OUTLIERS)
@pytest.mark.parametrize("dof", DOFS)
@pytest.mark.parametrize("n_data", SIZES)
@pytest.mark.parametrize("kind", ("banded", "dense"))
def test_the_covariance_marginal_matches_quadrature_with_and_without_an_outlier(
    kind, n_data, dof, outlier
):
    cov, model0, design, data = _cell(n_data, kind, dof, outlier)
    error = scored(cov, model0, design, data, dof) - quadrature(cov, model0, design, data, dof)
    print(f"QUAD cov={kind} N={n_data} dof={dof} outlier={outlier} error={error:+.5f}")
    assert abs(error) < TOLERANCE
