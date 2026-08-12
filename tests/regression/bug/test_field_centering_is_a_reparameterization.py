# SPDX-License-Identifier: BSD-3-Clause
"""Regression: partial centering must not change the model (#1355, §5).

``field_centering=a`` interpolates between the non-centered field in production
(``a=1``) and the fully centered one, to attack the ``(sigma, xi)`` funnel that
leaves static-HMC R-hat(sigma) at 4.42 while xi's own R-hat is 0.994-0.998.

It is a **reparameterization**: ``a=1`` and ``a<1`` must describe the same
``p(sigma, s)``. That holds only because the latent prior carries its
``-n(1-a) log sigma_s`` normalizer. Drop the normalizer and the sampler still
runs, cleanly, against a different distribution for every ``a`` — the failure
mode ``drw_partial_gp_from_zeta``'s own docstring warns about.

**Compare FIELD densities, never latent ones.** The raw latent-space difference
between ``a`` and ``a=1`` is ``n(1-a) log sigma_s`` and is *supposed to be*: it
is the Jacobian of ``zeta = sigma_s^(1-a) xi``. A probe asserting a flat
latent-space residual inverts the verdict — flat would mean the normalizer had
been dropped. That mistake was made while writing this test, and the wrong
signature reproduced to three decimals across five values of ``a``, which is
what a genuine identity looks like when you have mislabelled it a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.stellar.sfh.gp_sfh import (
    drw_latent_log_prior,
    drw_partial_gp_from_zeta,
    make_log_age_grid,
)

pytestmark = pytest.mark.regression_bug

_N = 16
_TAU_YR = 3.5e8
_SIGMAS = np.geomspace(0.2, 1.5, 9)
_CENTERINGS = (1.0, 0.75, 0.5, 0.25, 0.0)


def _log_age_grid():
    return make_log_age_grid(n_grid=_N)


def _field_and_neg_log_prior(xi, sigma_dex, centering, grid):
    """Field, and its negative log prior converted to FIELD-space density."""
    sigma_s = sigma_dex * np.log(10.0)
    zeta = xi * sigma_s ** (1.0 - centering)
    field = np.asarray(
        drw_partial_gp_from_zeta(zeta, sigma_dex, _TAU_YR, grid, centering=centering)[0]
    )
    nlp_latent = -float(drw_latent_log_prior(zeta, sigma_dex, centering=centering))
    # s = sigma_s^a L zeta, so converting latent -> field density adds a*n*log(sigma_s).
    return field, nlp_latent + centering * _N * np.log(sigma_s)


@pytest.mark.parametrize("centering", _CENTERINGS)
def test_the_field_is_identical_at_every_centering(centering):
    """Same latent content must give the same physical field, or the density
    comparison below would be comparing two different objects."""
    grid = _log_age_grid()
    xi = np.random.default_rng(0).normal(size=_N)
    reference, _ = _field_and_neg_log_prior(xi, 0.6, 1.0, grid)
    field, _ = _field_and_neg_log_prior(xi, 0.6, centering, grid)
    np.testing.assert_allclose(field, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("centering", _CENTERINGS)
def test_the_field_density_does_not_depend_on_centering(centering):
    """The invariant. Scanned over sigma because the normalizer is exactly the
    sigma-dependent piece — a single sigma cannot tell present from absent."""
    grid = _log_age_grid()
    xi = np.random.default_rng(0).normal(size=_N)
    residual = np.array(
        [
            _field_and_neg_log_prior(xi, s, centering, grid)[1]
            - _field_and_neg_log_prior(xi, s, 1.0, grid)[1]
            for s in _SIGMAS
        ]
    )
    assert float(np.abs(residual).max()) < 1e-9, residual


def test_dropping_the_normalizer_would_be_caught():
    """The test above must be able to fail. Re-scoring without the
    ``-n(1-a) log sigma_s`` term has to produce a sigma-dependent residual, or
    the invariant is vacuous and would pass on a broken implementation."""
    grid = _log_age_grid()
    xi = np.random.default_rng(0).normal(size=_N)
    centering = 0.5
    residual = []
    for s in _SIGMAS:
        sigma_s = s * np.log(10.0)
        zeta = xi * sigma_s ** (1.0 - centering)
        # Deliberately omit the normalizer: score zeta as if it were N(0, I)
        # apart from its scaled quadratic form.
        without = 0.5 * float(zeta @ zeta) * sigma_s ** (2 * centering - 2)
        without += centering * _N * np.log(sigma_s)
        reference = _field_and_neg_log_prior(xi, s, 1.0, grid)[1]
        residual.append(without - reference)
    slope = float(np.polyfit(np.log(_SIGMAS), np.asarray(residual), 1)[0])
    # Missing exactly n(1-a) log(sigma_s) => slope in log(sigma) of -n(1-a).
    assert slope == pytest.approx(-_N * (1.0 - centering), rel=1e-6)
