# SPDX-License-Identifier: BSD-3-Clause
"""``log_q_h_total`` with no young mass is ``-inf`` with a finite gradient.

A population with zero weight in every young age bin has no ionizing photons, so the
published ``log10 q_h_total`` is ``-inf``. A plain log-sum-exp over the all ``-inf`` terms
returned that value with a NaN derivative with respect to the present-day metallicity, which
every gradient-based fit would propagate. The sum is now masked before the exponential.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import log_q_h_total

pytestmark = pytest.mark.regression_bug


class _StubBackend:
    """Two young bins with a linear log Q_H in age and metallicity (stands in for a table)."""

    _young_idx = jnp.arange(4)

    def _get_log_qh_at(self, log_z, log_age_yr):
        return 47.0 + 0.3 * log_z + 0.2 * (log_age_yr - 7.0)


_AGES = jnp.array([6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
_LOG_Z = -2.0


def _total(weights, log_z=_LOG_Z, fesc=0.0):
    return log_q_h_total(_StubBackend(), _AGES, weights, log_z, fesc)


def test_all_zero_weights_give_minus_inf_value():
    value = float(_total(jnp.zeros(6)))
    assert value == -np.inf, f"no young mass should give log q_h_total = -inf, got {value}"


def test_all_zero_weights_give_finite_gradient_in_metallicity():
    grad = float(jax.grad(lambda lz: _total(jnp.zeros(6), log_z=lz))(_LOG_Z))
    assert np.isfinite(grad), f"d log q_h_total / d log Z is {grad} with no young mass"


def test_all_zero_weights_give_finite_gradient_in_weights_and_fesc():
    g_w = np.asarray(jax.grad(lambda w: _total(w))(jnp.zeros(6)))
    g_f = float(jax.grad(lambda f: _total(jnp.zeros(6), fesc=f))(0.3))
    assert np.all(np.isfinite(g_w)), f"d log q_h_total / d weights is {g_w}"
    assert np.isfinite(g_f), f"d log q_h_total / d fesc is {g_f}"


def test_positive_weights_match_the_linear_sum_and_its_gradient():
    weights = jnp.array([0.0, 1.0e9, 0.0, 2.0e9, 0.0, 0.0])
    value = float(_total(weights))
    lqh = 47.0 + 0.3 * _LOG_Z + 0.2 * (np.asarray(_AGES[:4]) - 7.0)
    manual = float(np.log10(np.sum(np.asarray(weights[:4]) * 10.0**lqh)))
    assert abs(value - manual) < 1e-9, f"log sum {value} != linear sum {manual}"

    grad = float(jax.grad(lambda lz: _total(weights, log_z=lz))(_LOG_Z))
    h = 1e-6
    fd = float(_total(weights, log_z=_LOG_Z + h)) - float(_total(weights, log_z=_LOG_Z - h))
    fd /= 2.0 * h
    assert abs(grad - fd) < 1e-6 * max(1.0, abs(fd)), f"AD {grad} vs FD {fd}"
