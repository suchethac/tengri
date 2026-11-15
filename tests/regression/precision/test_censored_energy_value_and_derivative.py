# SPDX-License-Identifier: BSD-3-Clause
r"""The censored energy keeps its value and has finite derivatives wherever it can (#3505).

``jnp.hypot`` and a plain division by the model-dependent noise floor carry
:math:`1/\sigma^2` in their derivatives (inf in float32 at a flux uncertainty). The
repair must not trade that for a value that goes non-finite where the closed form is
finite (a zero uncertainty with a calibration floor; a prediction of 1e160 beside a
model-error fraction), and it must leave every value the closed form returns exactly as
it was, so the value here is compared with the closed form bit for bit over a grid.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.noise import censored_neg_log_likelihood

pytestmark = pytest.mark.regression_bug


def _closed_form(data, noise, pred, mask, f_cal, dof, msf):
    """The energy as the function computed it before the float32 repair (the pin)."""
    if msf is None:
        sigma_eff = jnp.hypot(noise, f_cal * jnp.abs(pred))
        sigma_lim = noise
    else:
        model_term = jnp.abs(pred)
        sigma_model = jnp.hypot(f_cal * model_term, msf * model_term)
        sigma_eff = jnp.hypot(noise, sigma_model)
        sigma_lim = jnp.hypot(noise, msf * model_term)
    r = (data - pred) / sigma_eff
    if dof is not None:
        e_det = 0.5 * (dof + 1.0) * jnp.log(1.0 + r**2 / dof) + jnp.log(sigma_eff)
    else:
        e_det = 0.5 * r**2 + jnp.log(sigma_eff)
    e_up = -jax.scipy.stats.norm.logcdf((data - pred) / sigma_lim)
    e_lo = -jax.scipy.stats.norm.logcdf((pred - data) / sigma_lim)
    return jnp.sum(jnp.where(mask == 1, e_up, jnp.where(mask == -1, e_lo, e_det)))


# (noise, prediction, data) in units where the data are O(1); a zero uncertainty and a
# calibration floor, tiny and huge predictions, a prediction at zero.
SIGMAS = [0.0, 1e-3, 1.0]
PREDICTIONS = [0.0, 1e-12, 0.8, 1e6, 1e20]
HUGE_F64 = [1e160]
MASKS = [0, 1, -1]
DOFS = [None, 4.0]
MSFS = [None, 0.1]


def _grid(dtype, predictions):
    return [
        (s, p, m, dof, msf)
        for s, p, m, dof, msf in itertools.product(SIGMAS, predictions, MASKS, DOFS, MSFS)
    ]


def _energy(dtype, case, f_cal=0.05):
    s, p, m, dof, msf = case
    data = jnp.asarray([1.0, 1.5], dtype)
    noise = jnp.asarray([s, s], dtype)
    pred = jnp.asarray([p, p if p else 0.3], dtype)
    mask = jnp.asarray([m, 0], jnp.int32)
    f_cal = jnp.asarray(f_cal, dtype)
    msf_arr = None if msf is None else jnp.asarray(msf, dtype)
    got = censored_neg_log_likelihood(
        data, noise, pred, mask, f_cal, dof, model_sigma_frac=msf_arr
    )
    want = _closed_form(data, noise, pred, mask, f_cal, dof, msf_arr)
    return got, want


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_value_equals_the_closed_form_wherever_that_is_finite(x64):
    dtype = jnp.float64 if x64 else jnp.float32
    predictions = PREDICTIONS + (HUGE_F64 if x64 else [])
    checked = 0
    with jax.enable_x64(x64):
        for case in _grid(dtype, predictions):
            got, want = _energy(dtype, case)
            if np.isfinite(float(want)):
                assert float(got) == float(want), f"{case}: {float(got)!r} != {float(want)!r}"
                checked += 1
    assert checked > 60


@pytest.mark.parametrize("case", [(0.0, 0.8, 0, None, None), (0.0, 0.8, 0, 4.0, None)])
def test_zero_uncertainty_with_a_calibration_floor_keeps_its_value(case):
    with jax.enable_x64(True):
        got, want = _energy(jnp.float64, case)
    assert np.isfinite(float(want)) and float(got) == float(want)


def test_a_huge_prediction_beside_a_model_error_fraction_keeps_its_value():
    with jax.enable_x64(True):
        got, want = _energy(jnp.float64, (1.0, 1e160, 0, None, 0.1))
        assert np.isfinite(float(want)) and float(got) == float(want)
    with jax.enable_x64(False):
        got, want = _energy(jnp.float32, (1e-3, 1e20 * 1e-3, 0, None, 0.1))
        assert np.isfinite(float(want)) and float(got) == float(want)


def _theta_derivatives(dtype, mask, dof, msf, f_cal, scale):
    """Gradient and Hessian in the amplitude ``m = data * theta`` at flux scale ``scale``."""
    data = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype) * scale
    noise = data * 0.03
    mask = jnp.asarray(mask, jnp.int32)
    msf_arr = None if msf is None else jnp.asarray(msf, dtype)

    def energy(theta):
        return censored_neg_log_likelihood(
            data, noise, data * theta, mask, f_cal, dof, model_sigma_frac=msf_arr
        )

    theta = jnp.asarray([1.05, 0.97, 1.0, 1.02], dtype)
    return np.asarray(jax.grad(energy)(theta)), np.asarray(jax.hessian(energy)(theta))


@pytest.mark.parametrize("dof", DOFS)
@pytest.mark.parametrize("msf", MSFS)
@pytest.mark.parametrize("mask", [[1, 0, -1, 0], [0, 0, 0, 0]])
def test_float32_derivatives_are_finite_and_match_float64_at_flux_scale(mask, msf, dof):
    with jax.enable_x64(True):
        g64, h64 = _theta_derivatives(jnp.float64, mask, dof, msf, 0.01, 1e-29)
    with jax.enable_x64(False):
        g32, h32 = _theta_derivatives(jnp.float32, mask, dof, msf, 0.01, 1e-29)
    assert np.all(np.isfinite(g32)) and np.all(np.isfinite(h32))
    assert np.max(np.abs(g32 - g64)) <= 2e-4 * np.max(np.abs(g64))
    assert np.max(np.abs(h32 - h64)) <= 2e-4 * np.max(np.abs(h64))


def test_zero_uncertainty_has_a_non_finite_derivative_and_a_finite_value():
    """sigma_obs = 0 is outside the contract; the value survives, the derivative says so."""
    with jax.enable_x64(True):
        data = jnp.asarray([1.0, 1.5])
        noise = jnp.zeros(2)
        mask = jnp.zeros(2, jnp.int32)

        def energy(theta):
            return censored_neg_log_likelihood(data, noise, data * theta, mask, 0.05)

        theta = jnp.asarray([0.9, 1.1])
        assert np.isfinite(float(energy(theta)))
        assert not np.all(np.isfinite(np.asarray(jax.grad(energy)(theta))))
