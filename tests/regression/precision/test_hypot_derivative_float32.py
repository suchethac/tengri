# SPDX-License-Identifier: BSD-3-Clause
r"""The noise floor ``hypot(sigma, f_cal |m|)`` differentiates at photometric scales (#3505).

``jnp.hypot`` keeps its value in range by factoring out the larger argument, but its
autodiff divides by that argument squared, :math:`10^{-60}` at a flux uncertainty of
:math:`3\times10^{-31}`, which underflows float32 to zero. The derivative of the
censored likelihood with respect to the model flux was NaN at every real flux scale,
so the float32 Hessian of any fit with an upper limit was NaN.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.noise import censored_neg_log_likelihood, compute_effective_noise
from tengri.utils.scale import hypot

pytestmark = pytest.mark.regression_bug

SIGMA = 3e-31


@pytest.mark.parametrize("flux", [1e-29, -1e-29, 1e-31, 0.0, 3e-31])
def test_hypot_derivatives_float32_match_float64(flux):
    f_cal = 0.01

    def derivs(dtype):
        s = jnp.asarray(SIGMA, dtype)
        fn = lambda m: hypot(s, f_cal * jnp.abs(m))  # noqa: E731
        m = jnp.asarray(flux, dtype)
        return np.array([fn(m), jax.grad(fn)(m), jax.grad(jax.grad(fn))(m)], dtype=np.float64)

    with jax.enable_x64(True):
        ref = derivs(jnp.float64)
    with jax.enable_x64(False):
        got = derivs(jnp.float32)
    assert np.all(np.isfinite(got))
    # Second derivative at m = 0 is f_cal^2 / sigma and is the largest entry.
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-30)


def test_hypot_value_is_jnp_hypot_in_float64():
    with jax.enable_x64(True):
        a = jnp.asarray([3e-31, 1.0, -2.0, 0.0])
        b = jnp.asarray([4e-31, 1e-8, 5.0, 0.0])
        assert np.array_equal(np.asarray(hypot(a, b)), np.asarray(jnp.hypot(a, b)))


@pytest.mark.parametrize("model_sigma_frac", [None, 0.02])
@pytest.mark.parametrize("f_cal", [0.0, 0.01])
def test_censored_likelihood_curvature_finite_and_dtype_stable(model_sigma_frac, f_cal):
    """Energy gradient and Hessian in a standardized amplitude ``m = data * theta``.

    A Hessian with respect to the raw flux is 1/sigma^2 ~ 1e61 and exceeds float32 whatever
    the code does; the amplitude is the O(1) coordinate every fit differentiates in.
    """
    mask = jnp.asarray([1, 0, -1, 0])

    def curv(dtype):
        data = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype) * 1e-29
        noise = data * 0.03

        def energy(theta):
            return censored_neg_log_likelihood(
                data, noise, data * theta, mask, f_cal=f_cal, model_sigma_frac=model_sigma_frac
            )

        theta = jnp.asarray([1.05, 0.97, 1.0, 1.02], dtype)
        return np.asarray(jax.grad(energy)(theta)), np.asarray(jax.hessian(energy)(theta))

    with jax.enable_x64(True):
        g64, h64 = curv(jnp.float64)
    with jax.enable_x64(False):
        g32, h32 = curv(jnp.float32)
    assert np.all(np.isfinite(g32)) and np.all(np.isfinite(h32))
    assert np.max(np.abs(g32 - g64)) <= 1e-3 * np.max(np.abs(g64))
    assert np.max(np.abs(h32 - h64)) <= 1e-3 * np.max(np.abs(h64))


def test_effective_noise_gradient_finite_in_float32():
    with jax.enable_x64(False):
        noise = jnp.asarray([3e-31, 6e-31], jnp.float32)
        model = jnp.asarray([1e-29, 2e-29], jnp.float32)
        jac = jax.jacobian(lambda m: compute_effective_noise(noise, m, 0.02))(model)
        assert np.all(np.isfinite(np.asarray(jac)))
